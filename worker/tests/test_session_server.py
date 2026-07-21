"""Session lifecycle, real artifact, and framed-server behavior."""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

import cloudpoint_worker.session as session_module
from cloudpoint_worker.cpc import read_cpc
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.protocol.framing import (
    read_json_frame,
    write_json_frame,
)
from cloudpoint_worker.protocol.schema import (
    BeginSessionPayload,
    ConfigurePayload,
    ResumeCheckpoint,
    decode_command,
    decode_event,
)
from cloudpoint_worker.server import WorkerServer
from cloudpoint_worker.session import PersistedFrame, SessionRunner, SessionState

PROJECT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
CONFIG = ConfigurePayload(8, 32, 8, 1, 4, 1.5, 0.01)


class FakeModel:
    """Tiny deterministic scene predictor at the production model boundary."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def infer_direct(self, images: mx.array, scale_frames: int = 8) -> object:
        self.calls.append((int(images.shape[0]), scale_frames))
        frames = []
        height, width = (int(images.shape[1]), int(images.shape[2]))
        for index in range(int(images.shape[0])):
            confidence = np.zeros((height, width), dtype=np.float32)
            confidence[height // 2, width // 2] = 3.0 + index
            pose = np.zeros(9, dtype=np.float32)
            pose[6] = 1.0
            pose[7:] = 1.0
            camera_to_world = np.eye(4, dtype=np.float32)
            camera_to_world[0, 3] = index * 0.1
            frames.append(
                SimpleNamespace(
                    depth=np.full((height, width), 2.0, dtype=np.float32),
                    confidence=confidence,
                    pose_encoding=pose,
                    intrinsics=np.array(
                        [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
                        dtype=np.float32,
                    ),
                    camera_to_world=camera_to_world,
                )
            )
        return SimpleNamespace(frames=tuple(frames))


class AllocationFailingModel(FakeModel):
    def infer_direct(self, images: mx.array, scale_frames: int = 8) -> object:
        raise MemoryError("synthetic allocation failure")


def _manifest(project_id: uuid.UUID = PROJECT_ID) -> dict[str, object]:
    return {
        "formatVersion": 2,
        "projectID": str(project_id),
        "createdAt": "2026-07-20T00:00:00.0Z",
        "updatedAt": "2026-07-20T00:00:00.0Z",
        "engineConfiguration": {
            "scaleFrames": 8,
            "windowSize": 32,
            "windowOverlap": 8,
            "keyframeInterval": 1,
            "cameraRefinementIterations": 4,
            "confidenceThreshold": 1.5,
            "voxelSize": 0.01,
        },
        "frames": [],
        "completedWindows": [],
        "sessionState": {
            "phase": "empty",
            "isCapturing": False,
            "capturedCount": 0,
            "queuedCount": 0,
            "processedCount": 0,
            "failedCount": 0,
            "currentWindow": None,
        },
    }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "Smoke.cloudpoint"
    root.mkdir()
    for name in ("Frames", "Predictions", "Points", "Logs"):
        (root / name).mkdir()
    (root / "Manifest.json").write_text(
        json.dumps(_manifest(), sort_keys=True), encoding="utf-8"
    )
    for index, color in enumerate(((220, 30, 40), (30, 180, 220))):
        Image.new("RGB", (42, 28), color).save(root / "Frames" / f"{index:08d}.jpg")
    return root


def _frame(index: int) -> PersistedFrame:
    return PersistedFrame(index, index / 2, f"Frames/{index:08d}.jpg")


def _install_completed_window(
    project: Path,
    indices: tuple[int, ...],
    *,
    window_overrides: dict[str, object] | None = None,
    artifact_overrides: dict[int, dict[str, object]] | None = None,
) -> dict[Path, bytes]:
    manifest = _manifest()
    for index in indices:
        frame_path = project / f"Frames/{index:08d}.jpg"
        if not frame_path.exists():
            Image.new("RGB", (42, 28), (20 + index, 40, 60)).save(frame_path)
    manifest["frames"] = [
        {
            "index": index,
            "sourceTimestamp": index / 2,
            "relativePath": f"Frames/{index:08d}.jpg",
        }
        for index in indices
    ]
    artifacts: list[dict[str, object]] = []
    for index in indices:
        paths = (
            f"Predictions/{index:08d}.depth-f16",
            f"Predictions/{index:08d}.confidence-f16",
            f"Predictions/{index:08d}.geometry.json",
        )
        for relative in paths:
            (project / relative).write_bytes(f"committed-{relative}".encode())
        artifact: dict[str, object] = {
            "frameIndex": index,
            "windowIndex": 0,
            "depthRelativePath": paths[0],
            "confidenceRelativePath": paths[1],
            "geometryRelativePath": paths[2],
            "durationSeconds": 0,
        }
        if artifact_overrides and index in artifact_overrides:
            artifact.update(artifact_overrides[index])
        artifacts.append(artifact)
    point_path = project / "Points/window-00000000.cpc"
    point_path.write_bytes(b"committed-cpc")
    window: dict[str, object] = {
        "index": 0,
        "inferenceFrameStart": indices[0],
        "frameStart": indices[0],
        "frameEnd": indices[-1],
        "pointChunkRelativePath": "Points/window-00000000.cpc",
        "alignmentRowMajor": [
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ],
        "lastProcessedFrameIndex": indices[-1],
        "inlierCount": len(indices),
        "durationSeconds": 0,
        "frameArtifacts": artifacts,
    }
    if window_overrides:
        window.update(window_overrides)
    manifest["completedWindows"] = [window]
    (project / "Manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        path.relative_to(project): path.read_bytes()
        for path in (
            *sorted((project / "Predictions").iterdir()),
            point_path,
        )
    }


def test_direct_session_writes_predictions_and_one_nonempty_cpc(project: Path) -> None:
    events: list[object] = []
    model = FakeModel()
    manifest_before = (project / "Manifest.json").read_bytes()
    runner = SessionRunner(project, model, events.append, project_id=PROJECT_ID)

    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    runner.enqueue(_frame(0))
    runner.enqueue(_frame(1))
    runner.finish_input()
    runner.process()

    assert runner.state == SessionState.COMPLETED
    assert model.calls == [(2, 2)]
    assert [event.type for event in events] == [
        "frameStarted",
        "frameCompleted",
        "frameStarted",
        "frameCompleted",
        "windowCompleted",
        "sessionCompleted",
    ]
    assert len(list((project / "Predictions").glob("*.geometry.json"))) == 2
    chunk = read_cpc(project / "Points/window-00000000.cpc")
    assert chunk.point_count == 2
    assert (chunk.descriptor.frame_start, chunk.descriptor.frame_end) == (0, 1)
    assert (project / "Manifest.json").read_bytes() == manifest_before
    for event in events:
        decode_event(json.loads(write_event_json(event)))


def write_event_json(event: object) -> str:
    stream = io.BytesIO()
    write_json_frame(stream, event)
    stream.seek(0)
    return json.dumps(read_json_frame(stream))


def test_more_than_thirty_two_frames_fails_recoverably_without_outputs(
    project: Path,
) -> None:
    for index in range(2, 33):
        Image.new("RGB", (42, 28), (index, index, index)).save(
            project / "Frames" / f"{index:08d}.jpg"
        )
    events: list[object] = []
    runner = SessionRunner(project, FakeModel(), events.append, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    for index in range(32):
        runner.enqueue(_frame(index))

    with pytest.raises(WorkerFault, match="WINDOWING_UNAVAILABLE") as raised:
        runner.enqueue(_frame(32))

    assert raised.value.recoverable is True
    assert runner.state == SessionState.FAILED
    assert runner.queued_frames == 32
    with pytest.raises(WorkerFault, match="INVALID_STATE"):
        runner.finish_input()
    with pytest.raises(WorkerFault, match="INVALID_STATE"):
        runner.process()
    assert events == []
    assert not list((project / "Predictions").iterdir())
    assert not list((project / "Points").iterdir())


def test_memory_failure_maps_to_recoverable_allocation_fault(project: Path) -> None:
    runner = SessionRunner(
        project, AllocationFailingModel(), lambda _: None, project_id=PROJECT_ID
    )
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    runner.enqueue(_frame(0))
    runner.finish_input()

    with pytest.raises(WorkerFault, match="ALLOCATION_FAILED") as raised:
        runner.process()

    assert raised.value.recoverable is True
    assert runner.state == SessionState.FAILED


def test_cancel_after_reduction_cleans_only_uncommitted_outputs(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    runner = SessionRunner(project, FakeModel(), events.append, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    runner.enqueue(_frame(0))
    runner.enqueue(_frame(1))
    runner.finish_input()
    real_reduce = session_module.reduce_vertices

    def cancel_after_reduce(*args: object, **kwargs: object) -> object:
        result = real_reduce(*args, **kwargs)
        runner.cancel()
        return result

    monkeypatch.setattr(session_module, "reduce_vertices", cancel_after_reduce)

    runner.process()

    assert runner.state == SessionState.CANCELLED
    assert events[-1].type == "cancelled"
    assert not any(event.type == "windowCompleted" for event in events)
    assert not list((project / "Predictions").iterdir())
    assert not list((project / "Points").iterdir())


def test_cancel_after_cpc_promotion_prevents_completion_and_removes_owned_chunk(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    runner = SessionRunner(project, FakeModel(), events.append, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    runner.enqueue(_frame(0))
    runner.finish_input()
    real_write_cpc = session_module.write_cpc

    def cancel_after_write(*args: object, **kwargs: object) -> object:
        result = real_write_cpc(*args, **kwargs)
        runner.cancel()
        return result

    monkeypatch.setattr(session_module, "write_cpc", cancel_after_write)

    runner.process()

    assert runner.state == SessionState.CANCELLED
    assert events[-1].type == "cancelled"
    assert not any(event.type == "windowCompleted" for event in events)
    assert not list((project / "Predictions").iterdir())
    assert not list((project / "Points").iterdir())


def test_failed_frame_transaction_never_removes_preexisting_canonical_output(
    project: Path,
) -> None:
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    runner.enqueue(_frame(0))
    runner.finish_input()
    preexisting = project / "Predictions/00000000.depth-f16"
    preexisting.write_bytes(b"belongs-to-another-writer")

    with pytest.raises(WorkerFault, match="OUTPUT_ALREADY_EXISTS"):
        runner.process()

    assert preexisting.read_bytes() == b"belongs-to-another-writer"


def test_cancel_cleanup_preserves_output_replaced_after_worker_promotion(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement = b"replacement-owned-by-another-writer"
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    runner.enqueue(_frame(0))
    runner.finish_input()
    real_filter = session_module.filter_and_reduce_points

    def replace_after_promotion(*args: object, **kwargs: object) -> object:
        depth = project / "Predictions/00000000.depth-f16"
        depth.unlink()
        depth.write_bytes(replacement)
        runner.cancel()
        return real_filter(*args, **kwargs)

    monkeypatch.setattr(
        session_module, "filter_and_reduce_points", replace_after_promotion
    )

    runner.process()

    assert runner.state == SessionState.CANCELLED
    assert (project / "Predictions/00000000.depth-f16").read_bytes() == replacement


def test_replay_is_context_only_and_never_overwrites_committed_outputs(
    project: Path,
) -> None:
    original = _install_completed_window(project, (0, 1))
    Image.new("RGB", (42, 28), (90, 80, 70)).save(project / "Frames/00000002.jpg")
    events: list[object] = []
    runner = SessionRunner(project, FakeModel(), events.append, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(ResumeCheckpoint(1, 0, 1)))
    runner.enqueue(_frame(0))
    runner.enqueue(_frame(1))
    with pytest.raises(WorkerFault, match="WINDOWING_UNAVAILABLE") as raised:
        runner.enqueue(_frame(2))
    assert raised.value.recoverable is True
    assert runner.state == SessionState.FAILED
    with pytest.raises(WorkerFault, match="INVALID_STATE"):
        runner.finish_input()
    with pytest.raises(WorkerFault, match="INVALID_STATE"):
        runner.process()

    assert runner.processed_frames == 0
    assert events == []
    assert not (project / "Points/window-00000001.cpc").exists()
    for relative, contents in original.items():
        assert (project / relative).read_bytes() == contents


def test_replay_requires_every_exact_manifest_descriptor(project: Path) -> None:
    _install_completed_window(project, (0, 1, 2))
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(ResumeCheckpoint(2, 0, 1)))
    runner.enqueue(_frame(0))

    with pytest.raises(WorkerFault, match="REPLAY_ORDER_VIOLATION"):
        runner.enqueue(_frame(2))


def test_replay_exact_order_allows_legal_source_index_gaps(project: Path) -> None:
    original = _install_completed_window(project, (0, 2))
    events: list[object] = []
    runner = SessionRunner(project, FakeModel(), events.append, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(ResumeCheckpoint(2, 0, 1)))
    runner.enqueue(_frame(0))
    runner.enqueue(_frame(2))
    runner.finish_input()
    runner.process()

    assert [event.type for event in events] == ["sessionCompleted"]
    for relative, contents in original.items():
        assert (project / relative).read_bytes() == contents


def _command(kind: str, payload: dict[str, object], identifier: int) -> object:
    return decode_command(
        {
            "protocolVersion": 1,
            "id": str(uuid.UUID(int=identifier)),
            "projectId": str(PROJECT_ID),
            "type": kind,
            "payload": payload,
        }
    )


def test_server_acknowledges_each_command_once_and_hello_heartbeat_is_immediate(
    project: Path,
) -> None:
    output: list[object] = []
    server = WorkerServer(
        project,
        model_loader=lambda: FakeModel(),
        event_sink=output.append,
        heartbeat_interval=60,
    )
    hello = _command(
        "hello", {"clientVersion": "test", "supportedProtocolVersions": [1]}, 1
    )
    configure = _command(
        "configure",
        {
            "scaleFrames": 8,
            "windowSize": 32,
            "windowOverlap": 8,
            "keyframeInterval": 1,
            "cameraRefinementIterations": 4,
            "confidenceThreshold": 1.5,
            "voxelSize": 0.01,
        },
        2,
    )
    begin = _command("beginSession", {"resumeCheckpoint": None}, 3)
    enqueue = _command(
        "enqueueFrame",
        {"frameIndex": 0, "sourceTimestamp": 0, "relativePath": "Frames/00000000.jpg"},
        4,
    )
    finish = _command("finishInput", {}, 5)

    assert server.handle(hello) is True
    server.wait_for_model(timeout=10)
    for command in (configure, begin, enqueue, finish):
        assert server.handle(command) is True
    server.wait_for_idle(timeout=10)
    server.close()

    assert [event.type for event in output[:5]] == [
        "ack",
        "heartbeat",
        "modelProgress",
        "modelProgress",
        "ready",
    ]
    for command in (hello, configure, begin, enqueue, finish):
        responses = [
            event
            for event in output
            if event.type in {"ack", "error"} and event.command_id == command.id
        ]
        assert len(responses) == 1
    assert output[-1].type == "sessionCompleted"


def test_server_rejects_first_descriptor_beyond_direct_cap_without_ack(
    project: Path,
) -> None:
    Image.new("RGB", (42, 28), (10, 20, 30)).save(project / "Frames/00000002.jpg")
    output: list[object] = []
    server = WorkerServer(
        project,
        model_loader=lambda: FakeModel(),
        event_sink=output.append,
        heartbeat_interval=60,
    )
    hello = _command(
        "hello", {"clientVersion": "test", "supportedProtocolVersions": [1]}, 51
    )
    configure = _command(
        "configure",
        {
            "scaleFrames": 2,
            "windowSize": 2,
            "windowOverlap": 1,
            "keyframeInterval": 1,
            "cameraRefinementIterations": 4,
            "confidenceThreshold": 1.5,
            "voxelSize": 0.01,
        },
        52,
    )
    begin = _command("beginSession", {"resumeCheckpoint": None}, 53)
    frames = tuple(
        _command(
            "enqueueFrame",
            {
                "frameIndex": index,
                "sourceTimestamp": index / 2,
                "relativePath": f"Frames/{index:08d}.jpg",
            },
            54 + index,
        )
        for index in range(3)
    )
    server.handle(hello)
    server.wait_for_model(timeout=10)
    for command in (configure, begin, *frames):
        server.handle(command)
    server.close()

    final_responses = [
        event
        for event in output
        if event.type in {"ack", "error"} and event.command_id == frames[-1].id
    ]
    assert [event.type for event in final_responses] == ["error"]
    assert final_responses[0].payload.code == "WINDOWING_UNAVAILABLE"
    assert final_responses[0].payload.details["frameCount"] == 2


def test_heartbeat_continues_while_model_is_loading(project: Path) -> None:
    output: list[object] = []

    def slow_loader() -> FakeModel:
        time.sleep(0.09)
        return FakeModel()

    server = WorkerServer(
        project,
        model_loader=slow_loader,
        event_sink=output.append,
        heartbeat_interval=0.02,
    )
    hello = _command(
        "hello", {"clientVersion": "test", "supportedProtocolVersions": [1]}, 91
    )

    started = time.monotonic()
    server.handle(hello)
    handle_duration = time.monotonic() - started
    server.wait_for_model(timeout=10)
    server.close()

    assert handle_duration < 0.05
    ready_index = next(
        index for index, event in enumerate(output) if event.type == "ready"
    )
    loading_heartbeats = [
        event for event in output[:ready_index] if event.type == "heartbeat"
    ]
    assert len(loading_heartbeats) >= 3
    assert all(event.payload.busy for event in loading_heartbeats)


def test_shutdown_ack_remains_live_while_model_loader_is_blocked(
    project: Path,
) -> None:
    output: list[object] = []
    loading = threading.Event()
    release = threading.Event()

    def blocked_loader() -> FakeModel:
        loading.set()
        assert release.wait(timeout=5)
        return FakeModel()

    server = WorkerServer(
        project,
        model_loader=blocked_loader,
        event_sink=output.append,
        heartbeat_interval=60,
    )
    hello = _command(
        "hello", {"clientVersion": "test", "supportedProtocolVersions": [1]}, 92
    )
    shutdown = _command("shutdown", {}, 93)

    assert server.handle(hello) is True
    assert loading.wait(timeout=5)
    started = time.monotonic()
    assert server.handle(shutdown) is False
    shutdown_duration = time.monotonic() - started

    assert shutdown_duration < 0.05
    assert output[-1].type == "ack"
    assert output[-1].command_id == shutdown.id
    release.set()
    server.close()
    assert not any(event.type == "ready" for event in output)


def test_shutdown_during_reduction_acks_then_cancels_and_cleans_outputs(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output: list[object] = []
    reducing = threading.Event()
    release = threading.Event()
    real_reduce = session_module.reduce_vertices

    def blocked_reduce(*args: object, **kwargs: object) -> object:
        reducing.set()
        assert release.wait(timeout=5)
        return real_reduce(*args, **kwargs)

    monkeypatch.setattr(session_module, "reduce_vertices", blocked_reduce)
    server = WorkerServer(
        project,
        model_loader=lambda: FakeModel(),
        event_sink=output.append,
        heartbeat_interval=60,
    )
    commands = (
        _command(
            "hello",
            {"clientVersion": "test", "supportedProtocolVersions": [1]},
            94,
        ),
        _command(
            "configure",
            {
                "scaleFrames": 8,
                "windowSize": 32,
                "windowOverlap": 8,
                "keyframeInterval": 1,
                "cameraRefinementIterations": 4,
                "confidenceThreshold": 1.5,
                "voxelSize": 0.01,
            },
            95,
        ),
        _command("beginSession", {"resumeCheckpoint": None}, 96),
        _command(
            "enqueueFrame",
            {
                "frameIndex": 0,
                "sourceTimestamp": 0,
                "relativePath": "Frames/00000000.jpg",
            },
            97,
        ),
        _command("finishInput", {}, 98),
    )
    server.handle(commands[0])
    server.wait_for_model(timeout=10)
    for command in commands[1:]:
        server.handle(command)
    assert reducing.wait(timeout=5)
    shutdown = _command("shutdown", {}, 99)

    assert server.handle(shutdown) is False
    shutdown_ack = next(
        index
        for index, event in enumerate(output)
        if event.type == "ack" and event.command_id == shutdown.id
    )
    release.set()
    server.wait_for_idle(timeout=10)
    server.close()

    cancelled = next(
        index for index, event in enumerate(output) if event.type == "cancelled"
    )
    assert cancelled > shutdown_ack
    assert not any(event.type == "windowCompleted" for event in output)
    assert not list((project / "Predictions").iterdir())
    assert not list((project / "Points").iterdir())


def test_cancel_then_shutdown_does_not_duplicate_cancelled(project: Path) -> None:
    output: list[object] = []
    server = WorkerServer(
        project,
        model_loader=lambda: FakeModel(),
        event_sink=output.append,
        heartbeat_interval=60,
    )
    commands = (
        _command(
            "hello",
            {"clientVersion": "test", "supportedProtocolVersions": [1]},
            101,
        ),
        _command(
            "configure",
            {
                "scaleFrames": 8,
                "windowSize": 32,
                "windowOverlap": 8,
                "keyframeInterval": 1,
                "cameraRefinementIterations": 4,
                "confidenceThreshold": 1.5,
                "voxelSize": 0.01,
            },
            102,
        ),
        _command("beginSession", {"resumeCheckpoint": None}, 103),
        _command("cancel", {}, 104),
        _command("shutdown", {}, 105),
    )
    server.handle(commands[0])
    server.wait_for_model(timeout=10)
    for command in commands[1:]:
        server.handle(command)
    server.close()

    cancel_position = next(
        index
        for index, event in enumerate(output)
        if event.type == "ack" and event.command_id == commands[3].id
    )
    cancelled_positions = [
        index for index, event in enumerate(output) if event.type == "cancelled"
    ]
    assert len(cancelled_positions) == 1
    assert cancelled_positions[0] > cancel_position


def test_duplicate_command_id_receives_one_error_without_repeating_mutation(
    project: Path,
) -> None:
    output: list[object] = []
    server = WorkerServer(
        project,
        model_loader=lambda: FakeModel(),
        event_sink=output.append,
        heartbeat_interval=60,
    )
    hello = _command(
        "hello", {"clientVersion": "test", "supportedProtocolVersions": [1]}, 201
    )
    server.handle(hello)
    server.wait_for_model(timeout=10)
    server.handle(hello)
    server.close()

    owned = [
        event
        for event in output
        if event.type in {"ack", "error"} and event.command_id == hello.id
    ]
    assert [event.type for event in owned] == ["ack", "error"]
    assert owned[-1].payload.code == "DUPLICATE_COMMAND_ID"
    assert sum(event.type == "ready" for event in output) == 1


def test_shutdown_is_acknowledged_after_model_load_failure(project: Path) -> None:
    output: list[object] = []

    def unavailable() -> FakeModel:
        raise WorkerFault("MODEL_UNAVAILABLE", "missing", True)

    server = WorkerServer(
        project,
        model_loader=unavailable,
        event_sink=output.append,
        heartbeat_interval=60,
    )
    hello = _command(
        "hello", {"clientVersion": "test", "supportedProtocolVersions": [1]}, 211
    )
    shutdown = _command("shutdown", {}, 212)
    server.handle(hello)
    server.wait_for_model(timeout=10)

    should_continue = server.handle(shutdown)
    server.close()

    assert should_continue is False
    assert any(
        event.type == "ack" and event.command_id == shutdown.id for event in output
    )


def test_framed_server_stdout_contains_only_protocol_frames(project: Path) -> None:
    source = io.BytesIO()
    write_json_frame(
        source,
        {
            "protocolVersion": 1,
            "id": str(uuid.UUID(int=1)),
            "projectId": str(PROJECT_ID),
            "type": "hello",
            "payload": {"clientVersion": "test", "supportedProtocolVersions": [1]},
        },
    )
    write_json_frame(
        source,
        {
            "protocolVersion": 1,
            "id": str(uuid.UUID(int=2)),
            "projectId": str(PROJECT_ID),
            "type": "shutdown",
            "payload": {},
        },
    )
    source.seek(0)
    destination = io.BytesIO()
    server = WorkerServer(
        project,
        model_loader=lambda: (time.sleep(0.09), FakeModel())[1],
        input_stream=source,
        output_stream=destination,
        heartbeat_interval=60,
    )

    assert server.serve() == 0

    destination.seek(0)
    decoded: list[dict[str, object]] = []
    while destination.tell() < len(destination.getvalue()):
        decoded.append(read_json_frame(destination))
    assert [value["type"] for value in decoded[:2]] == ["ack", "heartbeat"]
    assert decoded[-1]["type"] == "ack"
    assert decoded[-1]["payload"]["command"] == "shutdown"
    assert not any(value["type"] == "ready" for value in decoded)
    assert all(decode_event(value) for value in decoded)


def test_framed_server_clean_eof_returns_success(project: Path) -> None:
    output: list[object] = []
    server = WorkerServer(
        project,
        model_loader=lambda: FakeModel(),
        input_stream=io.BytesIO(),
        event_sink=output.append,
        heartbeat_interval=60,
    )

    assert server.serve() == 0
    assert output == []


def test_framed_server_malformed_input_returns_protocol_exit(project: Path) -> None:
    output: list[object] = []
    source = io.BytesIO((1).to_bytes(4, "big") + b"{")
    server = WorkerServer(
        project,
        model_loader=lambda: FakeModel(),
        input_stream=source,
        event_sink=output.append,
        heartbeat_interval=60,
    )

    assert server.serve() == 3
    assert [event.type for event in output] == ["error"]
    assert output[0].payload.code == "INVALID_JSON"


def test_session_rejects_symlinked_frame(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (42, 28), "red").save(outside)
    link = project / "Frames/00000009.jpg"
    link.symlink_to(outside)
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))

    with pytest.raises(WorkerFault, match="PATH_OUTSIDE_PROJECT"):
        runner.enqueue(_frame(9))


def test_manifest_read_cannot_be_redirected_after_path_validation(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside-manifest.json"
    outside.write_text("{}", encoding="utf-8")
    original_read_text = Path.read_text

    def swap_before_path_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == project / "Manifest.json":
            path.rename(project / "Manifest.original.json")
            path.symlink_to(outside)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", swap_before_path_read)
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)

    runner.begin(BeginSessionPayload(None))

    assert runner.project_id == PROJECT_ID
    assert not (project / "Manifest.original.json").exists()


def test_frame_decode_uses_open_descriptor_across_path_swap(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (42, 28), (10, 240, 10)).save(outside)
    canonical = project / "Frames/00000000.jpg"
    original_preprocess = session_module.preprocess_image
    observed_red_minus_green: list[float] = []

    def swap_before_decode(path: Path) -> object:
        canonical.rename(project / "Frames/00000000.original.jpg")
        canonical.symlink_to(outside)
        result = original_preprocess(path)
        observed_red_minus_green.append(
            float(np.mean(result.rgb[..., 0]) - np.mean(result.rgb[..., 1]))
        )
        return result

    monkeypatch.setattr(session_module, "preprocess_image", swap_before_decode)
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    runner.enqueue(_frame(0))
    runner.finish_input()

    runner.process()

    assert observed_red_minus_green[0] > 0.5


def test_orphan_cleanup_holds_directory_descriptor_across_path_swap(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orphan = project / "Predictions/00000009.depth-f16"
    orphan.write_bytes(b"orphan")
    outside = tmp_path / "outside-predictions"
    outside.mkdir()
    outside_file = outside / orphan.name
    outside_file.write_bytes(b"must-survive")
    original_iterdir = Path.iterdir

    def swap_before_iteration(path: Path):  # type: ignore[no-untyped-def]
        if path == project / "Predictions":
            path.rename(project / "Predictions.original")
            path.symlink_to(outside, target_is_directory=True)
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", swap_before_iteration)
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)

    runner.begin(BeginSessionPayload(None))

    assert outside_file.read_bytes() == b"must-survive"
    assert not orphan.exists()


@pytest.mark.parametrize(
    ("window_overrides", "artifact_overrides"),
    [
        ({"alignmentRowMajor": [1.0] * 15}, None),
        ({"alignmentRowMajor": [1.0] * 15 + [float("inf")]}, None),
        ({"alignmentRowMajor": [1.0] * 15 + [10**400]}, None),
        ({"inferenceFrameStart": 1}, None),
        ({"lastProcessedFrameIndex": 0}, None),
        ({"inlierCount": -1}, None),
        ({"durationSeconds": -0.1}, None),
        ({"durationSeconds": 10**400}, None),
        (None, {0: {"windowIndex": 1}}),
        (None, {0: {"durationSeconds": float("nan")}}),
        (None, {0: {"durationSeconds": 10**400}}),
    ],
)
def test_recovery_rejects_malformed_window_metadata(
    project: Path,
    window_overrides: dict[str, object] | None,
    artifact_overrides: dict[int, dict[str, object]] | None,
) -> None:
    _install_completed_window(
        project,
        (0, 1),
        window_overrides=window_overrides,
        artifact_overrides=artifact_overrides,
    )
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)

    with pytest.raises(WorkerFault, match="PROJECT_INVALID_MANIFEST"):
        runner.begin(BeginSessionPayload(ResumeCheckpoint(1, 0, 1)))


def test_recovery_rejects_huge_frame_timestamp_as_structured_manifest_fault(
    project: Path,
) -> None:
    _install_completed_window(project, (0, 1))
    manifest = json.loads((project / "Manifest.json").read_text(encoding="utf-8"))
    manifest["frames"][0]["sourceTimestamp"] = 10**400
    (project / "Manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)

    with pytest.raises(WorkerFault, match="PROJECT_INVALID_MANIFEST"):
        runner.begin(BeginSessionPayload(ResumeCheckpoint(1, 0, 1)))


@pytest.mark.parametrize("artifact_indices", [(1, 0), (0,)])
def test_recovery_rejects_reordered_or_incomplete_artifact_windows(
    project: Path, artifact_indices: tuple[int, ...]
) -> None:
    _install_completed_window(project, (0, 1))
    manifest = json.loads((project / "Manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["completedWindows"][0]["frameArtifacts"]
    manifest["completedWindows"][0]["frameArtifacts"] = [
        artifacts[index] for index in artifact_indices
    ]
    (project / "Manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)

    with pytest.raises(WorkerFault, match="PROJECT_INVALID_MANIFEST"):
        runner.begin(BeginSessionPayload(ResumeCheckpoint(1, 0, 1)))
