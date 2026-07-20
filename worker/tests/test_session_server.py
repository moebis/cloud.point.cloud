"""Session lifecycle, real artifact, and framed-server behavior."""

from __future__ import annotations

import io
import json
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
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    for index in range(33):
        runner.enqueue(_frame(index))
    runner.finish_input()

    with pytest.raises(WorkerFault, match="WINDOWING_UNAVAILABLE") as raised:
        runner.process()

    assert raised.value.recoverable is True
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


def test_failed_frame_transaction_removes_owned_canonical_outputs(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_after_first_output(*_args: object, **_kwargs: object) -> object:
        (project / "Predictions/00000000.depth-f16").write_bytes(b"orphan")
        raise WorkerFault("INVALID_MODEL_OUTPUT", "synthetic output fault", False)

    monkeypatch.setattr(session_module, "write_frame_outputs", fail_after_first_output)
    runner = SessionRunner(project, FakeModel(), lambda _: None, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(None))
    runner.enqueue(_frame(0))
    runner.finish_input()

    with pytest.raises(WorkerFault, match="INVALID_MODEL_OUTPUT"):
        runner.process()

    assert not list((project / "Predictions").iterdir())


def test_replay_is_context_only_and_never_overwrites_committed_outputs(
    project: Path,
) -> None:
    manifest = _manifest()
    manifest["frames"] = [
        {"index": 0, "sourceTimestamp": 0, "relativePath": "Frames/00000000.jpg"},
        {"index": 1, "sourceTimestamp": 0.5, "relativePath": "Frames/00000001.jpg"},
    ]
    artifacts = []
    for index in (0, 1):
        paths = (
            f"Predictions/{index:08d}.depth-f16",
            f"Predictions/{index:08d}.confidence-f16",
            f"Predictions/{index:08d}.geometry.json",
        )
        for relative in paths:
            (project / relative).write_bytes(f"committed-{relative}".encode())
        artifacts.append(
            {
                "frameIndex": index,
                "windowIndex": 0,
                "depthRelativePath": paths[0],
                "confidenceRelativePath": paths[1],
                "geometryRelativePath": paths[2],
                "durationSeconds": 0,
            }
        )
    (project / "Points/window-00000000.cpc").write_bytes(b"committed-cpc")
    manifest["completedWindows"] = [
        {
            "index": 0,
            "inferenceFrameStart": 0,
            "frameStart": 0,
            "frameEnd": 1,
            "pointChunkRelativePath": "Points/window-00000000.cpc",
            "alignmentRowMajor": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "lastProcessedFrameIndex": 1,
            "inlierCount": 2,
            "durationSeconds": 0,
            "frameArtifacts": artifacts,
        }
    ]
    (project / "Manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    original = {
        path.relative_to(project): path.read_bytes()
        for path in (
            *sorted((project / "Predictions").iterdir()),
            project / "Points/window-00000000.cpc",
        )
    }
    Image.new("RGB", (42, 28), (90, 80, 70)).save(project / "Frames/00000002.jpg")
    events: list[object] = []
    runner = SessionRunner(project, FakeModel(), events.append, project_id=PROJECT_ID)
    runner.configure(CONFIG)
    runner.begin(BeginSessionPayload(ResumeCheckpoint(1, 0, 1)))
    for index in range(3):
        runner.enqueue(_frame(index))
    runner.finish_input()
    runner.process()

    assert runner.processed_frames == 1
    assert [
        event.payload.frame_index for event in events if event.type == "frameCompleted"
    ] == [2]
    assert (project / "Points/window-00000001.cpc").exists()
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

    for command in (hello, configure, begin, enqueue, finish):
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

    server.handle(hello)
    server.close()

    ready_index = next(
        index for index, event in enumerate(output) if event.type == "ready"
    )
    loading_heartbeats = [
        event for event in output[:ready_index] if event.type == "heartbeat"
    ]
    assert len(loading_heartbeats) >= 3
    assert all(event.payload.busy for event in loading_heartbeats)


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
    for command in commands:
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
        model_loader=lambda: FakeModel(),
        input_stream=source,
        output_stream=destination,
        heartbeat_interval=60,
    )

    assert server.serve() == 0

    destination.seek(0)
    decoded: list[dict[str, object]] = []
    while destination.tell() < len(destination.getvalue()):
        decoded.append(read_json_frame(destination))
    assert [value["type"] for value in decoded] == [
        "ack",
        "heartbeat",
        "modelProgress",
        "modelProgress",
        "ready",
        "ack",
    ]
    assert all(decode_event(value) for value in decoded)


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
