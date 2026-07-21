from __future__ import annotations

import errno
import json
import os
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import cloudpoint_worker.outputs as outputs_module
from cloudpoint_worker.cpc import CPCVertex, read_cpc, reduce_vertices, write_cpc
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.geometry import (
    decode_camera,
    filter_and_reduce_points,
    unproject_depth,
)
from cloudpoint_worker.outputs import write_frame_outputs
from cloudpoint_worker.preprocess import PreprocessedFrame


@contextmanager
def _propagate_exception() -> Iterator[None]:
    yield


def test_worker_fault_allows_contextmanager_traceback_assignment() -> None:
    fault = WorkerFault("TEST_FAULT", "traceback compatible", False)

    with pytest.raises(WorkerFault) as caught, _propagate_exception():
        raise fault

    assert caught.value is fault
    assert caught.value.__traceback__ is not None


def _translation(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    result[:3, 3] = (x, y, z)
    return result


def test_unprojection_uses_opencv_w2c_then_c2w() -> None:
    depth = np.array([[2.0]], np.float32)
    intrinsics = np.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        np.float32,
    )
    points = unproject_depth(depth, intrinsics, _translation(1, 2, 3))
    np.testing.assert_allclose(points[0, 0], [1, 2, 5])


def test_decode_camera_uses_scalar_last_quaternion_and_inverts_w2c() -> None:
    # 90 degrees about Z, scalar-last XYZW, with a world-to-camera translation.
    sine = np.sqrt(0.5)
    pose = np.array([1, 2, 3, 0, 0, sine, sine, np.pi / 2, np.pi / 2])
    decoded = decode_camera(pose, image_size=(8, 4))

    np.testing.assert_allclose(decoded.intrinsics, [[4, 0, 4], [0, 2, 2], [0, 0, 1]])
    np.testing.assert_allclose(
        decoded.world_to_camera,
        [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        decoded.camera_to_world @ decoded.world_to_camera,
        np.eye(4),
        atol=1e-6,
    )


def test_filtering_and_voxel_ties_are_deterministic() -> None:
    depth = np.array([[1.0, 1.0], [0.0, np.nan]], dtype=np.float32)
    confidence = np.array([[2.0, 3.0], [9.0, 9.0]], dtype=np.float32)
    rgb = np.array(
        [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]],
        dtype=np.float32,
    )
    intrinsics = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]], np.float32)

    vertices = filter_and_reduce_points(
        depth,
        confidence,
        rgb,
        intrinsics,
        np.eye(4, dtype=np.float32),
        source_frame=7,
        confidence_floor=1.5,
        voxel_size=1.0,
        flags=1,
    )

    assert len(vertices) == 1
    assert vertices[0].rgba == (0, 255, 0, 255)
    assert vertices[0].confidence == 3.0
    assert vertices[0].source_frame == 7

    tied = (
        CPCVertex((0.01, 0, 1), (1, 2, 3, 255), 2.0, 0, 8, pixel_index=9),
        CPCVertex((0.02, 0, 1), (4, 5, 6, 255), 2.0, 0, 7, pixel_index=10),
        CPCVertex((0.03, 0, 1), (7, 8, 9, 255), 2.0, 0, 7, pixel_index=3),
    )
    reduced = reduce_vertices(tied, voxel_size=1.0)
    assert len(reduced) == 1
    assert reduced[0].rgba == (7, 8, 9, 255)


@pytest.mark.parametrize("position", [1.0, 1e308])
def test_voxel_keys_reject_values_outside_int64(position: float) -> None:
    vertex = CPCVertex((position, 0, 1), (1, 2, 3, 255), 2.0, 0, 0)

    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        reduce_vertices((vertex,), voxel_size=1e-300)


def test_filtering_rejects_reserved_flag_bits() -> None:
    with pytest.raises(WorkerFault, match="INVALID_MODEL_OUTPUT"):
        filter_and_reduce_points(
            np.ones((1, 1), dtype=np.float32),
            np.full((1, 1), 2, dtype=np.float32),
            np.zeros((1, 1, 3), dtype=np.float32),
            np.eye(3, dtype=np.float32),
            np.eye(4, dtype=np.float32),
            source_frame=0,
            confidence_floor=1.5,
            voxel_size=0.01,
            flags=4,
        )


def test_cpc1_exact_layout_round_trip_and_no_clobber(tmp_path: Path) -> None:
    vertices = (
        CPCVertex((0, 0, 1), (255, 0, 0, 255), 2.0, 1, 1),
        CPCVertex((1, 0, 1), (0, 255, 0, 255), 3.0, 0, 2),
        CPCVertex((0, 1, 1), (0, 0, 255, 255), 4.0, 2, 2),
    )
    path = tmp_path / "window-00000001.cpc"
    descriptor = write_cpc(path, 1, 2, vertices)
    raw = path.read_bytes()

    assert raw[:4] == b"CPC1"
    assert len(raw) == 32 + 3 * 24
    assert struct.unpack("<4sHHQII8s", raw[:32]) == (
        b"CPC1",
        1,
        24,
        3,
        1,
        2,
        b"\0" * 8,
    )
    assert descriptor.point_count == 3
    assert descriptor.relative_path == "window-00000001.cpc"
    assert read_cpc(path).vertices == vertices

    with pytest.raises(WorkerFault, match="OUTPUT_ALREADY_EXISTS"):
        write_cpc(path, 1, 2, vertices)


def test_cpc_rejects_invalid_values_ranges_and_truncation(tmp_path: Path) -> None:
    path = tmp_path / "invalid.cpc"
    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        write_cpc(path, 2, 1, ())
    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        write_cpc(path, 0, 0, (CPCVertex((np.nan, 0, 0), (0, 0, 0, 0), 1, 0, 0),))
    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        write_cpc(path, 0, 0, (CPCVertex((0, 0, 0), (0, 0, 0, 256), 1, 0, 0),))

    truncated = tmp_path / "truncated.cpc"
    truncated.write_bytes(struct.pack("<4sHHQII8s", b"CPC1", 1, 24, 1, 0, 0, b"\0" * 8))
    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        read_cpc(truncated)


@pytest.mark.parametrize(
    ("vertex", "frame_start", "frame_end"),
    [
        (CPCVertex((0, 0, 1), (0, 0, 0, 255), 2, 4, 1), 1, 1),
        (CPCVertex((0, 0, 1), (0, 0, 0, 255), 2, 0, 9), 1, 2),
    ],
)
def test_cpc_writer_rejects_reserved_flags_and_frames_outside_header(
    tmp_path: Path,
    vertex: CPCVertex,
    frame_start: int,
    frame_end: int,
) -> None:
    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        write_cpc(tmp_path / "invalid.cpc", frame_start, frame_end, (vertex,))


@pytest.mark.parametrize(("flags", "source_frame"), [(4, 1), (0, 9)])
def test_cpc_reader_rejects_reserved_flags_and_frames_outside_header(
    tmp_path: Path, flags: int, source_frame: int
) -> None:
    path = tmp_path / "invalid.cpc"
    path.write_bytes(
        struct.pack("<4sHHQII8s", b"CPC1", 1, 24, 1, 1, 2, b"\0" * 8)
        + struct.pack("<fff4BeHI", 0, 0, 1, 0, 0, 0, 255, 2, flags, source_frame)
    )

    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        read_cpc(path)


def test_cpc_reader_rejects_symlink_without_following_it(tmp_path: Path) -> None:
    outside = tmp_path / "outside.cpc"
    write_cpc(
        outside,
        0,
        0,
        (CPCVertex((0, 0, 1), (0, 0, 0, 255), 2, 0, 0),),
    )
    link = tmp_path / "linked.cpc"
    link.symlink_to(outside)

    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        read_cpc(link)


def test_cpc_reader_checks_size_before_allocating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = tmp_path / "oversized.cpc"
    oversized.touch()
    os.truncate(oversized, 1_200_000_033)

    def unexpected_path_read(_: Path) -> bytes:
        raise AssertionError("read_bytes called before the CPC size bound")

    monkeypatch.setattr(Path, "read_bytes", unexpected_path_read)
    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        read_cpc(oversized)


def test_atomic_writer_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    inner = real / "inner"
    inner.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    destination = alias / inner.name / "escaped.cpc"

    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        write_cpc(
            destination,
            0,
            0,
            (CPCVertex((0, 0, 1), (0, 0, 0, 255), 2, 0, 0),),
        )

    assert not (inner / destination.name).exists()


def test_atomic_writer_cleans_partial_when_promotion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "failed.cpc"

    def fail_link(*_: object, **__: object) -> None:
        raise OSError(errno.EIO, "injected promotion failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        write_cpc(
            destination,
            0,
            0,
            (CPCVertex((0, 0, 1), (0, 0, 0, 255), 2, 0, 0),),
        )

    assert not destination.exists()
    assert not [path for path in tmp_path.iterdir() if path.name.endswith(".partial")]


def test_atomic_writer_rolls_back_owned_final_when_directory_sync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "failed-after-promotion.cpc"
    real_fsync = os.fsync
    calls = 0

    def fail_first_directory_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "injected directory sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_directory_sync)

    with pytest.raises(WorkerFault, match="INVALID_POINT_CHUNK"):
        write_cpc(
            destination,
            0,
            0,
            (CPCVertex((0, 0, 1), (0, 0, 0, 255), 2, 0, 0),),
        )

    assert not destination.exists()
    assert not [path for path in tmp_path.iterdir() if path.name.endswith(".partial")]


@dataclass(frozen=True)
class _Frame:
    index: int
    source_timestamp: float
    relative_path: str


@dataclass(frozen=True)
class _Prediction:
    depth: np.ndarray
    confidence: np.ndarray
    pose_encoding: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray


def _one_pixel_preprocessed() -> PreprocessedFrame:
    return PreprocessedFrame(
        rgb=np.zeros((1, 1, 3), np.float32),
        normalized=np.zeros((1, 1, 3), np.float32),
        model_to_source=np.eye(3, dtype=np.float64),
        source_size=(1, 1),
        model_size=(1, 1),
    )


def _one_pixel_prediction() -> _Prediction:
    return _Prediction(
        depth=np.ones((1, 1), np.float32),
        confidence=np.full((1, 1), 2, np.float32),
        pose_encoding=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
    )


def test_frame_outputs_reject_symlinked_project_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    real_project = real_parent / "Test.cloudpoint"
    real_project.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(WorkerFault, match="INVALID_MODEL_OUTPUT"):
        write_frame_outputs(
            alias / real_project.name,
            _Frame(0, 0, "Frames/00000000.jpg"),
            _one_pixel_prediction(),
            _one_pixel_preprocessed(),
            confidence_floor=1.5,
            engine_version="1.0.0",
            model_identifier="robbyant/lingbot-map",
            model_revision="204754b",
        )

    assert not (real_project / "Predictions").exists()


def test_frame_outputs_are_exact_atomic_and_descriptive(tmp_path: Path) -> None:
    project = tmp_path / "Test.cloudpoint"
    project.mkdir()
    preprocessed = PreprocessedFrame(
        rgb=np.zeros((2, 2, 3), np.float32),
        normalized=np.zeros((2, 2, 3), np.float32),
        model_to_source=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], np.float64),
        source_size=(4, 4),
        model_size=(2, 2),
    )
    prediction = _Prediction(
        depth=np.array([[1, 2], [3, 4]], np.float32),
        confidence=np.array([[2, 3], [4, 5]], np.float32),
        pose_encoding=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
    )

    paths = write_frame_outputs(
        project,
        _Frame(42, 1.25, "Frames/00000042.jpg"),
        prediction,
        preprocessed,
        confidence_floor=1.5,
        engine_version="1.0.0",
        model_identifier="robbyant/lingbot-map",
        model_revision="204754b",
    )

    assert paths.depth_path == "Predictions/00000042.depth-f16"
    assert paths.confidence_path == "Predictions/00000042.confidence-f16"
    assert paths.geometry_path == "Predictions/00000042.geometry.json"
    assert (project / paths.depth_path).read_bytes() == prediction.depth.astype(
        "<f2"
    ).tobytes()
    metadata = json.loads((project / paths.geometry_path).read_text())
    assert metadata["frameIndex"] == 42
    assert metadata["sourceSize"] == [4, 4]
    assert metadata["modelSize"] == [2, 2]
    assert metadata["reconstructionUnit"] == "model-depth-unit"
    assert metadata["cameraToWorld"] == list(np.eye(4).reshape(-1))
    assert not list(project.rglob("*.partial"))

    with pytest.raises(WorkerFault, match="OUTPUT_ALREADY_EXISTS"):
        write_frame_outputs(
            project,
            _Frame(42, 1.25, "Frames/00000042.jpg"),
            prediction,
            preprocessed,
            confidence_floor=1.5,
            engine_version="1.0.0",
            model_identifier="robbyant/lingbot-map",
            model_revision="204754b",
        )


@pytest.mark.parametrize("failure_ordinal", [2, 3])
def test_frame_output_transaction_rolls_back_earlier_promotions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_ordinal: int,
) -> None:
    project = tmp_path / "Test.cloudpoint"
    project.mkdir()
    real_write = outputs_module.atomic_write_bytes
    calls: list[Path] = []

    def fail_later_promotion(path: Path, payload: bytes) -> object:
        calls.append(path)
        if len(calls) == failure_ordinal:
            raise WorkerFault("SYNTHETIC_PROMOTION_FAILURE", "synthetic", False)
        return real_write(path, payload)

    monkeypatch.setattr(outputs_module, "atomic_write_bytes", fail_later_promotion)

    with pytest.raises(WorkerFault, match="SYNTHETIC_PROMOTION_FAILURE"):
        write_frame_outputs(
            project,
            _Frame(0, 0, "Frames/00000000.jpg"),
            _one_pixel_prediction(),
            _one_pixel_preprocessed(),
            confidence_floor=1.5,
            engine_version="1.0.0",
            model_identifier="robbyant/lingbot-map",
            model_revision="204754b",
        )

    canonical = (
        project / "Predictions/00000000.depth-f16",
        project / "Predictions/00000000.confidence-f16",
        project / "Predictions/00000000.geometry.json",
    )
    assert not any(path.exists() for path in canonical)
    assert not list(project.rglob("*.partial"))


def test_frame_output_rollback_preserves_replacement_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "Test.cloudpoint"
    project.mkdir()
    real_write = outputs_module.atomic_write_bytes
    calls: list[Path] = []
    replacement = b"replacement-owned-by-another-writer"

    def replace_then_fail_third(path: Path, payload: bytes) -> object:
        calls.append(path)
        if len(calls) == 3:
            calls[0].unlink()
            calls[0].write_bytes(replacement)
            raise WorkerFault("SYNTHETIC_PROMOTION_FAILURE", "synthetic", False)
        return real_write(path, payload)

    monkeypatch.setattr(outputs_module, "atomic_write_bytes", replace_then_fail_third)

    with pytest.raises(WorkerFault, match="SYNTHETIC_PROMOTION_FAILURE"):
        write_frame_outputs(
            project,
            _Frame(0, 0, "Frames/00000000.jpg"),
            _one_pixel_prediction(),
            _one_pixel_preprocessed(),
            confidence_floor=1.5,
            engine_version="1.0.0",
            model_identifier="robbyant/lingbot-map",
            model_revision="204754b",
        )

    depth = project / "Predictions/00000000.depth-f16"
    assert depth.read_bytes() == replacement
    assert not (project / "Predictions/00000000.confidence-f16").exists()
    assert not (project / "Predictions/00000000.geometry.json").exists()


@pytest.mark.parametrize("collision_ordinal", [2, 3])
def test_frame_output_rollback_preserves_racing_preexisting_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_ordinal: int,
) -> None:
    project = tmp_path / "Test.cloudpoint"
    project.mkdir()
    real_write = outputs_module.atomic_write_bytes
    calls: list[Path] = []
    preexisting = b"preexisting-from-another-writer"

    def collide_with_later_promotion(path: Path, payload: bytes) -> object:
        calls.append(path)
        if len(calls) == collision_ordinal:
            path.write_bytes(preexisting)
        return real_write(path, payload)

    monkeypatch.setattr(
        outputs_module, "atomic_write_bytes", collide_with_later_promotion
    )

    with pytest.raises(WorkerFault, match="OUTPUT_ALREADY_EXISTS"):
        write_frame_outputs(
            project,
            _Frame(0, 0, "Frames/00000000.jpg"),
            _one_pixel_prediction(),
            _one_pixel_preprocessed(),
            confidence_floor=1.5,
            engine_version="1.0.0",
            model_identifier="robbyant/lingbot-map",
            model_revision="204754b",
        )

    canonical = (
        project / "Predictions/00000000.depth-f16",
        project / "Predictions/00000000.confidence-f16",
        project / "Predictions/00000000.geometry.json",
    )
    assert canonical[collision_ordinal - 1].read_bytes() == preexisting
    for ordinal, path in enumerate(canonical, start=1):
        if ordinal != collision_ordinal:
            assert not path.exists()
