from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from cloudpoint_worker.cpc import CPCVertex, read_cpc, reduce_vertices, write_cpc
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.geometry import (
    decode_camera,
    filter_and_reduce_points,
    unproject_depth,
)
from cloudpoint_worker.outputs import write_frame_outputs
from cloudpoint_worker.preprocess import PreprocessedFrame


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
