"""Exact atomic per-frame prediction artifacts."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from cloudpoint_worker import PROTOCOL_VERSION
from cloudpoint_worker.cpc import _open_directory_fd, atomic_write_bytes
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.preprocess import PreprocessedFrame


class PersistedFrameLike(Protocol):
    index: int
    source_timestamp: float
    relative_path: str


class FramePredictionLike(Protocol):
    depth: object
    confidence: object
    pose_encoding: object
    intrinsics: object
    camera_to_world: object


@dataclass(frozen=True)
class FrameArtifactPaths:
    depth_path: str
    confidence_path: str
    geometry_path: str


def _fault(message: str, *, code: str = "INVALID_MODEL_OUTPUT") -> WorkerFault:
    return WorkerFault(code, message, False)


def _real_directory(path: Path, *, create: bool) -> None:
    descriptor = -1
    try:
        descriptor = _open_directory_fd(path, create_leaf=create)
    except (OSError, ValueError) as error:
        raise _fault("artifact directory must be a real symlink-free path") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _matrix_values(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise _fault(f"{label} has an invalid shape or value")
    return result


def write_frame_outputs(
    project_root: Path,
    frame: PersistedFrameLike,
    prediction: FramePredictionLike,
    preprocessed: PreprocessedFrame,
    *,
    confidence_floor: float,
    engine_version: str,
    model_identifier: str,
    model_revision: str,
) -> FrameArtifactPaths:
    """Write Float16 rasters plus canonical geometry metadata without clobbering."""

    if not project_root.is_absolute():
        raise _fault("project root must be absolute")
    _real_directory(project_root, create=False)
    if type(frame.index) is not int or not 0 <= frame.index <= 2**32 - 1:
        raise _fault("frame index exceeds UInt32")
    if not math.isfinite(float(frame.source_timestamp)) or frame.source_timestamp < 0:
        raise _fault("source timestamp must be finite and nonnegative")
    if not math.isfinite(confidence_floor) or confidence_floor <= 0:
        raise _fault("confidence floor must be finite and positive")

    predictions = project_root / "Predictions"
    _real_directory(predictions, create=True)
    stem = f"{frame.index:08d}"
    relative = FrameArtifactPaths(
        depth_path=f"Predictions/{stem}.depth-f16",
        confidence_path=f"Predictions/{stem}.confidence-f16",
        geometry_path=f"Predictions/{stem}.geometry.json",
    )
    finals = tuple(
        project_root / value
        for value in (
            relative.depth_path,
            relative.confidence_path,
            relative.geometry_path,
        )
    )
    for final in finals:
        if final.exists() or final.is_symlink():
            raise _fault(
                "prediction output already exists", code="OUTPUT_ALREADY_EXISTS"
            )

    depth = np.asarray(prediction.depth, dtype=np.float32)
    confidence = np.asarray(prediction.confidence, dtype=np.float32)
    expected_shape = (preprocessed.model_size[1], preprocessed.model_size[0])
    if (
        depth.shape != expected_shape
        or confidence.shape != expected_shape
        or not np.isfinite(depth).all()
        or not np.isfinite(confidence).all()
        or np.any(depth <= 0)
    ):
        raise _fault("depth or confidence raster is invalid")
    pose = _matrix_values(prediction.pose_encoding, (9,), "pose encoding")
    intrinsics = _matrix_values(prediction.intrinsics, (3, 3), "intrinsics")
    camera_to_world = _matrix_values(
        prediction.camera_to_world, (4, 4), "camera-to-world"
    )
    transform = _matrix_values(
        preprocessed.model_to_source, (3, 3), "model-to-source transform"
    )

    metadata = {
        "cameraToWorld": camera_to_world.reshape(-1).tolist(),
        "confidenceFloor": confidence_floor,
        "engineVersion": engine_version,
        "frameIndex": frame.index,
        "intrinsics": intrinsics.reshape(-1).tolist(),
        "modelIdentifier": model_identifier,
        "modelRevision": model_revision,
        "modelSize": list(preprocessed.model_size),
        "modelToSource": transform.reshape(-1).tolist(),
        "poseEncoding": pose.tolist(),
        "protocolVersion": PROTOCOL_VERSION,
        "reconstructionUnit": "model-depth-unit",
        "sourcePath": frame.relative_path,
        "sourceSize": list(preprocessed.source_size),
        "sourceTimestamp": float(frame.source_timestamp),
    }
    json_payload = (
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")

    atomic_write_bytes(finals[0], depth.astype("<f2", copy=False).tobytes(order="C"))
    atomic_write_bytes(
        finals[1], confidence.astype("<f2", copy=False).tobytes(order="C")
    )
    atomic_write_bytes(finals[2], json_payload)
    return relative


__all__ = ["FrameArtifactPaths", "write_frame_outputs"]
