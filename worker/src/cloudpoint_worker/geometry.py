"""Camera decoding and deterministic colored point generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cloudpoint_worker.cpc import CPCVertex, reduce_vertices
from cloudpoint_worker.errors import WorkerFault

_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class DecodedCamera:
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray


def _fault(message: str) -> WorkerFault:
    return WorkerFault("INVALID_MODEL_OUTPUT", message, False)


def _rotation_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise _fault("camera quaternion is invalid")
    x, y, z, w = quaternion.astype(np.float64) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def decode_camera(
    pose_encoding: np.ndarray, image_size: tuple[int, int]
) -> DecodedCamera:
    """Decode upstream ``absT_quaR_FoV`` W2C; size is ``(width, height)``."""

    pose = np.asarray(pose_encoding, dtype=np.float64)
    if pose.shape != (9,) or not np.isfinite(pose).all():
        raise _fault("pose encoding must contain nine finite values")
    width, height = image_size
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise _fault("image size must be positive")
    fov_h, fov_w = float(pose[7]), float(pose[8])
    if not 0 < fov_h < math.pi or not 0 < fov_w < math.pi:
        raise _fault("camera fields of view are outside (0, pi)")

    rotation = _rotation_from_xyzw(pose[3:7])
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = pose[:3]
    camera_to_world = np.linalg.inv(world_to_camera)

    intrinsics = np.array(
        [
            [(width / 2.0) / math.tan(fov_w / 2.0), 0.0, width / 2.0],
            [0.0, (height / 2.0) / math.tan(fov_h / 2.0), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(intrinsics).all() or not np.isfinite(camera_to_world).all():
        raise _fault("decoded camera is non-finite")
    return DecodedCamera(
        intrinsics=intrinsics.astype(np.float32),
        world_to_camera=world_to_camera.astype(np.float32),
        camera_to_world=camera_to_world.astype(np.float32),
    )


def unproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Unproject OpenCV pixel/depth coordinates and transform them into world space."""

    depth_array = np.asarray(depth, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if depth_array.ndim != 2 or k.shape != (3, 3) or c2w.shape != (4, 4):
        raise _fault("depth or camera matrix shape is invalid")
    if not np.isfinite(k).all() or not np.isfinite(c2w).all():
        raise _fault("camera matrices must be finite")
    try:
        inverse_k = np.linalg.inv(k)
    except np.linalg.LinAlgError as error:
        raise _fault("intrinsics matrix is singular") from error

    height, width = depth_array.shape
    v, u = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    rays = pixels @ inverse_k.T
    camera_points = rays * depth_array[..., None]
    homogeneous = np.concatenate(
        (camera_points, np.ones((height, width, 1), dtype=np.float64)), axis=-1
    )
    return (homogeneous @ c2w.T)[..., :3].astype(np.float32)


def filter_and_reduce_points(
    depth: np.ndarray,
    confidence: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    source_frame: int,
    confidence_floor: float,
    voxel_size: float,
    flags: int,
) -> tuple[CPCVertex, ...]:
    """Filter invalid pixels and retain one reproducible winner per voxel."""

    depth_array = np.asarray(depth, dtype=np.float32)
    confidence_array = np.asarray(confidence, dtype=np.float32)
    rgb_array = np.asarray(rgb, dtype=np.float32)
    if (
        depth_array.ndim != 2
        or confidence_array.shape != depth_array.shape
        or rgb_array.shape != (*depth_array.shape, 3)
    ):
        raise _fault("depth, confidence, and RGB shapes do not agree")
    if (
        not math.isfinite(confidence_floor)
        or confidence_floor <= 0
        or not math.isfinite(voxel_size)
        or voxel_size <= 0
    ):
        raise _fault("filter thresholds must be finite and positive")
    if type(source_frame) is not int or not 0 <= source_frame <= _UINT32_MAX:
        raise _fault("source frame exceeds UInt32")
    if type(flags) is not int or not 0 <= flags <= 0b11:
        raise _fault("point flags contain unsupported reserved bits")

    points = unproject_depth(depth_array, intrinsics, camera_to_world)
    mask = (
        np.isfinite(depth_array)
        & (depth_array > 0)
        & np.isfinite(confidence_array)
        & (confidence_array >= confidence_floor)
        & np.isfinite(rgb_array).all(axis=-1)
        & np.isfinite(points).all(axis=-1)
    )
    flat_indices = np.flatnonzero(mask.reshape(-1))
    flat_points = points.reshape(-1, 3)
    flat_confidence = confidence_array.reshape(-1)
    flat_rgb = np.clip(np.rint(rgb_array.reshape(-1, 3) * 255.0), 0, 255).astype(
        np.uint8
    )
    vertices = (
        CPCVertex(
            position=tuple(float(value) for value in flat_points[index]),
            rgba=(*tuple(int(value) for value in flat_rgb[index]), 255),
            confidence=float(flat_confidence[index]),
            flags=flags,
            source_frame=source_frame,
            pixel_index=int(index),
        )
        for index in flat_indices
    )
    return reduce_vertices(vertices, voxel_size=voxel_size)


__all__ = [
    "DecodedCamera",
    "decode_camera",
    "filter_and_reduce_points",
    "unproject_depth",
]
