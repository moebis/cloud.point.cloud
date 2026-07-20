"""3D geometry utilities for MLX."""

import numpy as np
import mlx.core as mx


def closed_form_inverse_se3(se3: mx.array, R=None, T=None) -> mx.array:
    """Invert batch of SE3 matrices. se3: (N, 4, 4) or (N, 3, 4)."""
    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    R_t = mx.transpose(R, axes=(0, 2, 1))
    top_right = -(R_t @ T)

    N = se3.shape[0]
    bottom = mx.concatenate([
        mx.zeros((N, 1, 3)),
        mx.ones((N, 1, 1)),
    ], axis=-1)

    top = mx.concatenate([R_t, top_right], axis=-1)
    return mx.concatenate([top, bottom], axis=-2)


def closed_form_inverse_se3_general(se3: mx.array, R=None, T=None) -> mx.array:
    """Invert SE3 with arbitrary batch dims. se3: (..., 4, 4)."""
    batch_shape = se3.shape[:-2]
    if R is None:
        R = se3[..., :3, :3]
    if T is None:
        T = se3[..., :3, 3:]

    R_t = mx.swapaxes(R, -2, -1)
    top_right = -(R_t @ T)

    eye = mx.eye(4)
    result = mx.broadcast_to(eye, (*batch_shape, 4, 4)).astype(se3.dtype)
    # Build result from components
    top = mx.concatenate([R_t, top_right], axis=-1)
    bottom_row = mx.zeros((*batch_shape, 1, 3))
    bottom_one = mx.ones((*batch_shape, 1, 1))
    bottom = mx.concatenate([bottom_row, bottom_one], axis=-1)
    return mx.concatenate([top, bottom], axis=-2)


def unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam):
    """Numpy-based unprojection (kept as-is for post-processing)."""
    if isinstance(depth_map, mx.array):
        depth_map = np.array(depth_map)
    if isinstance(extrinsics_cam, mx.array):
        extrinsics_cam = np.array(extrinsics_cam)
    if isinstance(intrinsics_cam, mx.array):
        intrinsics_cam = np.array(intrinsics_cam)

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points = _depth_to_world_np(
            depth_map[frame_idx].squeeze(-1),
            extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx],
        )
        world_points_list.append(cur_world_points)
    return np.stack(world_points_list, axis=0)


def _depth_to_world_np(depth_map, extrinsic, intrinsic, eps=1e-8):
    """Convert depth map to world coords (numpy)."""
    H, W = depth_map.shape[:2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x_cam = (u - cx) * depth_map / fx
    y_cam = (v - cy) * depth_map / fy
    z_cam = depth_map

    cam_coords = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # Invert extrinsic: w2c -> c2w
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    world_coords = np.dot(cam_coords, R_inv.T) + t_inv
    return world_coords
