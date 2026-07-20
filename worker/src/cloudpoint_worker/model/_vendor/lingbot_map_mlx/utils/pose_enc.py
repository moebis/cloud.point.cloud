"""Camera pose encoding/decoding for MLX."""

import mlx.core as mx
from lingbot_map_mlx.utils.rotation import quat_to_mat, mat_to_quat
from lingbot_map_mlx.utils.geometry import closed_form_inverse_se3_general


def pose_encoding_to_extri_intri(
    pose_encoding: mx.array,
    image_size_hw=None,
    pose_encoding_type="absT_quaR_FoV",
    build_intrinsics=True,
):
    """Convert pose encoding (B, S, 9) to extrinsics (B, S, 3, 4) and intrinsics (B, S, 3, 3)."""
    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = mx.concatenate([R, T[..., None]], axis=-1)

        if build_intrinsics and image_size_hw is not None:
            H, W = image_size_hw
            fy = (H / 2.0) / mx.tan(fov_h / 2.0)
            fx = (W / 2.0) / mx.tan(fov_w / 2.0)
            intrinsics = mx.zeros((*pose_encoding.shape[:2], 3, 3))
            # Build intrinsics manually since MLX doesn't support advanced index assignment
            # We'll construct it from components
            batch_shape = pose_encoding.shape[:2]
            zeros = mx.zeros(batch_shape)
            ones = mx.ones(batch_shape)
            cx = mx.full(batch_shape, W / 2.0)
            cy = mx.full(batch_shape, H / 2.0)

            row0 = mx.stack([fx, zeros, cx], axis=-1)
            row1 = mx.stack([zeros, fy, cy], axis=-1)
            row2 = mx.stack([zeros, zeros, ones], axis=-1)
            intrinsics = mx.stack([row0, row1, row2], axis=-2)

    elif pose_encoding_type == "absT_quaR":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        R = quat_to_mat(quat)
        extrinsics = mx.concatenate([R, T[..., None]], axis=-1)
    else:
        raise NotImplementedError(f"Unknown: {pose_encoding_type}")

    return extrinsics, intrinsics
