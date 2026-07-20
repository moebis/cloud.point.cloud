"""Rotation utilities for MLX — quaternion ↔ rotation matrix."""

import mlx.core as mx


def quat_to_mat(quaternions: mx.array) -> mx.array:
    """Convert quaternions (XYZW, scalar-last) to rotation matrices.

    Args:
        quaternions: (..., 4)
    Returns:
        Rotation matrices (..., 3, 3)
    """
    i = quaternions[..., 0]
    j = quaternions[..., 1]
    k = quaternions[..., 2]
    r = quaternions[..., 3]

    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = mx.stack([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ], axis=-1)

    return o.reshape(*quaternions.shape[:-1], 3, 3)


def mat_to_quat(matrix: mx.array) -> mx.array:
    """Convert rotation matrices to quaternions (XYZW, scalar-last).

    Args:
        matrix: (..., 3, 3)
    Returns:
        Quaternions (..., 4)
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(*batch_shape, 9)

    m00 = m[..., 0]; m01 = m[..., 1]; m02 = m[..., 2]
    m10 = m[..., 3]; m11 = m[..., 4]; m12 = m[..., 5]
    m20 = m[..., 6]; m21 = m[..., 7]; m22 = m[..., 8]

    q_abs = _sqrt_positive_part(mx.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], axis=-1))

    quat_by_rijk = mx.stack([
        mx.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
        mx.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
        mx.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
        mx.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
    ], axis=-2)

    flr = mx.array(0.1)
    quat_candidates = quat_by_rijk / (2.0 * mx.maximum(q_abs[..., None], flr))

    # Pick the best-conditioned quaternion (largest q_abs)
    best_idx = mx.argmax(q_abs, axis=-1)
    # Manual gather: iterate over batch dims
    flat_candidates = quat_candidates.reshape(-1, 4, 4)
    flat_idx = best_idx.reshape(-1)
    # Use one_hot selection
    one_hot = mx.eye(4)[flat_idx]  # (N, 4)
    # (N, 4, 4) * (N, 4, 1) -> sum over dim -2 -> (N, 4)
    out = (flat_candidates * one_hot[:, :, None]).sum(axis=-2)
    out = out.reshape(*batch_shape, 4)

    # Convert from rijk to ijkr
    out = mx.concatenate([out[..., 1:2], out[..., 2:3], out[..., 3:4], out[..., 0:1]], axis=-1)

    # Standardize: make real part non-negative
    out = mx.where(out[..., 3:4] < 0, -out, out)

    return out


def _sqrt_positive_part(x: mx.array) -> mx.array:
    return mx.sqrt(mx.maximum(x, 0.0))
