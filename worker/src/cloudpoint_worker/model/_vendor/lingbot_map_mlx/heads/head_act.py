"""Activation functions for prediction heads."""

import mlx.core as mx
import mlx.nn as nn


def activate_pose(pred_pose_enc: mx.array, trans_act="linear", quat_act="linear", fl_act="linear") -> mx.array:
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)

    return mx.concatenate([T, quat, fl], axis=-1)


def base_pose_act(pose_enc: mx.array, act_type="linear") -> mx.array:
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return mx.exp(pose_enc)
    elif act_type == "relu":
        return nn.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out: mx.array, activation="norm_exp", conf_activation="expp1"):
    """Process network output to extract 3D points and confidence.

    Args:
        out: (B*S, H, W, C) in NHWC format (MLX convention)

    Returns:
        (pts3d, conf_out) tuple
    """
    # out is already NHWC in MLX: (B*S, H, W, C)
    fmap = out
    xyz = fmap[..., :-1]
    conf = fmap[..., -1]

    if activation == "norm_exp":
        d = mx.maximum(mx.sqrt((xyz * xyz).sum(axis=-1, keepdims=True)), 1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * mx.expm1(d)
    elif activation == "norm":
        pts3d = xyz / mx.sqrt((xyz * xyz).sum(axis=-1, keepdims=True))
    elif activation == "exp":
        pts3d = mx.exp(xyz)
    elif activation == "relu":
        pts3d = nn.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        z = inverse_log_transform(z)
        pts3d = mx.concatenate([xy * z, z], axis=-1)
    elif activation == "sigmoid":
        pts3d = mx.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + mx.exp(conf)
    elif conf_activation == "expp0":
        conf_out = mx.exp(conf)
    elif conf_activation == "sigmoid":
        conf_out = mx.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


def inverse_log_transform(y: mx.array) -> mx.array:
    return mx.sign(y) * mx.expm1(mx.abs(y))
