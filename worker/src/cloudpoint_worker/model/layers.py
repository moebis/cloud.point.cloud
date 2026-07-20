"""Stable CloudPoint exports for Lingbot's MLX leaf layers."""

from __future__ import annotations

import mlx.core as mx

from cloudpoint_worker.model._vendor.lingbot_map_mlx.layers.attention import (
    Attention,
    CausalAttention,
    SDPAAttention,
)
from cloudpoint_worker.model._vendor.lingbot_map_mlx.layers.block import (
    Block,
    CameraBlock,
    SDPABlock,
)
from cloudpoint_worker.model._vendor.lingbot_map_mlx.layers.layer_scale import (
    LayerScale,
)
from cloudpoint_worker.model._vendor.lingbot_map_mlx.layers.mlp import Mlp
from cloudpoint_worker.model._vendor.lingbot_map_mlx.layers.patch_embed import (
    PatchEmbed,
)


def attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None,
) -> mx.array:
    """Apply stable Float32 scaled dot-product attention and restore q's dtype."""

    scale = q.shape[-1] ** -0.5
    scores = (q.astype(mx.float32) * scale) @ k.astype(mx.float32).swapaxes(-1, -2)
    if mask is not None:
        scores = mx.where(mask, scores, mx.array(-1e9, dtype=mx.float32))
    return (mx.softmax(scores, axis=-1) @ v.astype(mx.float32)).astype(q.dtype)


__all__ = [
    "Attention",
    "Block",
    "CameraBlock",
    "CausalAttention",
    "LayerScale",
    "Mlp",
    "PatchEmbed",
    "SDPAAttention",
    "SDPABlock",
    "attention",
]
