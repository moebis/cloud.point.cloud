"""Pinned Lingbot Map topology used by conversion and MLX inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for the one supported Lingbot Map topology."""

    image_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    heads: int = 16
    register_tokens: int = 4
    selected_layers: tuple[int, ...] = (4, 11, 17, 23)
    kv_cache_sliding_window: int = 64
    kv_cache_scale_frames: int = 8
    camera_refinement_iterations: int = 4
    enable_3d_rope: bool = True
    enable_camera_3d_rope: bool = False
    maximum_frames: int = 1024


__all__ = ["ModelConfig"]
