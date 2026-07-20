"""Pinned Lingbot Map topology used by conversion and MLX inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    heads: int = 16
    register_tokens: int = 4
    selected_layers: tuple[int, ...] = (4, 11, 17, 23)


__all__ = ["ModelConfig"]
