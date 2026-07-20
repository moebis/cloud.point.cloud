"""Utility functions for prediction heads."""

import mlx.core as mx


def position_grid_to_embed(pos_grid: mx.array, embed_dim: int, omega_0: float = 100) -> mx.array:
    """Convert 2D position grid (H, W, 2) to sinusoidal embeddings (H, W, embed_dim)."""
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # (H*W, 2)

    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)

    emb = mx.concatenate([emb_x, emb_y], axis=-1)  # (H*W, embed_dim)
    return emb.reshape(H, W, embed_dim)


def make_sincos_pos_embed(embed_dim: int, pos: mx.array, omega_0: float = 100) -> mx.array:
    """Generate 1D sinusoidal positional embedding."""
    assert embed_dim % 2 == 0
    omega = mx.arange(embed_dim // 2).astype(mx.float32)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (omega_0 ** omega)  # (D/2,)

    pos = pos.reshape(-1).astype(mx.float32)  # (M,)
    out = pos[:, None] * omega[None, :]  # (M, D/2)

    emb_sin = mx.sin(out)
    emb_cos = mx.cos(out)

    emb = mx.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def create_uv_grid(width: int, height: int, aspect_ratio: float = None, dtype=mx.float32) -> mx.array:
    """Create normalized UV grid of shape (height, width, 2)."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords = mx.linspace(left_x, right_x, width).astype(dtype)
    y_coords = mx.linspace(top_y, bottom_y, height).astype(dtype)

    # meshgrid with xy indexing: uu has shape (height, width)
    uu, vv = mx.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = mx.stack([uu, vv], axis=-1)  # (height, width, 2)

    return uv_grid
