"""Rotary Position Embeddings for MLX.

Rewrites PyTorch complex-number RoPE as real-valued cos/sin operations.
"""

from typing import Dict, List, Optional, Tuple, Union

import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid."""

    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], mx.array] = {}

    def __call__(self, batch_size: int, height: int, width: int) -> mx.array:
        if (height, width) not in self.position_cache:
            y_coords = mx.arange(height)
            x_coords = mx.arange(width)
            # meshgrid then stack to get (H*W, 2) positions
            yy, xx = mx.meshgrid(y_coords, x_coords, indexing="ij")
            positions = mx.stack([yy.reshape(-1), xx.reshape(-1)], axis=-1)  # (H*W, 2)
            self.position_cache[height, width] = positions

        cached = self.position_cache[height, width]
        # (1, H*W, 2) -> (B, H*W, 2)
        return mx.broadcast_to(cached[None], (batch_size, cached.shape[0], 2))


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding using real-valued cos/sin."""

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[mx.array, mx.array]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, dtype=mx.float32,
    ) -> Tuple[mx.array, mx.array]:
        cache_key = (dim, seq_len, dtype)
        if cache_key not in self.frequency_cache:
            exponents = mx.arange(0, dim, 2).astype(mx.float32) / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)

            positions = mx.arange(seq_len).astype(mx.float32)
            # outer product: (seq_len, dim//2)
            angles = positions[:, None] * inv_freq[None, :]
            angles = mx.concatenate([angles, angles], axis=-1)
            cos_comp = mx.cos(angles).astype(dtype)
            sin_comp = mx.sin(angles).astype(dtype)
            self.frequency_cache[cache_key] = (cos_comp, sin_comp)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: mx.array) -> mx.array:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    def _apply_1d_rope(
        self, tokens: mx.array, positions: mx.array,
        cos_comp: mx.array, sin_comp: mx.array,
    ) -> mx.array:
        # positions: (B, N) integer indices
        # cos_comp, sin_comp: (max_pos, dim)
        # Gather: (B, N, dim) -> (B, 1, N, dim) for broadcasting with (B, H, N, dim)
        positions = positions.astype(mx.int32)
        cos = cos_comp[positions][:, None, :, :]  # (B, 1, N, dim)
        sin = sin_comp[positions][:, None, :, :]

        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def __call__(self, tokens: mx.array, positions: mx.array) -> mx.array:
        # tokens: (B, H, N, D) where D must be divisible by 4
        # positions: (B, N, 2) with (y, x) coords
        assert tokens.shape[-1] % 2 == 0
        assert positions.ndim == 3 and positions.shape[-1] == 2

        feature_dim = tokens.shape[-1] // 2
        max_position = int(positions.max().item()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.dtype,
        )

        vert_features, horiz_features = mx.split(tokens, 2, axis=-1)
        vert_features = self._apply_1d_rope(vert_features, positions[..., 0], cos_comp, sin_comp)
        horiz_features = self._apply_1d_rope(horiz_features, positions[..., 1], cos_comp, sin_comp)

        return mx.concatenate([vert_features, horiz_features], axis=-1)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real: bool = True,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
):
    """Compute 1D rotary position embeddings as real cos/sin pairs.

    Always returns real-valued (cos, sin) tensors for MLX compatibility.
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = np.arange(pos, dtype=np.float64)
    elif isinstance(pos, mx.array):
        pos = np.array(pos, dtype=np.float64)
    elif not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    else:
        pos = pos.astype(np.float64)

    theta = theta * ntk_factor
    freq_indices = np.arange(0, dim, 2, dtype=np.float64)[: dim // 2]
    freqs = 1.0 / (theta ** (freq_indices / dim)) / linear_factor  # (D/2,)
    angles = np.outer(pos, freqs)  # (S, D/2)

    if repeat_interleave_real:
        # Interleaved: [cos_0, cos_0, cos_1, cos_1, ...]
        cos_vals = np.repeat(np.cos(angles), 2, axis=1).astype(np.float32)
        sin_vals = np.repeat(np.sin(angles), 2, axis=1).astype(np.float32)
    else:
        cos_vals = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(np.float32)
        sin_vals = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(np.float32)

    return mx.array(cos_vals), mx.array(sin_vals)


class WanRotaryPosEmbed(nn.Module):
    """3D Rotary Position Embedding using real-valued cos/sin pairs."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        fhw_dim: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        if fhw_dim is not None:
            fhw_dim = list(fhw_dim)
            assert attention_head_dim == sum(fhw_dim)
            t_dim, h_dim, w_dim = fhw_dim
        else:
            h_dim = w_dim = 2 * (attention_head_dim // 6)
            t_dim = attention_head_dim - h_dim - w_dim

        self.fhw_dim = (t_dim, h_dim, w_dim)

        # Precompute cos/sin for each dimension
        freqs_cos_list = []
        freqs_sin_list = []
        for dim in [t_dim, h_dim, w_dim]:
            cos_f, sin_f = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta,
                use_real=True, repeat_interleave_real=True,
            )
            freqs_cos_list.append(cos_f)
            freqs_sin_list.append(sin_f)

        # freqs_cos[i]: (max_seq_len, dim_i)
        self._freqs_cos = freqs_cos_list
        self._freqs_sin = freqs_sin_list

    def __call__(
        self, ppf: int, pph: int, ppw: int, patch_start_idx: int,
        f_start: int = 0, f_end: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Generate 3D RoPE as (cos, sin) pair.

        Returns:
            (freqs_cos, freqs_sin): each of shape (1, 1, total_tokens, head_dim)
        """
        t_dim, h_dim, w_dim = self.fhw_dim

        # cos/sin for each dimension: (max_seq_len, dim_i)
        cos_t, sin_t = self._freqs_cos[0], self._freqs_sin[0]
        cos_h, sin_h = self._freqs_cos[1], self._freqs_sin[1]
        cos_w, sin_w = self._freqs_cos[2], self._freqs_sin[2]

        if f_end is not None:
            ppf = f_end - f_start
            frame_slice = slice(f_start, f_end)
        else:
            frame_slice = slice(0, ppf)

        if patch_start_idx > 0:
            # Special tokens: each at diagonal position (f, i, i)
            # cos/sin for special tokens
            spec_cos_f = mx.broadcast_to(cos_t[frame_slice][:, None, :], (ppf, patch_start_idx, t_dim))
            spec_sin_f = mx.broadcast_to(sin_t[frame_slice][:, None, :], (ppf, patch_start_idx, t_dim))
            spec_cos_h = mx.broadcast_to(cos_h[:patch_start_idx][None, :, :], (ppf, patch_start_idx, h_dim))
            spec_sin_h = mx.broadcast_to(sin_h[:patch_start_idx][None, :, :], (ppf, patch_start_idx, h_dim))
            spec_cos_w = mx.broadcast_to(cos_w[:patch_start_idx][None, :, :], (ppf, patch_start_idx, w_dim))
            spec_sin_w = mx.broadcast_to(sin_w[:patch_start_idx][None, :, :], (ppf, patch_start_idx, w_dim))
            special_cos = mx.concatenate([spec_cos_f, spec_cos_h, spec_cos_w], axis=-1)
            special_sin = mx.concatenate([spec_sin_f, spec_sin_h, spec_sin_w], axis=-1)

            # Patches: position (f, patch_start_idx+h, patch_start_idx+w)
            p_cos_f = mx.broadcast_to(cos_t[frame_slice][:, None, None, :], (ppf, pph, ppw, t_dim))
            p_sin_f = mx.broadcast_to(sin_t[frame_slice][:, None, None, :], (ppf, pph, ppw, t_dim))
            p_cos_h = mx.broadcast_to(
                cos_h[patch_start_idx: patch_start_idx + pph][None, :, None, :], (ppf, pph, ppw, h_dim))
            p_sin_h = mx.broadcast_to(
                sin_h[patch_start_idx: patch_start_idx + pph][None, :, None, :], (ppf, pph, ppw, h_dim))
            p_cos_w = mx.broadcast_to(
                cos_w[patch_start_idx: patch_start_idx + ppw][None, None, :, :], (ppf, pph, ppw, w_dim))
            p_sin_w = mx.broadcast_to(
                sin_w[patch_start_idx: patch_start_idx + ppw][None, None, :, :], (ppf, pph, ppw, w_dim))
            patch_cos = mx.concatenate([p_cos_f, p_cos_h, p_cos_w], axis=-1).reshape(ppf, pph * ppw, -1)
            patch_sin = mx.concatenate([p_sin_f, p_sin_h, p_sin_w], axis=-1).reshape(ppf, pph * ppw, -1)

            # Combine: (ppf, patch_start_idx + pph*ppw, head_dim)
            all_cos = mx.concatenate([special_cos, patch_cos], axis=1)
            all_sin = mx.concatenate([special_sin, patch_sin], axis=1)

            total_tokens = ppf * (patch_start_idx + pph * ppw)
            all_cos = all_cos.reshape(total_tokens, -1)[None, None, :, :]
            all_sin = all_sin.reshape(total_tokens, -1)[None, None, :, :]
            return all_cos, all_sin

        # No special tokens — patches only
        p_cos_f = mx.broadcast_to(cos_t[frame_slice][:, None, None, :], (ppf, pph, ppw, t_dim))
        p_sin_f = mx.broadcast_to(sin_t[frame_slice][:, None, None, :], (ppf, pph, ppw, t_dim))
        p_cos_h = mx.broadcast_to(cos_h[:pph][None, :, None, :], (ppf, pph, ppw, h_dim))
        p_sin_h = mx.broadcast_to(sin_h[:pph][None, :, None, :], (ppf, pph, ppw, h_dim))
        p_cos_w = mx.broadcast_to(cos_w[:ppw][None, None, :, :], (ppf, pph, ppw, w_dim))
        p_sin_w = mx.broadcast_to(sin_w[:ppw][None, None, :, :], (ppf, pph, ppw, w_dim))

        all_cos = mx.concatenate([p_cos_f, p_cos_h, p_cos_w], axis=-1).reshape(1, 1, ppf * pph * ppw, -1)
        all_sin = mx.concatenate([p_sin_f, p_sin_h, p_sin_w], axis=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return all_cos, all_sin


def apply_rotary_emb(x: mx.array, freqs) -> mx.array:
    """Apply rotary embeddings using real-valued cos/sin rotation.

    Args:
        x: (batch, heads, seq_len, head_dim)
        freqs: Either a (cos, sin) tuple of (1, 1, seq_len, head_dim) arrays,
               or a single complex-style array (for backward compat — will be unused in MLX).

    Returns:
        Rotated x with same shape and dtype.
    """
    if isinstance(freqs, tuple):
        cos, sin = freqs
    else:
        raise ValueError("MLX apply_rotary_emb requires (cos, sin) tuple, not complex freqs")

    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    cos = cos.astype(mx.float32)
    sin = sin.astype(mx.float32)

    # Real-valued rotation: pairs of features rotated by angle
    # x = [..., d] where d is head_dim
    # Split into interleaved pairs: x[..., 0::2] and x[..., 1::2]
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices

    # cos/sin are interleaved (repeat_interleave_real=True):
    # cos = [cos_0, cos_0, cos_1, cos_1, ...]
    # Take every other element to get per-pair values
    cos_half = cos[..., 0::2]
    sin_half = sin[..., 0::2]

    # Apply rotation: (x1 + ix2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
    out1 = x1 * cos_half - x2 * sin_half
    out2 = x1 * sin_half + x2 * cos_half

    # Interleave back: [out1_0, out2_0, out1_1, out2_1, ...]
    out = mx.stack([out1, out2], axis=-1).reshape(*x.shape)

    return out.astype(orig_dtype)
