"""Attention layers for MLX.

Ports: Attention, SDPAAttention, CausalAttention from PyTorch.
Skips FlashInferAttention (CUDA-only).
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from lingbot_map_mlx.layers.rope import apply_rotary_emb


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def __call__(
        self, x: mx.array, pos=None,
        enable_ulysses_cp=False, num_patches=None, num_special=None,
        num_frames=None, enable_3d_rope=False,
    ) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, axes=(2, 0, 3, 1, 4))  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and not enable_3d_rope:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        elif enable_3d_rope and pos is not None:
            q = apply_rotary_emb(q, pos)
            k = apply_rotary_emb(k, pos)

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = mx.transpose(x, axes=(0, 2, 1, 3)).reshape(B, -1, self.num_heads * self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SDPAAttention(Attention):
    """SDPA attention with dict-based KV cache for streaming inference."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
    ):
        super().__init__(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer,
            qk_norm=qk_norm, fused_attn=fused_attn, rope=rope,
        )
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only

    def __call__(
        self, x: mx.array, pos=None,
        enable_ulysses_cp=False, num_patches=None, num_special=None,
        num_frames=None, enable_3d_rope=False,
        kv_cache=None, global_idx=0, num_frame_per_block=1,
        num_frame_for_scale=-1, num_register_tokens=4,
    ) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, axes=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        # ── Batch Mode ──
        if kv_cache is None:
            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif self.rope is not None and enable_3d_rope:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            x = mx.transpose(x, axes=(0, 2, 1, 3)).reshape(B, N, self.num_heads * self.head_dim)

        # ── Streaming Mode (dict KV cache) ──
        else:
            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif self.rope is not None and enable_3d_rope:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            camera_token_idx = 0
            scale_token_idx = camera_token_idx + num_register_tokens + 1

            tokens_per_frame = N // num_frame_per_block
            k_reshaped = k.reshape(B, self.num_heads, num_frame_per_block, tokens_per_frame, self.head_dim)
            v_reshaped = v.reshape(B, self.num_heads, num_frame_per_block, tokens_per_frame, self.head_dim)

            if kv_cache[f"k_{global_idx}"] is None:
                kv_cache[f"k_{global_idx}"] = k_reshaped
                kv_cache[f"v_{global_idx}"] = v_reshaped
            else:
                cached_tpf = kv_cache[f"k_{global_idx}"].shape[3]
                num_frame_per_block = k.shape[2] // cached_tpf
                k_reshaped = k.reshape(B, self.num_heads, num_frame_per_block, cached_tpf, self.head_dim)
                v_reshaped = v.reshape(B, self.num_heads, num_frame_per_block, cached_tpf, self.head_dim)
                kv_cache[f"k_{global_idx}"] = mx.concatenate(
                    [kv_cache[f"k_{global_idx}"], k_reshaped], axis=2)
                kv_cache[f"v_{global_idx}"] = mx.concatenate(
                    [kv_cache[f"v_{global_idx}"], v_reshaped], axis=2)

            self._apply_kv_cache_eviction(
                kv_cache, global_idx, camera_token_idx, scale_token_idx, num_register_tokens)

            k_cached = kv_cache[f"k_{global_idx}"]
            v_cached = kv_cache[f"v_{global_idx}"]
            a, b, c, d, e = k_cached.shape
            k_full = k_cached.reshape(a, b, c * d, e)
            v_full = v_cached.reshape(a, b, c * d, e)

            if f"k_{global_idx}_special" in kv_cache and kv_cache[f"k_{global_idx}_special"] is not None:
                special_k = kv_cache[f"k_{global_idx}_special"]
                special_v = kv_cache[f"v_{global_idx}_special"]
                sa, sb, sc, sd, se = special_k.shape
                k_full = mx.concatenate([special_k.reshape(sa, sb, sc * sd, se), k_full], axis=2)
                v_full = mx.concatenate([special_v.reshape(sa, sb, sc * sd, se), v_full], axis=2)

            q_seq_len = q.shape[2]
            x = mx.fast.scaled_dot_product_attention(q, k_full, v_full, scale=self.scale)
            x = mx.transpose(x, axes=(0, 2, 1, 3)).reshape(B, q_seq_len, self.num_heads * self.head_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _apply_kv_cache_eviction(self, kv_cache, global_idx, camera_token_idx, scale_token_idx, num_register_tokens):
        sliding_window_frames = self.kv_cache_sliding_window
        scale_frames = self.kv_cache_scale_frames

        if kv_cache[f"k_{global_idx}"].shape[3] > 1:
            num_cached_frames = kv_cache[f"k_{global_idx}"].shape[2]
            if num_cached_frames > sliding_window_frames + scale_frames:
                evict_start = scale_frames
                evict_end = num_cached_frames - sliding_window_frames
                if evict_end > evict_start:
                    evicted_k = kv_cache[f"k_{global_idx}"][:, :, evict_start:evict_end, :, :]
                    evicted_v = kv_cache[f"v_{global_idx}"][:, :, evict_start:evict_end, :, :]

                    if self.kv_cache_cross_frame_special:
                        if self.kv_cache_camera_only:
                            new_special_k = evicted_k[:, :, :, camera_token_idx:camera_token_idx+1, :]
                            new_special_v = evicted_v[:, :, :, camera_token_idx:camera_token_idx+1, :]
                        else:
                            new_special_k = evicted_k[:, :, :, camera_token_idx:scale_token_idx+1, :]
                            new_special_v = evicted_v[:, :, :, camera_token_idx:scale_token_idx+1, :]

                        if f"k_{global_idx}_special" not in kv_cache or kv_cache[f"k_{global_idx}_special"] is None:
                            kv_cache[f"k_{global_idx}_special"] = new_special_k
                            kv_cache[f"v_{global_idx}_special"] = new_special_v
                        else:
                            kv_cache[f"k_{global_idx}_special"] = mx.concatenate(
                                [kv_cache[f"k_{global_idx}_special"], new_special_k], axis=2)
                            kv_cache[f"v_{global_idx}_special"] = mx.concatenate(
                                [kv_cache[f"v_{global_idx}_special"], new_special_v], axis=2)

                    if self.kv_cache_include_scale_frames:
                        kv_cache[f"k_{global_idx}"] = mx.concatenate([
                            kv_cache[f"k_{global_idx}"][:, :, :scale_frames, :, :],
                            kv_cache[f"k_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        ], axis=2)
                        kv_cache[f"v_{global_idx}"] = mx.concatenate([
                            kv_cache[f"v_{global_idx}"][:, :, :scale_frames, :, :],
                            kv_cache[f"v_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        ], axis=2)
                    else:
                        kv_cache[f"k_{global_idx}"] = kv_cache[f"k_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        kv_cache[f"v_{global_idx}"] = kv_cache[f"v_{global_idx}"][:, :, -sliding_window_frames:, :, :]


class CausalAttention(nn.Module):
    """Causal attention with KV cache for camera head streaming inference."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        elementwise_attn_output_gate: bool = False,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.gate_proj = nn.Linear(dim, dim, bias=True) if elementwise_attn_output_gate else None

        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only

    def __call__(
        self, x: mx.array, block_mask=None, pos=None, pos_kv=None,
        frame_seqlen=None, video_mask=None, kv_cache=None,
        current_start=0, current_end=0, global_idx=0,
        num_frame_per_block=1, num_frame_for_scale=-1,
        enable_3d_rope=False, sliding_window_size=-1,
        attend_to_scale_frames=False, num_random_frames=0,
        attend_to_special_tokens=False, num_register_tokens=4,
        enable_ulysses_cp=False, is_scale_frames=False,
        full_attention=False,
    ) -> mx.array:
        B, N, C = x.shape
        camera_token_idx = 0
        scale_token_idx = camera_token_idx + num_register_tokens + 1

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, axes=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.gate_proj is not None:
            gate_score = self.gate_proj(x).reshape(B, N, self.num_heads, self.head_dim)
            gate_score = mx.transpose(gate_score, axes=(0, 2, 1, 3))

        if kv_cache is None:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif enable_3d_rope and pos is not None:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            if full_attention:
                x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            else:
                # Build block-causal mask
                seq_len = q.shape[2]
                if frame_seqlen is not None and frame_seqlen > 0:
                    num_f = seq_len // frame_seqlen
                    # Block-causal: each frame attends to current and previous frames
                    mask = mx.zeros((seq_len, seq_len), dtype=mx.bool_)
                    for i in range(num_f):
                        qs, qe = i * frame_seqlen, (i + 1) * frame_seqlen
                        ke = (i + 1) * frame_seqlen
                        # Attend to all frames up to and including current
                        mask = mask.at[qs:qe, :ke].add(mx.ones((frame_seqlen, ke), dtype=mx.bool_))
                    # Scale frames get full attention
                    if num_frame_for_scale > 0:
                        sf_end = num_frame_for_scale * frame_seqlen
                        mask = mask.at[:sf_end, :sf_end].add(mx.ones((sf_end, sf_end), dtype=mx.bool_))
                    # Use additive mask for SDPA: 0 for attend, -inf for block
                    float_mask = mx.where(mask, mx.array(0.0), mx.array(-1e9))
                    float_mask = float_mask[None, None, :, :]  # (1, 1, S, S)
                    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=float_mask)
                else:
                    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            # ── Streaming with KV cache ──
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif enable_3d_rope and pos is not None:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            skip_append = kv_cache.get("_skip_append", False)
            tokens_per_frame = N // num_frame_per_block
            k_reshaped = k.reshape(B, self.num_heads, num_frame_per_block, tokens_per_frame, self.head_dim)
            v_reshaped = v.reshape(B, self.num_heads, num_frame_per_block, tokens_per_frame, self.head_dim)

            if not skip_append:
                if kv_cache[f"k_{global_idx}"] is None:
                    kv_cache[f"k_{global_idx}"] = k_reshaped
                    kv_cache[f"v_{global_idx}"] = v_reshaped
                else:
                    cached_tpf = kv_cache[f"k_{global_idx}"].shape[3]
                    nfpb = k.shape[2] // cached_tpf
                    kr = k.reshape(B, self.num_heads, nfpb, cached_tpf, self.head_dim)
                    vr = v.reshape(B, self.num_heads, nfpb, cached_tpf, self.head_dim)
                    kv_cache[f"k_{global_idx}"] = mx.concatenate([kv_cache[f"k_{global_idx}"], kr], axis=2)
                    kv_cache[f"v_{global_idx}"] = mx.concatenate([kv_cache[f"v_{global_idx}"], vr], axis=2)

                self._apply_kv_cache_eviction(kv_cache, global_idx, camera_token_idx, scale_token_idx)
                k_use = kv_cache[f"k_{global_idx}"]
                v_use = kv_cache[f"v_{global_idx}"]
            else:
                if kv_cache[f"k_{global_idx}"] is not None:
                    k_use = mx.concatenate([kv_cache[f"k_{global_idx}"], k_reshaped], axis=2)
                    v_use = mx.concatenate([kv_cache[f"v_{global_idx}"], v_reshaped], axis=2)
                else:
                    k_use = k_reshaped
                    v_use = v_reshaped

            a, b, c, d, e = k_use.shape
            k_flat = k_use.reshape(a, b, c * d, e)
            v_flat = v_use.reshape(a, b, c * d, e)

            if f"k_{global_idx}_special" in kv_cache and kv_cache[f"k_{global_idx}_special"] is not None:
                sk = kv_cache[f"k_{global_idx}_special"]
                sv = kv_cache[f"v_{global_idx}_special"]
                sa, sb, sc, sd, se = sk.shape
                k_flat = mx.concatenate([sk.reshape(sa, sb, sc * sd, se), k_flat], axis=2)
                v_flat = mx.concatenate([sv.reshape(sa, sb, sc * sd, se), v_flat], axis=2)

            x = mx.fast.scaled_dot_product_attention(q, k_flat, v_flat, scale=self.scale)

        if self.gate_proj is not None:
            x = x * mx.sigmoid(gate_score)

        x = mx.transpose(x, axes=(0, 2, 1, 3)).reshape(B, -1, self.num_heads * self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _apply_kv_cache_eviction(self, kv_cache, global_idx, camera_token_idx, scale_token_idx):
        sliding_window_frames = self.kv_cache_sliding_window
        scale_frames = self.kv_cache_scale_frames

        if kv_cache[f"k_{global_idx}"].shape[3] > 1:
            num_cached_frames = kv_cache[f"k_{global_idx}"].shape[2]
            if num_cached_frames > sliding_window_frames + scale_frames:
                evict_start = scale_frames
                evict_end = num_cached_frames - sliding_window_frames
                if evict_end > evict_start:
                    evicted_k = kv_cache[f"k_{global_idx}"][:, :, evict_start:evict_end, :, :]
                    evicted_v = kv_cache[f"v_{global_idx}"][:, :, evict_start:evict_end, :, :]

                    if self.kv_cache_cross_frame_special:
                        if self.kv_cache_camera_only:
                            new_sk = evicted_k[:, :, :, camera_token_idx:camera_token_idx+1, :]
                            new_sv = evicted_v[:, :, :, camera_token_idx:camera_token_idx+1, :]
                        else:
                            new_sk = evicted_k[:, :, :, camera_token_idx:scale_token_idx+1, :]
                            new_sv = evicted_v[:, :, :, camera_token_idx:scale_token_idx+1, :]

                        if f"k_{global_idx}_special" not in kv_cache or kv_cache[f"k_{global_idx}_special"] is None:
                            kv_cache[f"k_{global_idx}_special"] = new_sk
                            kv_cache[f"v_{global_idx}_special"] = new_sv
                        else:
                            kv_cache[f"k_{global_idx}_special"] = mx.concatenate(
                                [kv_cache[f"k_{global_idx}_special"], new_sk], axis=2)
                            kv_cache[f"v_{global_idx}_special"] = mx.concatenate(
                                [kv_cache[f"v_{global_idx}_special"], new_sv], axis=2)

                    if self.kv_cache_include_scale_frames:
                        kv_cache[f"k_{global_idx}"] = mx.concatenate([
                            kv_cache[f"k_{global_idx}"][:, :, :scale_frames, :, :],
                            kv_cache[f"k_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        ], axis=2)
                        kv_cache[f"v_{global_idx}"] = mx.concatenate([
                            kv_cache[f"v_{global_idx}"][:, :, :scale_frames, :, :],
                            kv_cache[f"v_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        ], axis=2)
                    else:
                        kv_cache[f"k_{global_idx}"] = kv_cache[f"k_{global_idx}"][:, :, -sliding_window_frames:, :, :]
                        kv_cache[f"v_{global_idx}"] = kv_cache[f"v_{global_idx}"][:, :, -sliding_window_frames:, :, :]
