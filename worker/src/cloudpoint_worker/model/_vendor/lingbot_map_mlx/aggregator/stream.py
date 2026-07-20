"""AggregatorStream — MLX port. SDPA-only path, no FlashInfer."""

from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn

from lingbot_map_mlx.layers.block import Block, SDPABlock
from lingbot_map_mlx.layers.rope import WanRotaryPosEmbed
from lingbot_map_mlx.aggregator.base import AggregatorBase, slice_expand_and_flatten


class AggregatorStream(AggregatorBase):
    def __init__(
        self,
        sliding_window_size: int = -1,
        num_frame_for_scale: int = 1,
        num_random_frames: int = 0,
        attend_to_special_tokens: bool = False,
        attend_to_scale_frames: bool = False,
        enable_3d_rope: bool = False,
        max_frame_num: int = 1024,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
        **kwargs,
    ):
        self.sliding_window_size = sliding_window_size
        self.num_frame_for_scale = num_frame_for_scale
        self.num_random_frames = num_random_frames
        self.attend_to_special_tokens = attend_to_special_tokens
        self.attend_to_scale_frames = attend_to_scale_frames
        self.enable_3d_rope = enable_3d_rope
        self.max_frame_num = max_frame_num

        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only

        # Remove kwargs not needed by base
        for k in ('enable_stream_inference', 'use_flashinfer', 'use_flexflash', 'use_sdpa'):
            kwargs.pop(k, None)

        super().__init__(**kwargs)

        self._init_kv_cache()
        if self.enable_3d_rope:
            self._init_3d_rope()

    def _build_blocks(self, block_fn, depth, embed_dim, num_heads, mlp_ratio,
                      qkv_bias, proj_bias, ffn_bias, init_values, qk_norm):
        block_params = dict(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
            init_values=init_values, qk_norm=qk_norm,
        )
        self.frame_blocks = [
            block_fn(**block_params, rope=self.rope)
            for _ in range(depth)
        ]
        self.global_blocks = [
            SDPABlock(
                **block_params,
                rope=self.rope if not self.disable_global_rope else None,
                kv_cache_sliding_window=self.kv_cache_sliding_window,
                kv_cache_scale_frames=self.kv_cache_scale_frames,
                kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
                kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
                kv_cache_camera_only=self.kv_cache_camera_only,
            )
            for _ in range(depth)
        ]

    def _setup_special_tokens(self):
        self.camera_token = mx.random.normal((1, 2, 1, self.embed_dim)) * 1e-6
        self.register_token = mx.random.normal((1, 2, self.num_register_tokens, self.embed_dim)) * 1e-6
        self.scale_token = mx.random.normal((1, 2, 1, self.embed_dim)) * 1e-6

        self.patch_start_idx = 1 + self.num_register_tokens + 1  # camera + register + scale
        self.num_special_tokens = 1 + self.num_register_tokens + 1

    def _init_kv_cache(self):
        self.kv_cache = {}
        self.total_frames_processed = 0
        self._cached_pos3d = None

        for i in range(self.depth):
            self.kv_cache[f"k_{i}"] = None
            self.kv_cache[f"v_{i}"] = None
            self.kv_cache[f"k_{i}_special"] = None
            self.kv_cache[f"v_{i}_special"] = None

    def clean_kv_cache(self):
        for key in list(self.kv_cache.keys()):
            if key == "_skip_append":
                self.kv_cache[key] = False
            else:
                self.kv_cache[key] = None
        self.total_frames_processed = 0
        self._cached_pos3d = None

    def _init_3d_rope(self):
        if not self.enable_3d_rope:
            self.rope3d = None
            return
        head_dim = self.embed_dim // 16
        self.rope3d = WanRotaryPosEmbed(
            attention_head_dim=head_dim,
            patch_size=(1, self.patch_size, self.patch_size),
            max_seq_len=self.max_frame_num,
        )

    def _get_3d_positions_streaming(self, num_frames, H, W, f_start, f_end):
        if self.rope3d is None:
            return None
        pph = H // self.patch_size
        ppw = W // self.patch_size
        return self.rope3d(
            ppf=num_frames, pph=pph, ppw=ppw,
            patch_start_idx=self.num_special_tokens,
            f_start=f_start, f_end=f_end,
        )

    def _prepare_special_tokens(self, B, S_local, S_global, C, num_frame_for_scale=None):
        scale_frames = self.num_frame_for_scale if num_frame_for_scale is None else num_frame_for_scale

        has_cache = self.kv_cache.get("k_0") is not None
        if has_cache:
            S_cached = self.kv_cache["k_0"].shape[2]
            S_true = S_cached + S_global
        else:
            S_true = S_global

        if has_cache and S_true > S_global:
            effective_scale = min(scale_frames, S_true)
            cam_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            cam = cam_full[-S_global:]
            reg_full = slice_expand_and_flatten(self.register_token, B, S_true)
            reg = reg_full[-S_global:]
            scale_full = slice_expand_and_flatten(self.scale_token, B, S_true, first_num_frame=effective_scale)
            scale = scale_full[-S_global:]
        else:
            effective_scale = min(scale_frames, S_global)
            cam = slice_expand_and_flatten(self.camera_token, B, S_global)
            reg = slice_expand_and_flatten(self.register_token, B, S_global)
            scale = slice_expand_and_flatten(self.scale_token, B, S_global, first_num_frame=effective_scale)

        return mx.concatenate([cam, reg, scale], axis=1)

    def _process_global_attention(self, tokens, B, S_local, S_global, P, C,
                                  global_idx, pos=None, **kwargs):
        image_height = kwargs.get('image_height', self.img_size)
        image_width = kwargs.get('image_width', self.img_size)
        num_frame_per_block = kwargs.get('num_frame_per_block', 1)
        num_frame_for_scale = kwargs.get('num_frame_for_scale', self.num_frame_for_scale)

        scale_frames = num_frame_for_scale if num_frame_for_scale is not None else self.num_frame_for_scale

        # Reshape: (B*S, P, C) → (B, S*P, C)
        if tokens.shape != (B, S_local * P, C):
            tokens = tokens.reshape(B, S_local, P, C).reshape(B, S_local * P, C)

        num_frames = S_global
        num_patches = P - self.num_special_tokens
        is_first_block_group = (global_idx < self.aa_block_size)

        if self.enable_3d_rope and hasattr(self, 'rope3d') and self.rope3d is not None:
            if is_first_block_group:
                f_start = self.total_frames_processed
                f_end = self.total_frames_processed + S_global
                H = image_height
                W = image_width
                pos3d = self._get_3d_positions_streaming(S_global, H, W, f_start, f_end)
                self._cached_pos3d = pos3d
            else:
                pos3d = self._cached_pos3d
            pos = pos3d
        else:
            if pos is not None and pos.shape != (B, S_global * P, 2):
                pos = pos.reshape(B, S_global, P, 2).reshape(B, S_global * P, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](
                tokens, pos=pos, enable_ulysses_cp=False,
                num_patches=num_patches,
                num_special=self.num_special_tokens,
                num_frames=num_frames,
                enable_3d_rope=self.enable_3d_rope,
                kv_cache=self.kv_cache,
                global_idx=global_idx,
                num_frame_per_block=num_frame_per_block,
                num_frame_for_scale=scale_frames,
                num_register_tokens=self.num_register_tokens,
            )
            global_idx += 1
            intermediates.append(tokens.reshape(B, S_local, P, C))

        if is_first_block_group and not self.kv_cache.get("_skip_append", False):
            self.total_frames_processed += S_global

        return tokens, global_idx, intermediates
