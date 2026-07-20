"""Transformer block layers for MLX.

Ports: Block, SDPABlock, CameraBlock from PyTorch.
Skips FlashInferBlock (CUDA-only).
"""

from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention, SDPAAttention, CausalAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=Attention,
        ffn_layer=Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm,
            fused_attn=fused_attn, rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def __call__(
        self, x: mx.array, pos=None, enable_ulysses_cp=False,
        num_patches=None, num_special=None, num_frames=None,
        enable_3d_rope=False,
    ) -> mx.array:
        x = x + self.drop_path1(self.ls1(self.attn(
            self.norm1(x), pos=pos, enable_ulysses_cp=enable_ulysses_cp,
            num_patches=num_patches, num_special=num_special,
            num_frames=num_frames, enable_3d_rope=enable_3d_rope,
        )))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class SDPABlock(nn.Module):
    """SDPA block with dict-based KV cache for streaming inference."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ffn_layer=Mlp,
        qk_norm: bool = False,
        rope=None,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SDPAAttention(
            dim=dim, num_heads=num_heads, qk_norm=qk_norm, qkv_bias=qkv_bias,
            proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, rope=rope,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=kv_cache_scale_frames,
            kv_cache_cross_frame_special=kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=kv_cache_include_scale_frames,
            kv_cache_camera_only=kv_cache_camera_only,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop, bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def __call__(
        self, x: mx.array, pos=None, enable_ulysses_cp=False,
        num_patches=None, num_special=None, num_frames=None,
        enable_3d_rope=False, kv_cache=None, global_idx=0,
        num_frame_per_block=1, num_frame_for_scale=-1,
        num_register_tokens=4,
    ) -> mx.array:
        x = x + self.drop_path1(self.ls1(self.attn(
            self.norm1(x), pos=pos, enable_ulysses_cp=enable_ulysses_cp,
            num_patches=num_patches, num_special=num_special,
            num_frames=num_frames, enable_3d_rope=enable_3d_rope,
            kv_cache=kv_cache, global_idx=global_idx,
            num_frame_per_block=num_frame_per_block,
            num_frame_for_scale=num_frame_for_scale,
            num_register_tokens=num_register_tokens,
        )))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CameraBlock(nn.Module):
    """Camera block with causal attention and KV cache."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=CausalAttention,
        ffn_layer=Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        elementwise_attn_output_gate: bool = False,
        sliding_window_size: int = -1,
        attend_to_scale_frames: bool = False,
        num_random_frames: int = 0,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(
            dim=dim, num_heads=num_heads, qk_norm=qk_norm, qkv_bias=qkv_bias,
            rope=rope, elementwise_attn_output_gate=elementwise_attn_output_gate,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=kv_cache_scale_frames,
            kv_cache_cross_frame_special=kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=kv_cache_include_scale_frames,
            kv_cache_camera_only=kv_cache_camera_only,
        )
        self.sliding_window_size = sliding_window_size
        self.attend_to_scale_frames = attend_to_scale_frames
        self.num_random_frames = num_random_frames

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def __call__(
        self, x: mx.array, pos=None, video_mask=None,
        num_frames=0, frame_seqlen=0, kv_cache=None,
        current_start=0, current_end=0, global_idx=0,
        num_frame_per_block=8, num_frame_for_scale=-1,
        sliding_window_size=None, enable_ulysses_cp=False,
        full_attention=False, enable_3d_rope=False,
        is_scale_frames=False,
    ) -> mx.array:
        effective_sw = sliding_window_size if sliding_window_size is not None else self.sliding_window_size

        x = x + self.drop_path1(self.ls1(self.attn(
            self.norm1(x), pos=pos, frame_seqlen=frame_seqlen,
            video_mask=video_mask, kv_cache=kv_cache,
            current_start=current_start, current_end=current_end,
            global_idx=global_idx, num_frame_per_block=num_frame_per_block,
            num_frame_for_scale=num_frame_for_scale,
            enable_3d_rope=enable_3d_rope,
            sliding_window_size=effective_sw,
            attend_to_scale_frames=self.attend_to_scale_frames,
            num_random_frames=self.num_random_frames,
            enable_ulysses_cp=enable_ulysses_cp,
            is_scale_frames=is_scale_frames,
            full_attention=full_attention,
        )))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
