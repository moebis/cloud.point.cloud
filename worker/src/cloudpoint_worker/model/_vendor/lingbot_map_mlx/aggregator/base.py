"""AggregatorBase — MLX port. Inference only."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn

from lingbot_map_mlx.layers.patch_embed import PatchEmbed
from lingbot_map_mlx.layers.block import Block
from lingbot_map_mlx.layers.rope import RotaryPositionEmbedding2D, PositionGetter

_RESNET_MEAN = mx.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 1, 3)
_RESNET_STD = mx.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 1, 3)


def slice_expand_and_flatten(token: mx.array, B: int, S: int, first_num_frame: int = 1) -> mx.array:
    """Expand token [1, 2, N, C] to [B*S, N, C], using token[:,0] for first frames, token[:,1] for rest."""
    first_num_frame = max(1, first_num_frame)
    token_first = mx.broadcast_to(token[:, :1], (B, first_num_frame, token.shape[2], token.shape[3]))
    if S > first_num_frame:
        token_rest = mx.broadcast_to(token[:, 1:], (B, S - first_num_frame, token.shape[2], token.shape[3]))
        token_expanded = mx.concatenate([token_first, token_rest], axis=1)
    else:
        token_expanded = token_first[:, :S]
    return token_expanded.reshape(B * S, -1, token.shape[-1])


class AggregatorBase(nn.Module, ABC):
    def __init__(
        self,
        img_size=518, patch_size=14, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=4.0, num_register_tokens=4,
        block_fn=Block, qkv_bias=True, proj_bias=True, ffn_bias=True,
        qk_norm=True, init_values=0.01, patch_embed="dinov2_vitl14_reg",
        pretrained_path=None, aa_order=None, aa_block_size=1,
        rope_freq=100, disable_global_rope=False,
        use_reentrant=False, use_gradient_checkpoint=True,
    ):
        super().__init__()
        if aa_order is None:
            aa_order = ["frame", "global"]

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_register_tokens = num_register_tokens
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.disable_global_rope = disable_global_rope
        self.aa_block_num = depth // aa_block_size

        # Patch embed — weights loaded via convert_weights, not built from DINOv2 here
        from lingbot_map_mlx.layers.vision_transformer import DinoVisionTransformer
        self.patch_embed = DinoVisionTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            embed_dim=embed_dim, depth=24, num_heads=16,
            num_register_tokens=num_register_tokens, qk_norm=False,
            ffn_layer="mlp", init_values=1.0,
        )

        # RoPE
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # Build blocks (implemented by subclass)
        self._build_blocks(
            block_fn=block_fn, depth=depth, embed_dim=embed_dim,
            num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
            init_values=init_values, qk_norm=qk_norm,
        )
        self._setup_special_tokens()

    @abstractmethod
    def _build_blocks(self, block_fn, depth, embed_dim, num_heads, mlp_ratio,
                      qkv_bias, proj_bias, ffn_bias, init_values, qk_norm):
        pass

    @abstractmethod
    def _setup_special_tokens(self):
        pass

    @abstractmethod
    def _prepare_special_tokens(self, B, S_local, S_global, C, **kwargs):
        pass

    def _embed_images(self, images: mx.array, num_frame_for_scale=None):
        """Embed images: (B, S, H, W, C) → tokens (B*S, P, embed_dim)."""
        B, S, H, W, C_in = images.shape

        # Normalize
        images = (images - _RESNET_MEAN) / _RESNET_STD

        S_local = S_global = S

        # Reshape for patch embedding: (B*S, H, W, C)
        images_flat = images.reshape(B * S, H, W, C_in)

        # Run patch embedding (DINOv2 ViT)
        patch_out = self.patch_embed(images_flat)
        if isinstance(patch_out, dict):
            patch_tokens = patch_out["x_norm_patchtokens"]
        else:
            patch_tokens = patch_out

        _, P_patch, C = patch_tokens.shape

        # Prepare special tokens
        special_tokens = self._prepare_special_tokens(
            B, S_local, S_global, C, num_frame_for_scale=num_frame_for_scale
        )

        tokens = mx.concatenate([special_tokens, patch_tokens], axis=1)
        _, P, C = tokens.shape

        return tokens, B, S_local, S_global, P, C

    def _get_positions(self, B, S, H, W):
        if self.rope is None:
            return None
        pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = mx.zeros((B * S, self.patch_start_idx, 2))
            pos = mx.concatenate([pos_special, pos], axis=1)
        return pos

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B * S, P, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))
        return tokens, frame_idx, intermediates

    @abstractmethod
    def _process_global_attention(self, tokens, B, S_local, S_global, P, C,
                                  global_idx, pos=None, **kwargs):
        pass

    def __call__(self, images, selected_idx=None, num_frame_for_scale=None,
                 sliding_window_size=None, num_frame_per_block=1):
        B, S_input, H, W, C_in = images.shape

        tokens, B, S_local, S_global, P, C = self._embed_images(
            images, num_frame_for_scale=num_frame_for_scale)

        pos_local = self._get_positions(B, S_local, H, W)
        pos_global = self._get_positions(B, S_global, H, W)

        frame_idx = 0
        global_idx = 0
        output_list = []

        for block_group_idx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S_local, P, C, frame_idx, pos=pos_local)
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S_local, S_global, P, C, global_idx,
                        pos=pos_global,
                        num_frame_for_scale=num_frame_for_scale,
                        sliding_window_size=sliding_window_size,
                        num_frame_per_block=num_frame_per_block,
                        image_height=H, image_width=W,
                    )

            if selected_idx is None or block_group_idx in selected_idx:
                for i in range(len(frame_intermediates)):
                    concat = mx.concatenate([frame_intermediates[i], global_intermediates[i]], axis=-1)
                    output_list.append(concat)

        return output_list, self.patch_start_idx
