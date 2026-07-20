"""DINOv2 Vision Transformer for MLX.

Port of DinoVisionTransformer — inference only, no weight init or training code.
"""

from functools import partial
import math
from typing import Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFNFused
from .attention import Attention
from .block import Block


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        drop_cls_token=False,
        qk_norm=False,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1 if not drop_cls_token else 0
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.drop_cls_token = drop_cls_token

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Learnable tokens — stored as plain mx.array attributes
        if not drop_cls_token:
            self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, num_patches + self.num_tokens, embed_dim))
        if num_register_tokens > 0:
            self.register_tokens = mx.zeros((1, num_register_tokens, embed_dim))
        self.mask_token = mx.zeros((1, embed_dim))

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]

        if ffn_layer == "mlp":
            ffn_cls = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_cls = SwiGLUFFNFused
        elif ffn_layer == "identity":
            ffn_cls = lambda *a, **kw: nn.Identity()
        else:
            raise NotImplementedError(f"Unknown ffn_layer: {ffn_layer}")

        self.blocks = [
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                ffn_layer=ffn_cls, init_values=init_values, qk_norm=qk_norm,
            )
            for i in range(depth)
        ]

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

    def interpolate_pos_encoding(self, x: mx.array, w: int, h: int) -> mx.array:
        npatch = x.shape[1] - (1 if not self.drop_cls_token else 0)
        N = self.pos_embed.shape[1] - (1 if not self.drop_cls_token else 0)

        if npatch == N and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.astype(mx.float32)
        if not self.drop_cls_token:
            class_pos_embed = pos_embed[:, :1]
            patch_pos_embed = pos_embed[:, 1:]
        else:
            patch_pos_embed = pos_embed

        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M

        # Reshape to 2D grid and resize via bilinear interpolation
        # patch_pos_embed: (1, M*M, dim) -> (1, M, M, dim)
        grid = patch_pos_embed.reshape(1, M, M, dim)

        if w0 != M or h0 != M:
            # Use nn.Upsample for bilinear interpolation
            upsample = nn.Upsample(scale_factor=(h0 / M, w0 / M), mode="linear", align_corners=False)
            grid = upsample(grid)

        patch_pos_embed = grid.reshape(1, -1, dim)

        if not self.drop_cls_token:
            return mx.concatenate([class_pos_embed, patch_pos_embed], axis=1).astype(x.dtype)
        return patch_pos_embed.astype(x.dtype)

    def prepare_tokens_with_masks(self, x: mx.array, masks=None) -> mx.array:
        # x: (B, H, W, C) in MLX NHWC format
        B, H, W, C = x.shape
        x = self.patch_embed(x)  # (B, N, embed_dim)

        if masks is not None:
            mask_token = mx.broadcast_to(self.mask_token[None], x.shape)
            x = mx.where(masks[..., None], mask_token.astype(x.dtype), x)

        if not self.drop_cls_token:
            cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
            x = mx.concatenate([cls_tokens, x], axis=1)

        x = x + self.interpolate_pos_encoding(x, W, H)

        if self.num_register_tokens > 0:
            reg_tokens = mx.broadcast_to(self.register_tokens, (B, self.num_register_tokens, self.embed_dim))
            x = mx.concatenate([x[:, :1], reg_tokens, x[:, 1:]], axis=1)

        return x

    def forward_features(self, x: mx.array, masks=None) -> dict:
        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        result = {
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:] if not self.drop_cls_token else x_norm[:, self.num_register_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }
        if not self.drop_cls_token:
            result["x_norm_clstoken"] = x_norm[:, 0]
        if self.num_register_tokens > 0:
            start = 1 if not self.drop_cls_token else 0
            result["x_norm_regtokens"] = x_norm[:, start: start + self.num_register_tokens]

        return result

    def __call__(self, x: mx.array, masks=None) -> dict:
        return self.forward_features(x, masks)


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens, **kwargs,
    )
