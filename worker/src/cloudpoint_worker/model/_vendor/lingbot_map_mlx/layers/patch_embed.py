from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,H,W,C) -> (B,N,D)

    MLX uses NHWC layout (channels-last).
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten_embedding: bool = True,
    ):
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_HW, stride=patch_HW,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) in MLX NHWC layout
        B, H, W, C = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input height {H} not divisible by patch height {patch_H}"
        assert W % patch_W == 0, f"Input width {W} not divisible by patch width {patch_W}"

        x = self.proj(x)  # (B, H', W', embed_dim)
        pH, pW = x.shape[1], x.shape[2]

        if self.flatten_embedding:
            x = x.reshape(B, pH * pW, self.embed_dim)  # (B, N, D)
        x = self.norm(x)
        if not self.flatten_embedding and x.ndim == 3:
            x = x.reshape(B, pH, pW, self.embed_dim)
        return x
