"""DPT Head for dense prediction tasks — MLX port.

All convolutions use NHWC layout (MLX native).
PyTorch's NCHW reshapes become NHWC reshapes.
FloatFunctional.add() replaced with plain +.
F.interpolate replaced with nn.Upsample.
"""

from typing import List, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from lingbot_map_mlx.heads.head_act import activate_head
from lingbot_map_mlx.heads.utils import create_uv_grid, position_grid_to_embed


def custom_interpolate(x: mx.array, size=None, scale_factor=None, mode="linear", align_corners=True) -> mx.array:
    """Bilinear interpolation for NHWC tensors."""
    if size is not None:
        target_h, target_w = size
        src_h, src_w = x.shape[1], x.shape[2]
        sf_h = target_h / src_h
        sf_w = target_w / src_w
    elif scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            sf_h = sf_w = float(scale_factor)
        else:
            sf_h, sf_w = float(scale_factor), float(scale_factor)
    else:
        raise ValueError("Either size or scale_factor must be provided")

    upsample = nn.Upsample(scale_factor=(sf_h, sf_w), mode=mode, align_corners=align_corners)
    return upsample(x)


class ResidualConvUnit(nn.Module):
    def __init__(self, features, bn=False, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # PyTorch uses ReLU(inplace=True) which modifies x in-place.
        # The residual add uses the relu'd x, not the original.
        x = nn.relu(x)
        out = self.conv1(x)
        out = nn.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, deconv=False, bn=False, expand=False,
                 align_corners=True, size=None, has_residual=True, groups=1):
        super().__init__()
        self.align_corners = align_corners
        out_features = features // 2 if expand else features

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, bn=bn)
        self.resConfUnit2 = ResidualConvUnit(features, bn=bn)
        self.size = size

    def __call__(self, *xs, size=None) -> mx.array:
        output = xs[0]

        if self.has_residual and len(xs) > 1:
            res = self.resConfUnit1(xs[1])
            output = output + res

        output = self.resConfUnit2(output)

        if size is None and self.size is None:
            sf = 2.0
        elif size is None:
            sf = None
            target_size = self.size
        else:
            sf = None
            target_size = size

        if sf is not None:
            output = custom_interpolate(output, scale_factor=sf, mode="linear", align_corners=self.align_corners)
        else:
            output = custom_interpolate(output, size=target_size, mode="linear", align_corners=self.align_corners)

        output = self.out_conv(output)
        return output


class DPTHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [0, 1, 2, 3],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # 1x1 projection convolutions
        self.projects = [
            nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ]

        # Resize layers
        self.resize_layers = [
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ]

        # Scratch layers (1x1 conv projections)
        self.scratch_layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch_layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch_layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch_layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)

        self.scratch_refinenet1 = FeatureFusionBlock(features)
        self.scratch_refinenet2 = FeatureFusionBlock(features)
        self.scratch_refinenet3 = FeatureFusionBlock(features)
        self.scratch_refinenet4 = FeatureFusionBlock(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch_output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch_output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            conv2_in = head_features_1 // 2
            self.scratch_output_conv2_0 = nn.Conv2d(conv2_in, head_features_2, kernel_size=3, stride=1, padding=1)
            self.scratch_output_conv2_1 = nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0)

    def __call__(
        self,
        aggregated_tokens_list: List[mx.array],
        images: mx.array,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ):
        # images: (B, S, H, W, C) in MLX NHWC
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        S = aggregated_tokens_list[0].shape[1]

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        all_preds, all_conf = [], []
        for fs in range(0, S, frames_chunk_size):
            fe = min(fs + frames_chunk_size, S)
            if self.feature_only:
                all_preds.append(self._forward_impl(aggregated_tokens_list, images, patch_start_idx, fs, fe))
            else:
                p, c = self._forward_impl(aggregated_tokens_list, images, patch_start_idx, fs, fe)
                all_preds.append(p)
                all_conf.append(c)

        if self.feature_only:
            return mx.concatenate(all_preds, axis=1)
        return mx.concatenate(all_preds, axis=1), mx.concatenate(all_conf, axis=1)

    def _forward_impl(
        self, aggregated_tokens_list, images, patch_start_idx,
        frames_start_idx=None, frames_end_idx=None,
    ):
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            B_cur, S_cur = x.shape[0], x.shape[1]
            x = x.reshape(B_cur * S_cur, -1, x.shape[-1])

            x = self.norm(x)

            # Reshape from (BS, N, C) to NHWC: (BS, patch_h, patch_w, C)
            x = x.reshape(B_cur * S_cur, patch_h, patch_w, x.shape[-1])

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)

        # Fuse features
        out = self._scratch_forward(out)

        target_h = int(patch_h * self.patch_size / self.down_ratio)
        target_w = int(patch_w * self.patch_size / self.down_ratio)
        out = custom_interpolate(out, size=(target_h, target_w), mode="linear", align_corners=True)

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.reshape(B_cur, S_cur, *out.shape[1:])

        # output_conv2: Conv -> ReLU -> Conv
        out = nn.relu(self.scratch_output_conv2_0(out))
        out = self.scratch_output_conv2_1(out)

        # activate_head expects NHWC which is already the case
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        preds = preds.reshape(B_cur, S_cur, *preds.shape[1:])
        conf = conf.reshape(B_cur, S_cur, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: mx.array, W: int, H: int, ratio: float = 0.1) -> mx.array:
        # x: (BS, pH, pW, C)
        patch_h = x.shape[1]
        patch_w = x.shape[2]
        C = x.shape[3]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H)
        pos_embed = position_grid_to_embed(pos_embed, C)
        pos_embed = pos_embed * ratio
        # pos_embed: (pH, pW, C) -> (1, pH, pW, C)
        pos_embed = pos_embed[None]
        return x + pos_embed

    def _scratch_forward(self, features: List[mx.array]) -> mx.array:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch_layer1_rn(layer_1)
        layer_2_rn = self.scratch_layer2_rn(layer_2)
        layer_3_rn = self.scratch_layer3_rn(layer_3)
        layer_4_rn = self.scratch_layer4_rn(layer_4)
        mx.eval(layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn)

        out = self.scratch_refinenet4(layer_4_rn, size=(layer_3_rn.shape[1], layer_3_rn.shape[2]))
        out = self.scratch_refinenet3(out, layer_3_rn, size=(layer_2_rn.shape[1], layer_2_rn.shape[2]))
        out = self.scratch_refinenet2(out, layer_2_rn, size=(layer_1_rn.shape[1], layer_1_rn.shape[2]))
        out = self.scratch_refinenet1(out, layer_1_rn)
        out = self.scratch_output_conv1(out)
        return out
