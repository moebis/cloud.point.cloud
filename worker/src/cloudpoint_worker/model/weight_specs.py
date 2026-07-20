"""Exhaustive PyTorch-to-MLX weight layout specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cloudpoint_worker.model.config import ModelConfig

WeightTransform = Literal["identity", "conv2d", "conv_transpose2d"]


@dataclass(frozen=True)
class WeightSpec:
    source_key: str
    destination_key: str
    source_shape: tuple[int, ...]
    destination_shape: tuple[int, ...]
    transform: WeightTransform


def _spec(
    key: str,
    shape: tuple[int, ...],
    transform: WeightTransform = "identity",
) -> WeightSpec:
    if transform == "conv2d":
        destination_shape = (shape[0], shape[2], shape[3], shape[1])
    elif transform == "conv_transpose2d":
        destination_shape = (shape[1], shape[2], shape[3], shape[0])
    else:
        destination_shape = shape
    return WeightSpec(key, key, shape, destination_shape, transform)


def _transformer_block_specs(
    prefix: str,
    dimension: int,
    *,
    head_dimension: int | None,
) -> list[WeightSpec]:
    shapes: list[tuple[str, tuple[int, ...]]] = [
        ("norm1.weight", (dimension,)),
        ("norm1.bias", (dimension,)),
        ("attn.qkv.weight", (3 * dimension, dimension)),
        ("attn.qkv.bias", (3 * dimension,)),
    ]
    if head_dimension is not None:
        shapes.extend(
            [
                ("attn.q_norm.weight", (head_dimension,)),
                ("attn.q_norm.bias", (head_dimension,)),
                ("attn.k_norm.weight", (head_dimension,)),
                ("attn.k_norm.bias", (head_dimension,)),
            ]
        )
    shapes.extend(
        [
            ("attn.proj.weight", (dimension, dimension)),
            ("attn.proj.bias", (dimension,)),
            ("ls1.gamma", (dimension,)),
            ("norm2.weight", (dimension,)),
            ("norm2.bias", (dimension,)),
            ("mlp.fc1.weight", (4 * dimension, dimension)),
            ("mlp.fc1.bias", (4 * dimension,)),
            ("mlp.fc2.weight", (dimension, 4 * dimension)),
            ("mlp.fc2.bias", (dimension,)),
            ("ls2.gamma", (dimension,)),
        ]
    )
    return [_spec(f"{prefix}.{suffix}", shape) for suffix, shape in shapes]


def _backbone_specs(config: ModelConfig) -> list[WeightSpec]:
    dimension = config.embed_dim
    patch_count = (config.image_size // config.patch_size) ** 2
    specs = [
        _spec("aggregator.patch_embed.cls_token", (1, 1, dimension)),
        _spec("aggregator.patch_embed.pos_embed", (1, patch_count + 1, dimension)),
        _spec(
            "aggregator.patch_embed.register_tokens",
            (1, config.register_tokens, dimension),
        ),
        _spec("aggregator.patch_embed.mask_token", (1, dimension)),
        _spec(
            "aggregator.patch_embed.patch_embed.proj.weight",
            (dimension, 3, config.patch_size, config.patch_size),
            "conv2d",
        ),
        _spec("aggregator.patch_embed.patch_embed.proj.bias", (dimension,)),
    ]
    for index in range(config.depth):
        specs.extend(
            _transformer_block_specs(
                f"aggregator.patch_embed.blocks.{index}",
                dimension,
                head_dimension=None,
            )
        )
    specs.extend(
        [
            _spec("aggregator.patch_embed.norm.weight", (dimension,)),
            _spec("aggregator.patch_embed.norm.bias", (dimension,)),
        ]
    )
    return specs


def _aggregator_specs(config: ModelConfig) -> list[WeightSpec]:
    dimension = config.embed_dim
    specs = [
        _spec("aggregator.camera_token", (1, 2, 1, dimension)),
        _spec(
            "aggregator.register_token", (1, 2, config.register_tokens, dimension)
        ),
        _spec("aggregator.scale_token", (1, 2, 1, dimension)),
    ]
    specs.extend(_backbone_specs(config))
    head_dimension = dimension // config.heads
    for block_group in ("frame_blocks", "global_blocks"):
        for index in range(config.depth):
            specs.extend(
                _transformer_block_specs(
                    f"aggregator.{block_group}.{index}",
                    dimension,
                    head_dimension=head_dimension,
                )
            )
    return specs


def _camera_head_specs(config: ModelConfig) -> list[WeightSpec]:
    dimension = 2 * config.embed_dim
    specs = [_spec("camera_head.empty_pose_tokens", (1, 1, 9))]
    for index in range(4):
        specs.extend(
            _transformer_block_specs(
                f"camera_head.trunk.{index}", dimension, head_dimension=None
            )
        )
    specs.extend(
        [
            _spec("camera_head.token_norm.weight", (dimension,)),
            _spec("camera_head.token_norm.bias", (dimension,)),
            _spec("camera_head.trunk_norm.weight", (dimension,)),
            _spec("camera_head.trunk_norm.bias", (dimension,)),
            _spec("camera_head.embed_pose.weight", (dimension, 9)),
            _spec("camera_head.embed_pose.bias", (dimension,)),
            _spec("camera_head.poseLN_modulation.1.weight", (3 * dimension, dimension)),
            _spec("camera_head.poseLN_modulation.1.bias", (3 * dimension,)),
            _spec("camera_head.pose_branch.fc1.weight", (dimension // 2, dimension)),
            _spec("camera_head.pose_branch.fc1.bias", (dimension // 2,)),
            _spec("camera_head.pose_branch.fc2.weight", (9, dimension // 2)),
            _spec("camera_head.pose_branch.fc2.bias", (9,)),
        ]
    )
    return specs


def _depth_head_specs(config: ModelConfig) -> list[WeightSpec]:
    dimension = 2 * config.embed_dim
    channels = (256, 512, 1024, 1024)
    specs = [
        _spec("depth_head.norm.weight", (dimension,)),
        _spec("depth_head.norm.bias", (dimension,)),
    ]
    for index, output_channels in enumerate(channels):
        specs.extend(
            [
                _spec(
                    f"depth_head.projects.{index}.weight",
                    (output_channels, dimension, 1, 1),
                    "conv2d",
                ),
                _spec(f"depth_head.projects.{index}.bias", (output_channels,)),
            ]
        )
    specs.extend(
        [
            _spec(
                "depth_head.resize_layers.0.weight",
                (256, 256, 4, 4),
                "conv_transpose2d",
            ),
            _spec("depth_head.resize_layers.0.bias", (256,)),
            _spec(
                "depth_head.resize_layers.1.weight",
                (512, 512, 2, 2),
                "conv_transpose2d",
            ),
            _spec("depth_head.resize_layers.1.bias", (512,)),
            _spec(
                "depth_head.resize_layers.3.weight",
                (1024, 1024, 3, 3),
                "conv2d",
            ),
            _spec("depth_head.resize_layers.3.bias", (1024,)),
        ]
    )
    for index, input_channels in enumerate(channels, start=1):
        specs.append(
            _spec(
                f"depth_head.scratch.layer{index}_rn.weight",
                (256, input_channels, 3, 3),
                "conv2d",
            )
        )
    for index in range(1, 5):
        prefix = f"depth_head.scratch.refinenet{index}"
        specs.extend(
            [
                _spec(f"{prefix}.out_conv.weight", (256, 256, 1, 1), "conv2d"),
                _spec(f"{prefix}.out_conv.bias", (256,)),
            ]
        )
        residual_units = (2,) if index == 4 else (1, 2)
        for unit in residual_units:
            for convolution in (1, 2):
                specs.extend(
                    [
                        _spec(
                            f"{prefix}.resConfUnit{unit}.conv{convolution}.weight",
                            (256, 256, 3, 3),
                            "conv2d",
                        ),
                        _spec(
                            f"{prefix}.resConfUnit{unit}.conv{convolution}.bias",
                            (256,),
                        ),
                    ]
                )
    specs.extend(
        [
            _spec(
                "depth_head.scratch.output_conv1.weight",
                (128, 256, 3, 3),
                "conv2d",
            ),
            _spec("depth_head.scratch.output_conv1.bias", (128,)),
            _spec(
                "depth_head.scratch.output_conv2.0.weight",
                (32, 128, 3, 3),
                "conv2d",
            ),
            _spec("depth_head.scratch.output_conv2.0.bias", (32,)),
            _spec(
                "depth_head.scratch.output_conv2.2.weight",
                (2, 32, 1, 1),
                "conv2d",
            ),
            _spec("depth_head.scratch.output_conv2.2.bias", (2,)),
        ]
    )
    return specs


def build_weight_specs(config: ModelConfig) -> tuple[WeightSpec, ...]:
    """Return the complete pinned checkpoint layout in deterministic key order."""

    specs = [
        *_aggregator_specs(config),
        *_camera_head_specs(config),
        *_depth_head_specs(config),
    ]
    return tuple(specs)


__all__ = ["WeightSpec", "WeightTransform", "build_weight_specs"]
