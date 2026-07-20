"""Focused numerical and weight-mapping checks for the MLX topology."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from cloudpoint_worker.model.layers import attention
from cloudpoint_worker.model.lingbot import checkpoint_key_to_mlx_key


def test_attention_matches_float32_scaled_dot_product_reference() -> None:
    q = mx.array(
        [[[[1.0, 0.0], [0.0, 1.0]]]],
        dtype=mx.float16,
    )
    k = mx.array(
        [[[[1.0, 0.0], [0.0, 1.0]]]],
        dtype=mx.float16,
    )
    v = mx.array(
        [[[[2.0, 4.0], [8.0, 16.0]]]],
        dtype=mx.float16,
    )

    actual = np.asarray(attention(q, k, v, mask=None), dtype=np.float32)
    scores = np.asarray(q, dtype=np.float32) @ np.swapaxes(
        np.asarray(k, dtype=np.float32), -1, -2
    )
    scores *= 2**-0.5
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = weights @ np.asarray(v, dtype=np.float32)

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    assert attention(q, k, v, mask=None).dtype == mx.float16


def test_attention_boolean_mask_excludes_disallowed_keys() -> None:
    q = mx.ones((1, 1, 1, 2), dtype=mx.float32)
    k = mx.ones((1, 1, 2, 2), dtype=mx.float32)
    v = mx.array([[[[3.0, 5.0], [100.0, 200.0]]]])
    mask = mx.array([[[[True, False]]]])

    actual = np.asarray(attention(q, k, v, mask=mask))

    np.testing.assert_allclose(actual, np.array([[[[3.0, 5.0]]]]))


def test_checkpoint_key_mapping_covers_the_three_structural_renames() -> None:
    assert (
        checkpoint_key_to_mlx_key("camera_head.poseLN_modulation.1.weight")
        == "camera_head.poseLN_modulation_1.weight"
    )
    assert (
        checkpoint_key_to_mlx_key("depth_head.scratch.layer1_rn.weight")
        == "depth_head.scratch_layer1_rn.weight"
    )
    assert (
        checkpoint_key_to_mlx_key("depth_head.scratch.output_conv2.2.bias")
        == "depth_head.scratch_output_conv2_1.bias"
    )
