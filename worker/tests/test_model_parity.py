"""Topology, public prediction API, and opt-in real-model checks."""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.model.config import ModelConfig
from cloudpoint_worker.model.lingbot import (
    CONVERTED_MODEL_SHA256,
    FrameBatchPrediction,
    LingbotMap,
)


class _DeterministicBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool, bool]] = []

    def forward(
        self,
        images: mx.array,
        *,
        scale_frames: int,
        reset_cache: bool,
        append_cache: bool,
    ) -> tuple[dict[str, mx.array], dict[int, mx.array]]:
        print("vendor diagnostic")
        batch, frames, height, width, _ = images.shape
        self.calls.append((frames, reset_cache, append_cache))
        pose = np.zeros((batch, frames, 9), dtype=np.float32)
        pose[..., 6] = 1.0
        pose[..., 7:] = 1.0
        depth = np.full((batch, frames, height, width, 1), 2.0, dtype=np.float32)
        confidence = np.full((batch, frames, height, width), 2.5, dtype=np.float32)
        features = {
            layer: mx.full((batch, frames, 2, 4), float(layer), dtype=mx.float16)
            for layer in (4, 11, 17, 23)
        }
        return (
            {
                "pose_enc": mx.array(pose),
                "depth": mx.array(depth),
                "depth_conf": mx.array(confidence),
            },
            features,
        )


def _model_with_backend() -> tuple[LingbotMap, _DeterministicBackend]:
    model = LingbotMap(ModelConfig())
    backend = _DeterministicBackend()
    model._install_backend_for_testing(backend)
    return model, backend


def test_topology_matches_pinned_upstream() -> None:
    model = LingbotMap(ModelConfig())

    assert (model.config.embed_dim, model.config.depth, model.config.heads) == (
        1024,
        24,
        16,
    )
    assert model.config.patch_size == 14
    assert model.aggregator.selected_layers == (4, 11, 17, 23)
    assert len(model.aggregator.frame_blocks) == 24
    assert len(model.aggregator.global_blocks) == 24
    assert model.aggregator.patch_start == 6
    assert model.point_head is None
    assert model.depth_head is not None
    assert model.camera_head is not None
    assert len(model.weight_specs()) == 1_342


def test_converted_model_digest_is_a_compiled_trust_anchor() -> None:
    assert (
        CONVERTED_MODEL_SHA256
        == "eb966484923b5a205677b3ce7316d079c46fc6503bc9b6ac256b6e11560ea2e5"
    )


def test_infer_direct_returns_typed_predictions_and_preserves_all_frames(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, backend = _model_with_backend()
    images = mx.full((3, 28, 42, 3), 0.5, dtype=mx.float32)

    result = model.infer_direct(images, scale_frames=2)

    assert isinstance(result, FrameBatchPrediction)
    assert len(result.frames) == 3
    assert result.frames[0].depth.shape == (28, 42)
    assert result.frames[0].depth.dtype == mx.float16
    assert result.frames[0].confidence.dtype == mx.float16
    assert result.frames[0].pose_encoding.shape == (9,)
    assert result.frames[0].pose_encoding.dtype == mx.float32
    assert result.frames[0].intrinsics.shape == (3, 3)
    assert result.frames[0].camera_to_world.shape == (4, 4)
    np.testing.assert_allclose(np.asarray(result.frames[0].camera_to_world), np.eye(4))
    assert set(result.selected_features) == {4, 11, 17, 23}
    assert result.selected_features[4].shape == (3, 2, 4)
    assert backend.calls == [(2, True, True), (1, False, True)]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("vendor diagnostic") == 2


def test_forward_frame_can_read_without_appending_to_cache() -> None:
    model, backend = _model_with_backend()
    model.forward_scale(mx.full((2, 28, 42, 3), 0.25))

    prediction = model.forward_frame(mx.full((28, 42, 3), 0.25), append_cache=False)

    assert prediction.depth.shape == (28, 42)
    assert backend.calls[-1] == (1, False, False)


def test_infer_direct_clamps_scale_prefix_to_a_short_sequence() -> None:
    model, backend = _model_with_backend()
    images = mx.full((3, 28, 42, 3), 0.25)

    result = model.infer_direct(images)

    assert len(result.frames) == 3
    assert backend.calls == [(3, True, True)]


@pytest.mark.parametrize(
    "images",
    [
        mx.zeros((1, 28, 42)),
        mx.zeros((1, 28, 43, 3)),
        mx.full((1, 28, 42, 3), 1.5),
    ],
)
def test_model_rejects_invalid_raw_rgb_inputs(images: mx.array) -> None:
    model, _ = _model_with_backend()

    with pytest.raises(WorkerFault, match="MODEL_INVALID_INPUT"):
        model.forward_scale(images)


def test_load_rejects_relative_model_directory_before_reading() -> None:
    with pytest.raises(WorkerFault, match="MODEL_INVALID_PATH"):
        LingbotMap.load(Path("relative/model"))


def _real_model_dir() -> Path:
    value = os.environ.get("CLOUDPOINT_MODEL_DIR")
    if not value:
        pytest.skip("CLOUDPOINT_MODEL_DIR is not set")
    return Path(value)


def _real_inputs() -> mx.array:
    configured = os.environ.get("CLOUDPOINT_REAL_FRAMES")
    directory = (
        Path(configured)
        if configured
        else Path(__file__).parent / "fixtures" / "courthouse"
    )
    paths = sorted(directory.glob("*.png")) or sorted(directory.glob("*.jpg"))
    if len(paths) < 2:
        pytest.skip("real-model frame fixture is unavailable")
    arrays = []
    for path in paths[:8]:
        image = Image.open(path).convert("RGB")
        width = 518
        height = round((image.height * width / image.width) / 14) * 14
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
    return mx.array(np.stack(arrays))


@pytest.mark.real_model
def test_real_model_loads_every_weight_and_predicts_scene_geometry() -> None:
    model = LingbotMap.load(_real_model_dir())
    images = _real_inputs()

    result = model.infer_direct(images, scale_frames=min(8, images.shape[0]))
    depth = np.stack(
        [np.asarray(frame.depth, dtype=np.float32) for frame in result.frames]
    )
    confidence = np.stack(
        [np.asarray(frame.confidence, dtype=np.float32) for frame in result.frames]
    )
    camera_centers = np.stack(
        [np.asarray(frame.camera_to_world[:3, 3]) for frame in result.frames]
    )

    assert np.isfinite(depth).mean() == 1.0
    assert (depth > 0).mean() > 0.99
    assert (confidence > 1.5).mean() > 0.05
    translation_span = np.linalg.norm(
        camera_centers.max(axis=0) - camera_centers.min(axis=0)
    )
    assert translation_span > 1e-4
