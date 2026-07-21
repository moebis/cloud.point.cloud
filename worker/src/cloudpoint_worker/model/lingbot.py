"""Production Lingbot Map MLX model and prediction boundary."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import mlx.core as mx

from cloudpoint_worker import ENGINE_VERSION
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.model.config import ModelConfig
from cloudpoint_worker.model.weight_specs import WeightSpec, build_weight_specs
from cloudpoint_worker.model_prep.artifacts import (
    CONVERTED_FILENAME,
    MODEL_MANIFEST_FILENAME,
    WEIGHTS_MANIFEST_FILENAME,
)
from cloudpoint_worker.model_prep.provenance import (
    MODEL_FILENAME,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE,
    SOURCE_COMMIT,
)

_MODEL_FILES = frozenset(
    {CONVERTED_FILENAME, WEIGHTS_MANIFEST_FILENAME, MODEL_MANIFEST_FILENAME}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONVERTED_MODEL_SHA256 = (
    "eb966484923b5a205677b3ce7316d079c46fc6503bc9b6ac256b6e11560ea2e5"
)


@dataclass(frozen=True)
class FramePrediction:
    """One frame of dense reconstruction and decoded camera state."""

    depth: mx.array
    confidence: mx.array
    pose_encoding: mx.array
    intrinsics: mx.array
    camera_to_world: mx.array


@dataclass(frozen=True)
class FrameBatchPrediction:
    """Ordered frame predictions plus the four selected aggregator features."""

    frames: tuple[FramePrediction, ...]
    selected_features: dict[int, mx.array]

    @property
    def depth(self) -> mx.array:
        return mx.stack([frame.depth for frame in self.frames])

    @property
    def confidence(self) -> mx.array:
        return mx.stack([frame.confidence for frame in self.frames])

    @property
    def pose_encoding(self) -> mx.array:
        return mx.stack([frame.pose_encoding for frame in self.frames])

    @property
    def intrinsics(self) -> mx.array:
        return mx.stack([frame.intrinsics for frame in self.frames])

    @property
    def camera_to_world(self) -> mx.array:
        return mx.stack([frame.camera_to_world for frame in self.frames])

    @property
    def c2w(self) -> mx.array:
        return self.camera_to_world


@dataclass(frozen=True)
class _AggregatorTopology:
    selected_layers: tuple[int, ...]
    frame_blocks: tuple[int, ...]
    global_blocks: tuple[int, ...]
    patch_start: int


class _Backend(Protocol):
    def forward(
        self,
        images: mx.array,
        *,
        scale_frames: int,
        reset_cache: bool,
        append_cache: bool,
    ) -> tuple[dict[str, mx.array], dict[int, mx.array]]: ...


def _fault(code: str, message: str, **details: object) -> WorkerFault:
    return WorkerFault(code, message, False, dict(details))


def checkpoint_key_to_mlx_key(checkpoint_key: str) -> str:
    """Map one strict-converter destination key into the vendored MLX tree."""

    key = checkpoint_key.replace(
        "camera_head.poseLN_modulation.1.",
        "camera_head.poseLN_modulation_1.",
    )
    key = key.replace("depth_head.scratch.", "depth_head.scratch_")
    key = key.replace(
        "depth_head.scratch_output_conv2.0.",
        "depth_head.scratch_output_conv2_0.",
    )
    return key.replace(
        "depth_head.scratch_output_conv2.2.",
        "depth_head.scratch_output_conv2_1.",
    )


def _load_json(path: Path, *, maximum_bytes: int) -> object:
    if path.is_symlink() or not path.is_file():
        raise _fault("MODEL_INVALID_PATH", f"{path.name} must be a regular file")
    if path.stat().st_size > maximum_bytes:
        raise _fault("MODEL_INVALID_MANIFEST", f"{path.name} is too large")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text("utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise _fault(
            "MODEL_INVALID_MANIFEST", f"{path.name} is not canonical JSON"
        ) from error


def _require_mapping(value: object, filename: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _fault("MODEL_INVALID_MANIFEST", f"{filename} must be a JSON object")
    return cast(dict[str, object], value)


def _validate_model_directory(model_dir: Path) -> Path:
    if not model_dir.is_absolute() or model_dir.is_symlink() or not model_dir.is_dir():
        raise _fault("MODEL_INVALID_PATH", "converted model directory must be absolute")
    try:
        entries = {entry.name for entry in model_dir.iterdir()}
    except OSError as error:
        raise _fault(
            "MODEL_INVALID_PATH", "converted model directory is unreadable"
        ) from error
    if entries != _MODEL_FILES:
        raise _fault(
            "MODEL_INVALID_PATH",
            "converted model directory must contain exactly three prepared files",
            expected=sorted(_MODEL_FILES),
            actual=sorted(entries),
        )

    manifest = _require_mapping(
        _load_json(model_dir / MODEL_MANIFEST_FILENAME, maximum_bytes=32_768),
        MODEL_MANIFEST_FILENAME,
    )
    required_manifest_keys = {
        "schemaVersion",
        "modelIdentifier",
        "modelRevision",
        "sourceSHA256",
        "convertedSha256",
        "tensorCount",
        "mlxVersion",
        "engineVersion",
        "sourceCommit",
        "modelFilename",
        "modelSize",
        "conversionUTC",
    }
    if set(manifest) != required_manifest_keys:
        raise _fault("MODEL_INVALID_MANIFEST", "model manifest fields are not exact")
    expected_values: dict[str, object] = {
        "schemaVersion": 1,
        "modelIdentifier": MODEL_REPO,
        "modelRevision": MODEL_REVISION,
        "sourceSHA256": MODEL_SHA256,
        "tensorCount": 1_342,
        "mlxVersion": "0.32.0",
        "engineVersion": ENGINE_VERSION,
        "sourceCommit": SOURCE_COMMIT,
        "modelFilename": MODEL_FILENAME,
        "modelSize": MODEL_SIZE,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise _fault("MODEL_INVALID_MANIFEST", f"model manifest {key} is invalid")
    converted_sha256 = manifest.get("convertedSha256")
    if not isinstance(converted_sha256, str) or not _SHA256_PATTERN.fullmatch(
        converted_sha256
    ):
        raise _fault("MODEL_INVALID_MANIFEST", "converted model digest is invalid")
    if converted_sha256 != CONVERTED_MODEL_SHA256:
        raise _fault("MODEL_DIGEST_MISMATCH", "converted model is not the pinned build")
    if not isinstance(manifest.get("conversionUTC"), str):
        raise _fault("MODEL_INVALID_MANIFEST", "conversion timestamp is invalid")
    if importlib.metadata.version("mlx") != "0.32.0":
        raise _fault("RUNTIME_INCOMPATIBLE", "CloudPoint requires MLX 0.32.0")

    weights_path = model_dir / CONVERTED_FILENAME
    if weights_path.is_symlink() or not weights_path.is_file():
        raise _fault("MODEL_INVALID_PATH", "converted weights must be a regular file")
    try:
        with weights_path.open("rb") as source:
            actual_digest = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as error:
        raise _fault(
            "MODEL_INVALID_PATH", "converted weights are unreadable"
        ) from error
    if actual_digest != converted_sha256:
        raise _fault("MODEL_DIGEST_MISMATCH", "converted model digest does not match")

    rows_value = _load_json(
        model_dir / WEIGHTS_MANIFEST_FILENAME, maximum_bytes=2_000_000
    )
    if not isinstance(rows_value, list) or len(rows_value) != 1_342:
        raise _fault(
            "MODEL_INVALID_MANIFEST", "weight manifest tensor count is invalid"
        )
    rows: dict[str, dict[str, object]] = {}
    row_keys = {
        "sourceKey",
        "destinationKey",
        "sourceShape",
        "destinationShape",
        "sourceDtype",
        "destinationDtype",
        "transform",
        "sha256",
    }
    for value in rows_value:
        if not isinstance(value, dict) or set(value) != row_keys:
            raise _fault("MODEL_INVALID_MANIFEST", "weight manifest row is invalid")
        row = cast(dict[str, object], value)
        key = row.get("destinationKey")
        if not isinstance(key, str) or key in rows:
            raise _fault(
                "MODEL_INVALID_MANIFEST", "weight manifest keys are not unique"
            )
        rows[key] = row

    for spec in build_weight_specs(ModelConfig()):
        row = rows.get(spec.destination_key)
        if row is None:
            raise _fault("MODEL_INVALID_MANIFEST", "weight manifest is incomplete")
        digest = row.get("sha256")
        if (
            row.get("sourceKey") != spec.source_key
            or row.get("sourceShape") != list(spec.source_shape)
            or row.get("destinationShape") != list(spec.destination_shape)
            or row.get("destinationDtype") != "float16"
            or row.get("transform") != spec.transform
            or not isinstance(digest, str)
            or not _SHA256_PATTERN.fullmatch(digest)
        ):
            raise _fault(
                "MODEL_INVALID_MANIFEST",
                f"weight manifest row {spec.destination_key} is invalid",
            )
    return weights_path


class _VendorBackendAdapter:
    def __init__(self, model: Any, selected_layers: tuple[int, ...]) -> None:
        self._model = model
        self._selected_layers = selected_layers

    @classmethod
    def load(cls, weights_path: Path, config: ModelConfig) -> _VendorBackendAdapter:
        from mlx.utils import tree_flatten

        from cloudpoint_worker.model._vendor.lingbot_map_mlx.models.gct_stream import (
            GCTStream,
        )

        model = GCTStream(
            img_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            enable_camera=True,
            enable_depth=True,
            enable_point=False,
            enable_local_point=False,
            kv_cache_sliding_window=config.kv_cache_sliding_window,
            kv_cache_scale_frames=config.kv_cache_scale_frames,
            camera_num_iterations=config.camera_refinement_iterations,
            enable_3d_rope=config.enable_3d_rope,
            enable_camera_3d_rope=config.enable_camera_3d_rope,
            max_frame_num=config.maximum_frames,
            use_sdpa=True,
        )
        parameters = dict(tree_flatten(model.parameters()))
        loaded = mx.load(str(weights_path))
        if not isinstance(loaded, Mapping):
            raise _fault("MODEL_INVALID_WEIGHTS", "SafeTensors root is not a mapping")
        specs = build_weight_specs(config)
        source_keys = {spec.destination_key for spec in specs}
        if set(loaded) != source_keys:
            raise _fault("MODEL_INVALID_WEIGHTS", "SafeTensors keys are not exact")

        mapped: dict[str, mx.array] = {}
        for spec in specs:
            value = loaded[spec.destination_key]
            if not isinstance(value, mx.array) or value.shape != spec.destination_shape:
                raise _fault(
                    "MODEL_INVALID_WEIGHTS",
                    f"SafeTensors shape for {spec.destination_key} is invalid",
                )
            destination = checkpoint_key_to_mlx_key(spec.destination_key)
            if destination in mapped:
                raise _fault("MODEL_INVALID_WEIGHTS", "MLX weight mapping collides")
            mapped[destination] = value

        if set(mapped) != set(parameters):
            missing = sorted(set(parameters) - set(mapped))
            extra = sorted(set(mapped) - set(parameters))
            raise _fault(
                "MODEL_INVALID_WEIGHTS",
                "MLX parameter coverage is not a strict bijection",
                missing=missing[:5],
                extra=extra[:5],
            )
        for key, value in mapped.items():
            if value.shape != parameters[key].shape:
                raise _fault("MODEL_INVALID_WEIGHTS", f"MLX shape for {key} is invalid")
        model.load_weights(list(mapped.items()), strict=True)
        model.eval()
        return cls(model, config.selected_layers)

    def forward(
        self,
        images: mx.array,
        *,
        scale_frames: int,
        reset_cache: bool,
        append_cache: bool,
    ) -> tuple[dict[str, mx.array], dict[int, mx.array]]:
        if reset_cache:
            self._model.clean_kv_cache()
        if not append_cache:
            self._model._set_skip_append(True)
        try:
            features, patch_start = self._model._aggregate_features(
                images,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=images.shape[1],
            )
            predictions: dict[str, mx.array] = {}
            predictions.update(
                self._model._predict_camera(
                    features,
                    causal_inference=True,
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=images.shape[1],
                )
            )
            predictions.update(
                self._model._predict_depth(features, images, patch_start)
            )
            selected = dict(zip(self._selected_layers, features, strict=True))
            mx.eval(
                predictions["pose_enc"],
                predictions["depth"],
                predictions["depth_conf"],
                *selected.values(),
            )
            return predictions, selected
        finally:
            if not append_cache:
                self._model._set_skip_append(False)


class LingbotMap:
    """Strict, stdout-neutral production wrapper around Lingbot Map MLX."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._validate_config()
        self.aggregator = _AggregatorTopology(
            selected_layers=self.config.selected_layers,
            frame_blocks=tuple(range(self.config.depth)),
            global_blocks=tuple(range(self.config.depth)),
            patch_start=1 + self.config.register_tokens + 1,
        )
        self.camera_head: object | None = object()
        self.depth_head: object | None = object()
        self.point_head: None = None
        self._backend: _Backend | None = None
        self._scale_frames: int | None = None
        self._image_size: tuple[int, int] | None = None

    def _validate_config(self) -> None:
        fixed = (
            self.config.image_size,
            self.config.patch_size,
            self.config.embed_dim,
            self.config.depth,
            self.config.heads,
            self.config.register_tokens,
            self.config.selected_layers,
        )
        expected = (518, 14, 1024, 24, 16, 4, (4, 11, 17, 23))
        if fixed != expected:
            raise _fault(
                "MODEL_UNSUPPORTED_TOPOLOGY",
                "only the pinned Lingbot topology is supported",
            )
        if (
            self.config.kv_cache_sliding_window < 1
            or self.config.kv_cache_scale_frames < 1
            or self.config.camera_refinement_iterations < 1
            or self.config.maximum_frames < 1
        ):
            raise _fault("MODEL_INVALID_CONFIG", "model configuration must be positive")

    @classmethod
    def load(cls, model_dir: Path, *, config: ModelConfig | None = None) -> LingbotMap:
        model = cls(config)
        weights_path = _validate_model_directory(model_dir)
        try:
            with redirect_stdout(sys.stderr):
                model._backend = _VendorBackendAdapter.load(weights_path, model.config)
        except WorkerFault:
            raise
        except Exception as error:
            raise _fault(
                "MODEL_LOAD_FAILED", "Lingbot MLX model could not be loaded"
            ) from error
        return model

    @classmethod
    def weight_specs(cls) -> tuple[WeightSpec, ...]:
        return build_weight_specs(ModelConfig())

    def _install_backend_for_testing(self, backend: _Backend) -> None:
        """Install a tiny deterministic backend without weakening production load."""

        self._backend = backend

    def _prepare_images(self, images: mx.array) -> mx.array:
        if not isinstance(images, mx.array):
            raise _fault("MODEL_INVALID_INPUT", "images must be an MLX array")
        if images.ndim == 3:
            images = images[None, None]
        elif images.ndim == 4:
            images = images[None]
        elif images.ndim != 5:
            raise _fault("MODEL_INVALID_INPUT", "images must be HWC, SHWC, or BSHWC")
        if images.shape[0] != 1 or images.shape[1] < 1 or images.shape[-1] != 3:
            raise _fault(
                "MODEL_INVALID_INPUT",
                "only one non-empty RGB sequence is supported",
            )
        height, width = images.shape[2:4]
        if (
            height < self.config.patch_size
            or width < self.config.patch_size
            or height % self.config.patch_size
            or width % self.config.patch_size
            or height > 8_192
            or width > 8_192
            or images.size > 134_217_728
        ):
            raise _fault(
                "MODEL_INVALID_INPUT",
                "image dimensions must be bounded multiples of the 14-pixel patch size",
            )
        minimum = float(mx.min(images).item())
        maximum = float(mx.max(images).item())
        if not (0.0 <= minimum <= maximum <= 1.0):
            raise _fault(
                "MODEL_INVALID_INPUT", "model input must be finite raw RGB in [0, 1]"
            )
        return images.astype(mx.float16)

    def _execute(
        self,
        images: mx.array,
        *,
        scale_frames: int,
        reset_cache: bool,
        append_cache: bool,
    ) -> FrameBatchPrediction:
        if self._backend is None:
            raise _fault("MODEL_NOT_LOADED", "Lingbot model weights are not loaded")
        try:
            with redirect_stdout(sys.stderr):
                raw, selected = self._backend.forward(
                    images,
                    scale_frames=scale_frames,
                    reset_cache=reset_cache,
                    append_cache=append_cache,
                )
        except WorkerFault:
            raise
        except Exception as error:
            raise _fault(
                "MODEL_INFERENCE_FAILED", "Lingbot MLX inference failed"
            ) from error
        return self._decode_batch(raw, selected, images.shape[2:4])

    def _decode_batch(
        self,
        raw: dict[str, mx.array],
        selected: dict[int, mx.array],
        image_size: tuple[int, int],
    ) -> FrameBatchPrediction:
        from cloudpoint_worker.model._vendor.lingbot_map_mlx.utils.geometry import (
            closed_form_inverse_se3_general,
        )
        from cloudpoint_worker.model._vendor.lingbot_map_mlx.utils.pose_enc import (
            pose_encoding_to_extri_intri,
        )

        try:
            pose = raw["pose_enc"].astype(mx.float32)
            depth = raw["depth"]
            confidence = raw["depth_conf"]
        except (KeyError, AttributeError) as error:
            raise _fault(
                "MODEL_INVALID_OUTPUT", "model output tensors are incomplete"
            ) from error
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if (
            pose.ndim != 3
            or pose.shape[0] != 1
            or pose.shape[-1] != 9
            or depth.shape != (1, pose.shape[1], *image_size)
            or confidence.shape != (1, pose.shape[1], *image_size)
        ):
            raise _fault(
                "MODEL_INVALID_OUTPUT", "model output tensor shapes are invalid"
            )
        world_to_camera, intrinsics = pose_encoding_to_extri_intri(
            pose, image_size_hw=image_size
        )
        if intrinsics is None:
            raise _fault("MODEL_INVALID_OUTPUT", "model did not decode intrinsics")
        bottom = mx.zeros((1, pose.shape[1], 1, 4), dtype=mx.float32)
        bottom = bottom.at[..., 0, 3].add(1.0)
        world_to_camera_4x4 = mx.concatenate(
            [world_to_camera.astype(mx.float32), bottom], axis=-2
        )
        camera_to_world = closed_form_inverse_se3_general(world_to_camera_4x4)
        depth = depth.astype(mx.float16)
        confidence = confidence.astype(mx.float16)
        intrinsics = intrinsics.astype(mx.float32)
        camera_to_world = camera_to_world.astype(mx.float32)
        mx.eval(depth, confidence, pose, intrinsics, camera_to_world)
        frames = tuple(
            FramePrediction(
                depth=depth[0, index],
                confidence=confidence[0, index],
                pose_encoding=pose[0, index],
                intrinsics=intrinsics[0, index],
                camera_to_world=camera_to_world[0, index],
            )
            for index in range(pose.shape[1])
        )
        clean_features: dict[int, mx.array] = {}
        for layer in self.config.selected_layers:
            feature = selected.get(layer)
            if feature is None or feature.ndim != 4 or feature.shape[0] != 1:
                raise _fault(
                    "MODEL_INVALID_OUTPUT", "selected feature tensors are incomplete"
                )
            clean_features[layer] = feature[0]
        return FrameBatchPrediction(frames, clean_features)

    def forward_scale(self, images: mx.array) -> FrameBatchPrediction:
        prepared = self._prepare_images(images)
        scale_frames = prepared.shape[1]
        prediction = self._execute(
            prepared,
            scale_frames=scale_frames,
            reset_cache=True,
            append_cache=True,
        )
        self._scale_frames = scale_frames
        self._image_size = prepared.shape[2:4]
        return prediction

    def forward_frame(self, image: mx.array, append_cache: bool) -> FramePrediction:
        if self._scale_frames is None or self._image_size is None:
            raise _fault("MODEL_CACHE_UNINITIALIZED", "forward_scale must run first")
        prepared = self._prepare_images(image)
        if prepared.shape[1] != 1 or prepared.shape[2:4] != self._image_size:
            raise _fault("MODEL_INVALID_INPUT", "streaming frame dimensions changed")
        return self._execute(
            prepared,
            scale_frames=self._scale_frames,
            reset_cache=False,
            append_cache=append_cache,
        ).frames[0]

    def infer_direct(
        self, images: mx.array, scale_frames: int = 8
    ) -> FrameBatchPrediction:
        prepared = self._prepare_images(images)
        frame_count = prepared.shape[1]
        if scale_frames < 1:
            raise _fault("MODEL_INVALID_INPUT", "scale_frames must be positive")
        scale_frames = min(scale_frames, frame_count)
        batches = [
            self._execute(
                prepared[:, :scale_frames],
                scale_frames=scale_frames,
                reset_cache=True,
                append_cache=True,
            )
        ]
        for index in range(scale_frames, frame_count):
            batches.append(
                self._execute(
                    prepared[:, index : index + 1],
                    scale_frames=scale_frames,
                    reset_cache=False,
                    append_cache=True,
                )
            )
        self._scale_frames = scale_frames
        self._image_size = prepared.shape[2:4]
        features = {
            layer: mx.concatenate(
                [batch.selected_features[layer] for batch in batches], axis=0
            )
            for layer in self.config.selected_layers
        }
        return FrameBatchPrediction(
            tuple(frame for batch in batches for frame in batch.frames),
            features,
        )


__all__ = [
    "CONVERTED_MODEL_SHA256",
    "FrameBatchPrediction",
    "FramePrediction",
    "LingbotMap",
    "checkpoint_key_to_mlx_key",
]
