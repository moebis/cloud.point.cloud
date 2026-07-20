"""Deterministic Lingbot Map crop-mode image preprocessing.

The transform intentionally follows ``lingbot_map/utils/load_fn.py`` from
Robbyant/lingbot-map commit 7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2
(Apache-2.0). In particular, crop mode fixes the width at 518 pixels and
rounds the aspect-derived height to the nearest 14-pixel patch row.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SOURCE_COMMIT = "7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"
SOURCE_REPOSITORY = "https://github.com/Robbyant/lingbot-map"
TARGET_WIDTH = 518
PATCH_SIZE = 14
MAX_SOURCE_DIMENSION = 8192
MAX_SOURCE_PIXELS = 33_554_432

_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FixtureProvenanceError(ValueError):
    """A committed upstream fixture does not match its pinned provenance."""


class ImageBoundsError(ValueError):
    """A source image exceeds CloudPoint's stable safe-decoding bounds."""

    code = "IMAGE_BOUNDS_EXCEEDED"

    def __init__(self) -> None:
        super().__init__(
            f"{self.code}: source image exceeds {MAX_SOURCE_DIMENSION} pixels "
            f"per dimension or {MAX_SOURCE_PIXELS} total pixels"
        )


@dataclass(frozen=True)
class PreprocessedFrame:
    """One orientation-corrected frame and its model-space metadata.

    Sizes use ``(width, height)``. ``model_to_source`` maps model pixel-center
    coordinates back into the orientation-corrected source image used by the
    pinned transform.
    """

    rgb: np.ndarray
    normalized: np.ndarray
    model_to_source: np.ndarray
    source_size: tuple[int, int]
    model_size: tuple[int, int]


def _model_to_source_transform(
    source_size: tuple[int, int], resized_size: tuple[int, int], crop_top: int
) -> np.ndarray:
    source_width, source_height = source_size
    resized_width, resized_height = resized_size
    scale_x = source_width / resized_width
    scale_y = source_height / resized_height
    return np.array(
        [
            [scale_x, 0.0, 0.5 * scale_x - 0.5],
            [0.0, scale_y, (crop_top + 0.5) * scale_y - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _validate_source_size(size: tuple[int, int]) -> None:
    width, height = size
    if (
        width <= 0
        or height <= 0
        or width > MAX_SOURCE_DIMENSION
        or height > MAX_SOURCE_DIMENSION
        or width * height > MAX_SOURCE_PIXELS
    ):
        raise ImageBoundsError


def preprocess_image(path: Path) -> PreprocessedFrame:
    """Apply the pinned Lingbot crop-mode transform to one JPEG or PNG."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                # Image.open() parses only enough data to identify the image. Check
                # header dimensions before EXIF transpose triggers pixel decoding.
                _validate_source_size(opened.size)
                image = ImageOps.exif_transpose(opened)

                # Match the pinned source exactly. It composites RGBA over opaque
                # white, then uses ordinary RGB conversion without ICC management.
                if image.mode == "RGBA":
                    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, image)
                image = image.convert("RGB")

                source_size = image.size
                source_width, source_height = source_size
                resized_width = TARGET_WIDTH
                resized_height = (
                    round(source_height * (resized_width / source_width) / PATCH_SIZE)
                    * PATCH_SIZE
                )
                if resized_height <= 0:
                    raise ValueError(
                        "source aspect ratio produces a zero-height patch grid"
                    )

                image = image.resize(
                    (resized_width, resized_height), Image.Resampling.BICUBIC
                )
                crop_top = 0
                if resized_height > TARGET_WIDTH:
                    crop_top = (resized_height - TARGET_WIDTH) // 2
                    image = image.crop(
                        (0, crop_top, resized_width, crop_top + TARGET_WIDTH)
                    )

                model_size = image.size
                pixels = np.asarray(image, dtype=np.uint8)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageBoundsError from error

    rgb = pixels.astype(np.float32) / np.float32(255.0)
    normalized = (rgb - _RGB_MEAN) / _RGB_STD
    transform = _model_to_source_transform(
        source_size, (resized_width, resized_height), crop_top
    )

    return PreprocessedFrame(
        rgb=np.ascontiguousarray(rgb),
        normalized=np.ascontiguousarray(normalized),
        model_to_source=transform,
        source_size=source_size,
        model_size=model_size,
    )


def _fixture_error(message: str) -> FixtureProvenanceError:
    return FixtureProvenanceError(f"courthouse fixture provenance: {message}")


def verify_courthouse_fixtures(directory: Path) -> tuple[Path, ...]:
    """Verify and return the nine committed courthouse fixture paths."""

    provenance_path = directory / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fixture_error("provenance.json is unavailable or invalid") from error

    if provenance.get("schemaVersion") != 1:
        raise _fixture_error("unsupported schemaVersion")
    if provenance.get("sourceRepository") != SOURCE_REPOSITORY:
        raise _fixture_error("unexpected sourceRepository")
    if provenance.get("sourceCommit") != SOURCE_COMMIT:
        raise _fixture_error("unexpected sourceCommit")
    license_row = provenance.get("license")
    if not isinstance(license_row, dict) or license_row.get("spdx") != "Apache-2.0":
        raise _fixture_error("unexpected license")

    expected_names = tuple(f"{index:06d}.png" for index in range(9))
    files = provenance.get("files")
    if not isinstance(files, dict) or tuple(sorted(files)) != expected_names:
        raise _fixture_error("file set is not exactly 000000.png through 000008.png")

    verified: list[Path] = []
    for name in expected_names:
        expected_digest = files[name]
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise _fixture_error(f"{name} has an invalid expected SHA-256")
        path = directory / name
        try:
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
        except OSError as error:
            raise _fixture_error(f"{name} is unavailable") from error
        if digest != expected_digest:
            raise _fixture_error(f"{name} SHA-256 mismatch")
        verified.append(path)
    return tuple(verified)


__all__ = [
    "MAX_SOURCE_DIMENSION",
    "MAX_SOURCE_PIXELS",
    "PATCH_SIZE",
    "SOURCE_COMMIT",
    "SOURCE_REPOSITORY",
    "TARGET_WIDTH",
    "FixtureProvenanceError",
    "ImageBoundsError",
    "PreprocessedFrame",
    "preprocess_image",
    "verify_courthouse_fixtures",
]
