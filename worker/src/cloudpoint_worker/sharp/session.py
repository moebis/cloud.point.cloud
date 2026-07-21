"""Validated, atomic SHARP single-frame reconstruction session."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from cloudpoint_worker.model._vendor.ml_sharp.utils.gaussians import (
    Gaussians3D,
    save_ply,
)

from . import SOURCE_COMMIT
from .inference import predict as real_predict

EventEmitter = Callable[[dict[str, object]], None]
Predictor = Callable[[np.ndarray, float, str, EventEmitter], Gaussians3D]

_INPUT_PATTERN = re.compile(r"Frames/[0-9]{8}\.jpg")
_OUTPUT_PATTERN = re.compile(r"Outputs/Gaussians/[0-9]{8}\.ply")


class SharpSessionError(RuntimeError):
    """A fail-closed SHARP session validation error."""


@dataclass(frozen=True)
class SharpResult:
    source_frame_index: int
    ply_relative_path: str
    provenance_relative_path: str
    gaussian_count: int
    duration_seconds: float
    device: str
    used_cpu_fallback: bool


def _safe_project_file(
    project_root: Path,
    relative: str,
    pattern: re.Pattern[str],
) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != relative
        or pattern.fullmatch(relative) is None
    ):
        raise SharpSessionError(f"unsafe project-relative path: {relative}")
    root = project_root.resolve(strict=True)
    candidate = project_root / relative
    parent = candidate.parent.resolve(strict=True)
    if parent != root / pure.parent:
        raise SharpSessionError(f"project path traverses a symbolic link: {relative}")
    return candidate


def _regular_file(path: Path, description: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise SharpSessionError(f"{description} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SharpSessionError(f"{description} must be a regular file")


def _focal_length_px(
    width: int,
    height: int,
    focal_length_35mm: float = 30.0,
) -> float:
    diagonal = math.sqrt(width**2 + height**2)
    return focal_length_35mm * diagonal / math.sqrt(36**2 + 24**2)


def _validate_gaussians(gaussians: Gaussians3D) -> int:
    tensors = [
        gaussians.mean_vectors,
        gaussians.singular_values,
        gaussians.quaternions,
        gaussians.colors,
        gaussians.opacities,
    ]
    expected_tail = [(3,), (3,), (4,), (3,), ()]
    if any(tensor.ndim < 2 or tensor.shape[0] != 1 for tensor in tensors):
        raise SharpSessionError("invalid Gaussian tensor shape")
    count = int(gaussians.mean_vectors.shape[1])
    if count <= 0:
        raise SharpSessionError("invalid Gaussian count")
    for tensor, tail in zip(tensors, expected_tail, strict=True):
        if int(tensor.shape[1]) != count or tuple(tensor.shape[2:]) != tail:
            raise SharpSessionError("invalid Gaussian tensor shape")
        if not bool(torch.isfinite(tensor).all().item()):
            raise SharpSessionError("invalid Gaussian non-finite value")
    if not bool((gaussians.mean_vectors[..., 2] > 0).all().item()):
        raise SharpSessionError("invalid Gaussian depth")
    if not bool((gaussians.singular_values > 0).all().item()):
        raise SharpSessionError("invalid Gaussian scale")
    if not bool(
        ((gaussians.opacities > 0) & (gaussians.opacities < 1)).all().item()
    ):
        raise SharpSessionError("invalid Gaussian opacity")
    return count


def _validate_ply(path: Path, expected_count: int) -> None:
    _regular_file(path, "staged Gaussian PLY")
    ply = PlyData.read(path)
    try:
        vertex = ply["vertex"]
    except KeyError as error:
        raise SharpSessionError("invalid Gaussian PLY vertex element") from error
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    names = set(vertex.data.dtype.names or ())
    if len(vertex.data) != expected_count or not required.issubset(names):
        raise SharpSessionError("invalid Gaussian PLY schema")
    values = np.column_stack([np.asarray(vertex[name]) for name in sorted(required)])
    if not np.isfinite(values).all() or not (np.asarray(vertex["z"]) > 0).all():
        raise SharpSessionError("invalid Gaussian PLY values")


def _sync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _default_predictor(checkpoint: Path) -> Predictor:
    def invoke(
        image: np.ndarray,
        focal_px: float,
        device: str,
        emit: EventEmitter,
    ) -> Gaussians3D:
        return real_predict(
            image,
            focal_px,
            device,
            emit,
            checkpoint=checkpoint,
        )

    return invoke


def _recoverable_mps_failure(error: RuntimeError) -> bool:
    message = str(error).lower()
    recoverable_tokens = ("mps", "metal", "out of memory", "not implemented")
    return any(token in message for token in recoverable_tokens)


def reconstruct(
    *,
    project_root: Path,
    checkpoint: Path,
    input_relative_path: str,
    output_relative_path: str,
    prefer_mps: bool,
    checkpoint_sha256: str,
    source_commit: str = SOURCE_COMMIT,
    predictor: Predictor | None = None,
    mps_available: Callable[[], bool] = torch.backends.mps.is_available,
    emit: EventEmitter,
) -> SharpResult:
    """Run one SHARP inference and atomically publish validated PLY/provenance."""
    started = time.monotonic()
    if not project_root.is_absolute() or not checkpoint.is_absolute():
        raise SharpSessionError("project and checkpoint paths must be absolute")
    _regular_file(checkpoint, "SHARP checkpoint")
    input_path = _safe_project_file(project_root, input_relative_path, _INPUT_PATTERN)
    output_path = _safe_project_file(
        project_root,
        output_relative_path,
        _OUTPUT_PATTERN,
    )
    _regular_file(input_path, "SHARP input frame")

    emit({"type": "progress", "stage": "loading", "fraction": 0.0})
    with Image.open(input_path) as source:
        image = np.asarray(source.convert("RGB"))
    height, width = image.shape[:2]
    focal_px = _focal_length_px(width, height)
    selected_device = "mps" if prefer_mps and mps_available() else "cpu"
    used_cpu_fallback = False
    run_predictor = predictor or _default_predictor(checkpoint)

    emit({"type": "progress", "stage": "inference", "fraction": 0.0})
    try:
        gaussians = run_predictor(image, focal_px, selected_device, emit)
    except RuntimeError as error:
        if selected_device != "mps" or not _recoverable_mps_failure(error):
            raise
        emit(
            {
                "type": "warning",
                "code": "MPS_FALLBACK",
                "message": "SHARP could not finish on MPS and will retry on CPU.",
                "recoverable": True,
            }
        )
        torch.mps.empty_cache()
        selected_device = "cpu"
        used_cpu_fallback = True
        gaussians = run_predictor(image, focal_px, selected_device, emit)

    emit({"type": "progress", "stage": "validating", "fraction": 0.0})
    gaussian_count = _validate_gaussians(gaussians)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged_ply = output_path.with_name(f".{output_path.name}.partial.{token}")
    provenance_path = output_path.with_suffix(".json")
    staged_provenance = provenance_path.with_name(
        f".{provenance_path.name}.partial.{token}"
    )
    try:
        save_ply(gaussians, focal_px, (height, width), staged_ply)
        _validate_ply(staged_ply, gaussian_count)
        duration = time.monotonic() - started
        provenance = {
            "schemaVersion": 1,
            "modelIdentifier": "apple/ml-sharp",
            "sourceCommit": source_commit,
            "checkpointSHA256": checkpoint_sha256,
            "sourceFrameIndex": int(Path(input_relative_path).stem),
            "inputRelativePath": input_relative_path,
            "plyRelativePath": output_relative_path,
            "gaussianCount": gaussian_count,
            "device": selected_device,
            "usedCPUFallback": used_cpu_fallback,
            "focalLengthPixels": focal_px,
            "imageWidth": width,
            "imageHeight": height,
            "durationSeconds": duration,
            "generatedAt": datetime.now(UTC).isoformat(),
        }
        staged_provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _sync(staged_ply)
        _sync(staged_provenance)
        emit({"type": "progress", "stage": "committing", "fraction": 0.0})
        os.replace(staged_ply, output_path)
        os.replace(staged_provenance, provenance_path)
        directory_descriptor = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        staged_ply.unlink(missing_ok=True)
        staged_provenance.unlink(missing_ok=True)

    return SharpResult(
        source_frame_index=int(Path(input_relative_path).stem),
        ply_relative_path=output_relative_path,
        provenance_relative_path=str(PurePosixPath(output_relative_path).with_suffix(".json")),
        gaussian_count=gaussian_count,
        duration_seconds=time.monotonic() - started,
        device=selected_device,
        used_cpu_fallback=used_cpu_fallback,
    )


def result_payload(result: SharpResult) -> dict[str, object]:
    """Convert an internal result into the canonical JSON-lines payload."""
    payload = asdict(result)
    return {
        "sourceFrameIndex": payload["source_frame_index"],
        "plyRelativePath": payload["ply_relative_path"],
        "provenanceRelativePath": payload["provenance_relative_path"],
        "gaussianCount": payload["gaussian_count"],
        "durationSeconds": payload["duration_seconds"],
        "device": payload["device"],
        "usedCPUFallback": payload["used_cpu_fallback"],
    }
