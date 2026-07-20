"""Strict, exhaustive PyTorch-to-SafeTensors conversion."""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file

from cloudpoint_worker import ENGINE_VERSION
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.model.weight_specs import WeightSpec
from cloudpoint_worker.model_prep.provenance import (
    MODEL_FILENAME,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE,
    SOURCE_COMMIT,
    VerifiedArtifact,
    verify_checkpoint,
    verify_open_file,
)

CONVERTED_FILENAME = "lingbot-map-long-f16.safetensors"
WEIGHTS_MANIFEST_FILENAME = "weights-manifest.json"
MODEL_MANIFEST_FILENAME = "model-manifest.json"


@dataclass(frozen=True)
class ModelManifest:
    schema_version: Literal[1]
    model_identifier: str
    model_revision: str
    source_sha256: str
    converted_sha256: str
    tensor_count: int
    mlx_version: str
    engine_version: str


def _fault(code: str, message: str) -> WorkerFault:
    return WorkerFault(code, message, False)


def _state_dict(payload: object) -> dict[str, torch.Tensor]:
    candidate: object = payload
    if isinstance(payload, Mapping) and "model" in payload:
        candidate = payload["model"]
    if not isinstance(candidate, Mapping):
        raise _fault("MODEL_INVALID_CHECKPOINT", "checkpoint payload is not a mapping")
    state: dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise _fault(
                "MODEL_INVALID_CHECKPOINT",
                "model payload must contain only string-keyed tensors",
            )
        state[key] = value
    return state


def _open_checkpoint(path: Path) -> object:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise _fault("MODEL_INVALID_PATH", "checkpoint could not be opened") from error
    return os.fdopen(fd, "rb", closefd=True)


def _fingerprint_open_file(source: object) -> tuple[int, str]:
    fileno = source.fileno()  # type: ignore[attr-defined]
    info = os.fstat(fileno)
    if not stat.S_ISREG(info.st_mode):
        raise _fault("MODEL_INVALID_PATH", "checkpoint must be a regular file")
    source.seek(0)  # type: ignore[attr-defined]
    digest = hashlib.file_digest(source, "sha256").hexdigest()  # type: ignore[arg-type]
    source.seek(0)  # type: ignore[attr-defined]
    return info.st_size, digest


def _restricted_environment() -> dict[str, str]:
    return {
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }


def _trusted_load_in_child(
    checkpoint: Path,
    expected_size: int,
    expected_sha256: str,
    environment: dict[str, str],
) -> dict[str, torch.Tensor]:
    with tempfile.TemporaryDirectory(prefix="cloudpoint-trusted-load-") as temporary:
        output = Path(temporary) / "trusted.safetensors"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "cloudpoint_worker.model_prep.convert",
                "--trusted-child",
                str(checkpoint),
                str(expected_size),
                expected_sha256,
                str(output),
            ],
            check=False,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.returncode != 0:
            raise _fault(
                "MODEL_CHECKPOINT_LOAD_FAILED",
                "trusted checkpoint loader failed in its restricted child",
            )
        return load_file(output, device="cpu")


def load_checkpoint(
    path: Path,
    verified: VerifiedArtifact | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Load restricted-first, retrying unsafe pickle only in a scrubbed child."""

    with _open_checkpoint(path) as source:  # type: ignore[attr-defined]
        size, digest = _fingerprint_open_file(source)
        if verified is not None and (
            size != verified.size or not hmac.compare_digest(digest, verified.sha256)
        ):
            raise WorkerFault("MODEL_CHECKSUM_MISMATCH", "checkpoint changed", True)
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except Exception:
            verify_open_file(source, size, digest)
            if progress is not None:
                progress("trustedArtifactLoading")
            state = _trusted_load_in_child(
                path.resolve(), size, digest, _restricted_environment()
            )
            return state, True
    return _state_dict(payload), False


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    if contiguous.ndim == 0:
        contiguous = contiguous.reshape(1)
    return contiguous.view(torch.uint8).numpy().tobytes()


def _convert_tensor(tensor: torch.Tensor, spec: WeightSpec) -> torch.Tensor:
    if tuple(tensor.shape) != spec.source_shape:
        raise _fault(
            "MODEL_SOURCE_SHAPE_MISMATCH",
            f"{spec.source_key} has source shape {tuple(tensor.shape)}",
        )
    if tensor.is_floating_point():
        tensor = tensor.detach().cpu().to(torch.float16)
    elif spec.transform != "identity" or tensor.ndim != 0:
        raise _fault(
            "MODEL_UNSUPPORTED_DTYPE",
            f"{spec.source_key} has unsupported dtype {tensor.dtype}",
        )
    else:
        tensor = tensor.detach().cpu()
    if spec.transform == "conv2d":
        converted = tensor.permute(0, 2, 3, 1)
    elif spec.transform == "conv_transpose2d":
        converted = tensor.permute(1, 2, 3, 0)
    elif spec.transform == "identity":
        converted = tensor
    else:  # pragma: no cover - WeightSpec's Literal protects ordinary callers.
        raise _fault("MODEL_UNKNOWN_TRANSFORM", f"unknown transform {spec.transform}")
    converted = converted.contiguous()
    if tuple(converted.shape) != spec.destination_shape:
        raise _fault(
            "MODEL_DESTINATION_SHAPE_MISMATCH",
            f"{spec.destination_key} has destination shape {tuple(converted.shape)}",
        )
    return converted


def convert_state_dict(
    state: Mapping[str, torch.Tensor], specs: Sequence[WeightSpec]
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    """Convert a state dict only when tensor coverage is a strict bijection."""

    source_keys = [spec.source_key for spec in specs]
    destination_keys = [spec.destination_key for spec in specs]
    if len(set(source_keys)) != len(source_keys):
        raise _fault("MODEL_DUPLICATE_SOURCE", "weight specs repeat a source key")
    if len(set(destination_keys)) != len(destination_keys):
        raise _fault(
            "MODEL_DUPLICATE_DESTINATION", "weight specs repeat a destination key"
        )
    missing = sorted(set(source_keys) - set(state))
    if missing:
        raise _fault("MODEL_MISSING_TENSOR", f"missing source tensor {missing[0]}")
    extra = sorted(set(state) - set(source_keys))
    if extra:
        raise _fault("MODEL_EXTRA_TENSOR", f"unexpected source tensor {extra[0]}")

    converted: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for spec in specs:
        source = state[spec.source_key]
        destination = _convert_tensor(source, spec)
        converted[spec.destination_key] = destination
        rows.append(
            {
                "sourceKey": spec.source_key,
                "destinationKey": spec.destination_key,
                "sourceShape": list(spec.source_shape),
                "destinationShape": list(spec.destination_shape),
                "sourceDtype": str(source.dtype).removeprefix("torch."),
                "destinationDtype": str(destination.dtype).removeprefix("torch."),
                "transform": spec.transform,
                "sha256": hashlib.sha256(_tensor_bytes(destination)).hexdigest(),
            }
        )
    return (
        dict(sorted(converted.items())),
        sorted(rows, key=lambda row: str(row["destinationKey"])),
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    partial = path.with_name(f"{path.name}.partial")
    try:
        with partial.open("wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    content = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, content)


def _atomic_write_safetensors(
    path: Path, tensors: Mapping[str, torch.Tensor]
) -> None:
    partial = path.with_name(f"{path.name}.partial")
    try:
        save_file(dict(tensors), partial)
        with partial.open("rb+") as destination:
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _validate_paths(checkpoint: Path, destination: Path) -> None:
    if not checkpoint.is_absolute() or not destination.is_absolute():
        raise _fault(
            "MODEL_INVALID_PATH", "checkpoint and destination must be absolute"
        )
    if not destination.is_dir() or any(destination.iterdir()):
        raise _fault(
            "MODEL_INVALID_PATH", "destination must be an existing empty directory"
        )


def prepare_model(
    checkpoint: Path,
    destination: Path,
    specs: tuple[WeightSpec, ...],
    *,
    progress: Callable[[str], None] | None = None,
) -> ModelManifest:
    """Re-verify, strictly convert, and atomically publish the pinned model."""

    _validate_paths(checkpoint, destination)
    if progress is not None:
        progress("verifying")
    artifact = verify_checkpoint(checkpoint)
    if progress is not None:
        progress("restrictedLoading")
    state, _ = load_checkpoint(checkpoint, artifact, progress)
    if progress is not None:
        progress("converting")
    converted, rows = convert_state_dict(state, specs)
    weights_path = destination / CONVERTED_FILENAME
    _atomic_write_safetensors(weights_path, converted)
    with weights_path.open("rb") as weights_source:
        converted_sha256 = hashlib.file_digest(weights_source, "sha256").hexdigest()
    if progress is not None:
        progress("validating")
    _atomic_write_json(destination / WEIGHTS_MANIFEST_FILENAME, rows)
    mlx_version = importlib.metadata.version("mlx")
    manifest = ModelManifest(
        schema_version=1,
        model_identifier=MODEL_REPO,
        model_revision=MODEL_REVISION,
        source_sha256=artifact.sha256,
        converted_sha256=converted_sha256,
        tensor_count=len(converted),
        mlx_version=mlx_version,
        engine_version=ENGINE_VERSION,
    )
    _atomic_write_json(
        destination / MODEL_MANIFEST_FILENAME,
        {
            "schemaVersion": manifest.schema_version,
            "modelIdentifier": manifest.model_identifier,
            "modelRevision": manifest.model_revision,
            "modelFilename": MODEL_FILENAME,
            "modelSize": MODEL_SIZE,
            "sourceSHA256": MODEL_SHA256,
            "sourceCommit": SOURCE_COMMIT,
            "convertedSha256": manifest.converted_sha256,
            "tensorCount": manifest.tensor_count,
            "mlxVersion": manifest.mlx_version,
            "engineVersion": manifest.engine_version,
            "conversionUTC": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )
    if progress is not None:
        progress("ready")
    return manifest


def _trusted_child(arguments: list[str]) -> int:
    if len(arguments) != 4:
        return 2
    checkpoint = Path(arguments[0])
    expected_size = int(arguments[1])
    expected_sha256 = arguments[2]
    output = Path(arguments[3])
    verified = VerifiedArtifact(checkpoint, expected_size, expected_sha256)
    with _open_checkpoint(checkpoint) as source:  # type: ignore[attr-defined]
        verify_open_file(source, verified.size, verified.sha256)
        payload = torch.load(source, map_location="cpu", weights_only=False)
    state = _state_dict(payload)
    save_file(dict(sorted(state.items())), output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess.
    if sys.argv[1:2] != ["--trusted-child"]:
        raise SystemExit(2)
    raise SystemExit(_trusted_child(sys.argv[2:]))


__all__ = [
    "CONVERTED_FILENAME",
    "MODEL_MANIFEST_FILENAME",
    "WEIGHTS_MANIFEST_FILENAME",
    "ModelManifest",
    "convert_state_dict",
    "load_checkpoint",
    "prepare_model",
]
