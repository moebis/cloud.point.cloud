#!/usr/bin/env python3
"""Export deterministic CPU PyTorch parity fixtures for the MLX port.

This development-only tool consumes the exact Lingbot Map source revision and
checkpoint pinned by CloudPoint. It performs no network requests. The generated
SafeTensors files are intentionally ignored by Git; their manifest records the
complete source, checkpoint, frame, tensor, and file provenance needed to
reproduce them.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

from cloudpoint_worker.model_prep.provenance import (
    MODEL_FILENAME,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE,
)
from cloudpoint_worker.preprocess import (
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    preprocess_image,
    verify_courthouse_fixtures,
)

SELECTED_LAYERS = (4, 11, 17, 23)
TENSOR_FILENAMES = (
    "preprocess.safetensors",
    "leaf-fp32.safetensors",
    "e2e-fp32.safetensors",
)
MANIFEST_FILENAME = "reference-manifest.json"

# Keep this dictionary limited to explicit GCTStream constructor arguments. Its
# serialized form is part of the reference provenance contract.
MODEL_ARGUMENTS: dict[str, object] = {
    "img_size": 518,
    "patch_size": 14,
    "enable_3d_rope": True,
    "max_frame_num": 1024,
    "kv_cache_sliding_window": 64,
    "kv_cache_scale_frames": 8,
    "kv_cache_cross_frame_special": True,
    "kv_cache_include_scale_frames": True,
    "use_sdpa": True,
    "enable_point": False,
    "enable_depth": True,
    "camera_num_iterations": 1,
}


class ReferenceExportError(RuntimeError):
    """The requested reference export is not pinned or reproducible."""


@dataclass(frozen=True)
class ExportResult:
    source_commit: str
    frame_count: int
    restricted_load_attempted: bool
    unsafe_fallback_used: bool
    output_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _TreeEntry:
    mode: bytes
    kind: bytes
    object_id: bytes
    path: bytes


@dataclass(frozen=True)
class _WorktreeEntry:
    mode: bytes
    kind: bytes
    absolute_path: bytes


def _sha256(path: Path) -> str:
    try:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as error:
        raise ReferenceExportError(f"could not hash {path}") from error


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ReferenceExportError("git is unavailable") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or "git command failed"
        raise ReferenceExportError(detail)
    return result.stdout.strip()


def _git_bytes(checkout: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
        )
    except OSError as error:
        raise ReferenceExportError("git is unavailable") from error
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise ReferenceExportError(detail or "git command failed")
    return result.stdout


def _pinned_tree(checkout: Path) -> dict[bytes, _TreeEntry]:
    raw = _git_bytes(
        checkout,
        "ls-tree",
        "-r",
        "-t",
        "-z",
        "--full-tree",
        SOURCE_COMMIT,
    )
    entries: dict[bytes, _TreeEntry] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, path = record.split(b"\t", 1)
            mode, kind, object_id = header.split(b" ", 2)
        except ValueError as error:
            raise ReferenceExportError("pinned source tree is malformed") from error
        if (
            not path
            or path.startswith(b"/")
            or any(component in {b"", b".", b".."} for component in path.split(b"/"))
            or path in entries
        ):
            raise ReferenceExportError("pinned source tree has an unsafe path")
        if (mode, kind) not in {
            (b"040000", b"tree"),
            (b"100644", b"blob"),
            (b"100755", b"blob"),
            (b"120000", b"blob"),
        }:
            raise ReferenceExportError("pinned source tree has an unsupported mode")
        entries[path] = _TreeEntry(mode, kind, object_id, path)
    if not entries:
        raise ReferenceExportError("pinned source tree is empty")
    return entries


def _worktree_entries(checkout: Path) -> dict[bytes, _WorktreeEntry]:
    root = os.fsencode(checkout)
    pending: list[tuple[bytes, bytes]] = [(root, b"")]
    entries: dict[bytes, _WorktreeEntry] = {}
    try:
        while pending:
            directory, relative_directory = pending.pop()
            with os.scandir(directory) as children:
                for child in children:
                    name = child.name
                    if not relative_directory and name == b".git":
                        continue
                    relative = (
                        name
                        if not relative_directory
                        else relative_directory + b"/" + name
                    )
                    info = child.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        entry = _WorktreeEntry(b"040000", b"tree", child.path)
                        pending.append((child.path, relative))
                    elif stat.S_ISREG(info.st_mode):
                        mode = b"100755" if info.st_mode & 0o111 else b"100644"
                        entry = _WorktreeEntry(mode, b"blob", child.path)
                    elif stat.S_ISLNK(info.st_mode):
                        entry = _WorktreeEntry(b"120000", b"blob", child.path)
                    else:
                        raise ReferenceExportError(
                            "upstream checkout has dirty content: special file at "
                            f"{os.fsdecode(relative)!r}"
                        )
                    entries[relative] = entry
    except OSError as error:
        raise ReferenceExportError(
            "upstream checkout could not be read safely"
        ) from error
    return entries


def _read_worktree_blob(entry: _WorktreeEntry) -> bytes:
    try:
        if entry.mode == b"120000":
            target = os.readlink(entry.absolute_path)
            return target if isinstance(target, bytes) else os.fsencode(target)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(entry.absolute_path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ReferenceExportError(
                    "upstream checkout has dirty content: file type changed"
                )
            return source.read()
    except ReferenceExportError:
        raise
    except OSError as error:
        raise ReferenceExportError(
            "upstream checkout blob could not be read"
        ) from error


def _read_exact(source: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = source.read(remaining)
        if not chunk:
            raise ReferenceExportError("git object stream ended unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _compare_blob_objects(
    checkout: Path,
    expected: Mapping[bytes, _TreeEntry],
    actual: Mapping[bytes, _WorktreeEntry],
) -> None:
    try:
        process = subprocess.Popen(
            ["git", "-C", str(checkout), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise ReferenceExportError("git is unavailable") from error
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise ReferenceExportError("git object stream could not be opened")

    try:
        for path in sorted(expected):
            tree_entry = expected[path]
            if tree_entry.kind != b"blob":
                continue
            process.stdin.write(tree_entry.object_id + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n")
            fields = header.split(b" ")
            if len(fields) != 3:
                raise ReferenceExportError(
                    "git object stream returned invalid metadata"
                )
            object_id, kind, raw_size = fields
            try:
                size = int(raw_size)
            except ValueError as error:
                raise ReferenceExportError(
                    "git object stream returned an invalid size"
                ) from error
            if object_id != tree_entry.object_id or kind != b"blob":
                raise ReferenceExportError(
                    "git object stream returned the wrong object"
                )
            pinned_content = _read_exact(process.stdout, size)
            if process.stdout.read(1) != b"\n":
                raise ReferenceExportError("git object stream framing is invalid")
            worktree_content = _read_worktree_blob(actual[path])
            if worktree_content != pinned_content:
                raise ReferenceExportError(
                    "upstream checkout has dirty content: content mismatch at "
                    f"{os.fsdecode(path)!r}"
                )
        process.stdin.close()
        if process.wait() != 0:
            detail = process.stderr.read().decode(errors="replace").strip()
            raise ReferenceExportError(detail or "git object stream failed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def verify_upstream_checkout(checkout: Path) -> str:
    """Verify every checkout entry against the exact pinned Git object tree."""

    resolved = checkout.expanduser().resolve()
    if not resolved.is_dir():
        raise ReferenceExportError("upstream checkout is not a directory")
    if _git_output(resolved, "rev-parse", "--is-inside-work-tree") != "true":
        raise ReferenceExportError("upstream path is not a Git checkout")
    head = _git_output(resolved, "rev-parse", "HEAD")
    if head != SOURCE_COMMIT:
        raise ReferenceExportError(
            f"source commit {head!r} does not match pinned {SOURCE_COMMIT}"
        )
    expected = _pinned_tree(resolved)
    actual = _worktree_entries(resolved)
    expected_paths = set(expected)
    actual_paths = set(actual)
    extra = sorted(actual_paths - expected_paths)
    if extra:
        raise ReferenceExportError(
            f"upstream checkout has dirty content: extra path {os.fsdecode(extra[0])!r}"
        )
    missing = sorted(expected_paths - actual_paths)
    if missing:
        raise ReferenceExportError(
            "upstream checkout has dirty content: missing path "
            f"{os.fsdecode(missing[0])!r}"
        )
    for path in sorted(expected_paths):
        tree_entry = expected[path]
        worktree_entry = actual[path]
        if (tree_entry.mode, tree_entry.kind) != (
            worktree_entry.mode,
            worktree_entry.kind,
        ):
            raise ReferenceExportError(
                "upstream checkout has dirty content: mode mismatch at "
                f"{os.fsdecode(path)!r}"
            )
    required = (
        b"LICENSE.txt",
        b"lingbot_map/models/gct_stream.py",
        b"lingbot_map/utils/load_fn.py",
        b"lingbot_map/utils/pose_enc.py",
    )
    if not all(
        path in expected and expected[path].kind == b"blob" for path in required
    ):
        raise ReferenceExportError("upstream checkout is missing required source files")
    _compare_blob_objects(resolved, expected, actual)
    return head


def load_reference_checkpoint(checkpoint: Path) -> tuple[dict[str, Any], bool]:
    """Load through the hardened public restricted-first checkpoint API."""

    from cloudpoint_worker.model_prep.convert import load_checkpoint

    return load_checkpoint(checkpoint)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    content = (
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write_bytes(path, content)


def _atomic_write_safetensors(path: Path, tensors: Mapping[str, Any]) -> None:
    import torch
    from safetensors.torch import save_file

    ordered: dict[str, Any] = {}
    for name, tensor in sorted(tensors.items()):
        if not torch.is_tensor(tensor):
            raise ReferenceExportError(f"{name} is not a tensor")
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        ordered[name] = value
    if not ordered:
        raise ReferenceExportError(f"refusing to write empty fixture {path.name}")

    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        save_file(ordered, partial)
        with partial.open("rb") as destination:
            os.fsync(destination.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _tensor_inventory(path: Path) -> list[dict[str, object]]:
    try:
        with safe_open(path, framework="np") as source:
            inventory = []
            for name in sorted(source.keys()):
                tensor = source.get_slice(name)
                dtype = tensor.get_dtype()
                if dtype != "F32":
                    raise ReferenceExportError(
                        f"{path.name}:{name} has non-Float32 dtype {dtype}"
                    )
                inventory.append(
                    {
                        "dtype": dtype,
                        "name": name,
                        "shape": list(tensor.get_shape()),
                    }
                )
    except ReferenceExportError:
        raise
    except Exception as error:
        raise ReferenceExportError(f"could not inspect {path.name}") from error
    if not inventory:
        raise ReferenceExportError(f"{path.name} contains no tensors")
    return inventory


def write_reference_manifest(
    output: Path,
    *,
    frame_paths: Sequence[Path],
    restricted_load_attempted: bool,
    unsafe_fallback_used: bool,
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
) -> Path:
    """Inventory and hash all generated tensors, then atomically publish JSON."""

    files: dict[str, dict[str, object]] = {}
    for name in sorted(TENSOR_FILENAMES):
        path = output / name
        if not path.is_file():
            raise ReferenceExportError(f"generated fixture {name} is missing")
        files[name] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
            "tensors": _tensor_inventory(path),
        }

    frames = [
        {
            "filename": path.name,
            "sha256": _sha256(path),
        }
        for path in frame_paths
    ]
    manifest = {
        "schemaVersion": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "license": "Apache-2.0",
            "licenseURL": f"{SOURCE_REPOSITORY}/blob/{SOURCE_COMMIT}/LICENSE.txt",
        },
        "checkpoint": {
            "repository": MODEL_REPO,
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "size": MODEL_SIZE,
            "sha256": MODEL_SHA256,
        },
        "frames": frames,
        "modelArguments": MODEL_ARGUMENTS,
        "selectedAggregatorLayers": list(SELECTED_LAYERS),
        "execution": {
            "device": "cpu",
            "dtype": "float32",
            "torchDeterministicAlgorithms": True,
            "torchThreads": 1,
        },
        "load": {
            "restrictedLoadAttempted": restricted_load_attempted,
            "unsafeFallbackUsed": unsafe_fallback_used,
            "missingKeys": sorted(missing_keys),
            "unexpectedKeys": sorted(unexpected_keys),
        },
        "files": files,
    }
    destination = output / MANIFEST_FILENAME
    _atomic_write_json(destination, manifest)
    return destination


def _flatten_hook_tensors(value: object, prefix: str = "") -> Iterable[tuple[str, Any]]:
    import torch

    if torch.is_tensor(value):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            suffix = str(key) if not prefix else f"{prefix}.{key}"
            yield from _flatten_hook_tensors(value[key], suffix)
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            suffix = str(index) if not prefix else f"{prefix}.{index}"
            yield from _flatten_hook_tensors(item, suffix)


def _cpu_float_copy(tensor: Any) -> Any:
    import torch

    return (
        tensor.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .clone(memory_format=torch.contiguous_format)
    )


class _ReferenceRecorder:
    def __init__(self) -> None:
        self.leaf: dict[str, Any] = {}
        self._call_count: defaultdict[str, int] = defaultdict(int)
        self._selected: dict[int, list[Any]] = {index: [] for index in SELECTED_LAYERS}
        self._handles: list[Any] = []

    def _leaf_hook(self, prefix: str) -> Any:
        def hook(_module: object, _arguments: object, output: object) -> None:
            call_index = self._call_count[prefix]
            self._call_count[prefix] += 1
            tensors = list(_flatten_hook_tensors(output))
            if not tensors:
                raise ReferenceExportError(f"hook {prefix} produced no tensors")
            for suffix, tensor in tensors:
                name = f"{prefix}.call.{call_index:03d}"
                if suffix:
                    name = f"{name}.{suffix}"
                self.leaf[name] = _cpu_float_copy(tensor)

        return hook

    def _aggregator_hook(
        self, _module: object, _arguments: object, output: object
    ) -> None:
        if not isinstance(output, tuple) or len(output) != 2:
            raise ReferenceExportError("aggregator hook produced an unexpected result")
        selected = output[0]
        if not isinstance(selected, list) or len(selected) != len(SELECTED_LAYERS):
            raise ReferenceExportError("aggregator selected-layer count changed")
        for layer, tensor in zip(SELECTED_LAYERS, selected, strict=True):
            self._selected[layer].append(_cpu_float_copy(tensor))

    def register(self, model: object) -> None:
        aggregator = model.aggregator  # type: ignore[attr-defined]
        modules: list[tuple[str, object]] = [("patch_embed", aggregator.patch_embed)]
        modules.extend(
            (f"frame.{index}", aggregator.frame_blocks[index]) for index in (0, 11, 23)
        )
        modules.extend(
            (f"global.{index}", aggregator.global_blocks[index])
            for index in (0, 11, 23)
        )
        depth_head = model.depth_head  # type: ignore[attr-defined]
        modules.extend(
            (f"dpt.resize.{index}", layer)
            for index, layer in enumerate(depth_head.resize_layers)
        )
        camera_head = model.camera_head  # type: ignore[attr-defined]
        modules.extend(
            (f"camera.trunk.{index}", layer)
            for index, layer in enumerate(camera_head.trunk)
        )
        for name, module in modules:
            self._handles.append(module.register_forward_hook(self._leaf_hook(name)))
        self._handles.append(aggregator.register_forward_hook(self._aggregator_hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def selected_tensors(self, frame_count: int) -> dict[str, Any]:
        import torch

        result: dict[str, Any] = {}
        for layer in SELECTED_LAYERS:
            calls = self._selected[layer]
            if not calls:
                raise ReferenceExportError(f"aggregator layer {layer} was not captured")
            combined = torch.cat(calls, dim=1)
            if combined.shape[0] != 1 or combined.shape[1] != frame_count:
                raise ReferenceExportError(
                    f"aggregator layer {layer} has unexpected frame shape"
                )
            result[f"selected.{layer}"] = combined[0].contiguous()
        return result


def _verify_upstream_frames(upstream: Path, frame_paths: Sequence[Path]) -> None:
    upstream_frames = upstream / "example" / "courthouse"
    for local in frame_paths:
        candidate = upstream_frames / local.name
        if not candidate.is_file() or _sha256(candidate) != _sha256(local):
            raise ReferenceExportError(
                f"upstream courthouse frame {local.name} does not match provenance"
            )


def _preprocessing_tensors(
    frame_paths: Sequence[Path], upstream_images: Any
) -> dict[str, Any]:
    import torch

    if tuple(upstream_images.shape[:2]) != (len(frame_paths), 3):
        raise ReferenceExportError(
            "upstream preprocessing returned an unexpected batch"
        )
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    tensors: dict[str, Any] = {}
    for index, path in enumerate(frame_paths):
        frame = preprocess_image(path)
        upstream_rgb = upstream_images[index].to(dtype=torch.float32, device="cpu")
        own_rgb = torch.from_numpy(frame.rgb).permute(2, 0, 1)
        if upstream_rgb.shape != own_rgb.shape or not torch.allclose(
            upstream_rgb, own_rgb, rtol=0, atol=1 / 255
        ):
            raise ReferenceExportError(
                f"local preprocessing diverges from upstream for {path.name}"
            )
        prefix = f"frame.{index}"
        tensors[f"{prefix}.rgb"] = upstream_rgb.permute(1, 2, 0).contiguous()
        tensors[f"{prefix}.normalized"] = (
            ((upstream_rgb - mean) / std).permute(1, 2, 0).contiguous()
        )
        tensors[f"{prefix}.model_to_source"] = torch.from_numpy(
            frame.model_to_source.astype("float32")
        )
        tensors[f"{prefix}.source_size"] = torch.tensor(
            frame.source_size, dtype=torch.float32
        )
        tensors[f"{prefix}.model_size"] = torch.tensor(
            frame.model_size, dtype=torch.float32
        )
    return tensors


def _geometry_tensors(model: object, predictions: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from lingbot_map.utils.geometry import closed_form_inverse_se3
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    pose = predictions["pose_enc"].to(device="cpu", dtype=torch.float32)
    depth = predictions["depth"].to(device="cpu", dtype=torch.float32)
    confidence = predictions["depth_conf"].to(device="cpu", dtype=torch.float32)
    height, width = depth.shape[2:4]
    world_to_camera_3x4, intrinsics = pose_encoding_to_extri_intri(
        pose, image_size_hw=(height, width), build_intrinsics=True
    )
    batch, frames = pose.shape[:2]
    world_to_camera = torch.zeros(
        (batch, frames, 4, 4), dtype=torch.float32, device="cpu"
    )
    world_to_camera[..., :3, :] = world_to_camera_3x4
    world_to_camera[..., 3, 3] = 1.0
    camera_to_world = closed_form_inverse_se3(
        world_to_camera.reshape(batch * frames, 4, 4)
    ).reshape(batch, frames, 4, 4)
    reprojected = model._unproject_depth_to_world(depth, pose)  # type: ignore[attr-defined]

    return {
        "pose_encoding": pose[0],
        "depth": depth[0, ..., 0],
        "confidence": confidence[0],
        "intrinsics": intrinsics[0].to(dtype=torch.float32),
        "world_to_camera": world_to_camera[0],
        "camera_to_world": camera_to_world[0],
        "reprojected_world_points": reprojected[0].to(dtype=torch.float32),
    }


def export_reference(upstream: Path, checkpoint: Path, output: Path) -> ExportResult:
    """Generate the complete pinned reference bundle with CPU Float32 inference."""

    import torch

    upstream = upstream.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    verify_upstream_checkout(upstream)
    state, unsafe_fallback_used = load_reference_checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise ReferenceExportError("output path is not a directory")

    fixture_directory = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "courthouse"
    )
    frame_paths = verify_courthouse_fixtures(fixture_directory)
    _verify_upstream_frames(upstream, frame_paths)

    upstream_text = str(upstream)
    if upstream_text not in sys.path:
        sys.path.insert(0, upstream_text)
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    images = load_and_preprocess_images(
        [str(path) for path in frame_paths],
        mode="crop",
        image_size=518,
        patch_size=14,
    ).to(device="cpu", dtype=torch.float32)
    preprocessing = _preprocessing_tensors(frame_paths, images)
    _atomic_write_safetensors(output / TENSOR_FILENAMES[0], preprocessing)
    del preprocessing

    model = GCTStream(**MODEL_ARGUMENTS)
    incompatible = model.load_state_dict(state, strict=False)
    missing_keys = tuple(incompatible.missing_keys)
    unexpected_keys = tuple(incompatible.unexpected_keys)
    del state
    gc.collect()
    model = model.to(device="cpu", dtype=torch.float32).eval()

    recorder = _ReferenceRecorder()
    recorder.register(model)
    try:
        with torch.inference_mode():
            predictions = model.inference_streaming(
                images,
                num_scale_frames=8,
                keyframe_interval=1,
                output_device=torch.device("cpu"),
            )
    finally:
        recorder.remove()

    selected = recorder.selected_tensors(len(frame_paths))
    end_to_end = _geometry_tensors(model, predictions)
    end_to_end.update(selected)
    _atomic_write_safetensors(output / TENSOR_FILENAMES[1], recorder.leaf)
    _atomic_write_safetensors(output / TENSOR_FILENAMES[2], end_to_end)

    manifest_path = write_reference_manifest(
        output,
        frame_paths=frame_paths,
        restricted_load_attempted=True,
        unsafe_fallback_used=unsafe_fallback_used,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
    return ExportResult(
        source_commit=SOURCE_COMMIT,
        frame_count=len(frame_paths),
        restricted_load_attempted=True,
        unsafe_fallback_used=unsafe_fallback_used,
        output_paths=(
            *tuple(output / name for name in TENSOR_FILENAMES),
            manifest_path,
        ),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    namespace = _argument_parser().parse_args(arguments)
    try:
        result = export_reference(
            namespace.upstream, namespace.checkpoint, namespace.output
        )
    except Exception as error:
        print(f"reference export failed: {error}", file=sys.stderr)
        return 1

    print(f"source commit {result.source_commit}")
    print(f"{result.frame_count} frames")
    print("restricted load attempted")
    print(f"unsafe fallback used: {str(result.unsafe_fallback_used).lower()}")
    for path in result.output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANIFEST_FILENAME",
    "MODEL_ARGUMENTS",
    "SELECTED_LAYERS",
    "TENSOR_FILENAMES",
    "ReferenceExportError",
    "export_reference",
    "load_reference_checkpoint",
    "main",
    "verify_upstream_checkout",
    "write_reference_manifest",
]
