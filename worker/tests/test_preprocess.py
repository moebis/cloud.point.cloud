from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image
from safetensors.numpy import load_file, save_file

import cloudpoint_worker.preprocess as preprocess_module
from cloudpoint_worker.preprocess import (
    FixtureProvenanceError,
    preprocess_image,
    verify_courthouse_fixtures,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
EXPORTER_PATH = Path(__file__).parents[1] / "tools" / "export_reference.py"


def _write_png_header(path: Path, size: tuple[int, int]) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    width, height = size
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


def _load_reference_exporter() -> ModuleType:
    assert EXPORTER_PATH.is_file(), "reference exporter is not implemented"
    spec = importlib.util.spec_from_file_location(
        "cloudpoint_reference_exporter", EXPORTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _create_clean_upstream_checkout(path: Path) -> tuple[Path, str]:
    path.mkdir()
    required = (
        "LICENSE.txt",
        "lingbot_map/models/gct_stream.py",
        "lingbot_map/utils/load_fn.py",
        "lingbot_map/utils/pose_enc.py",
    )
    for name in required:
        destination = path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"fixture for {name}\n", encoding="utf-8")
    (path / ".gitignore").write_text("*.so\n__pycache__/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", path], check=True)
    subprocess.run(
        ["git", "-C", path, "config", "user.name", "CloudPoint Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", path, "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", path, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", path, "commit", "-q", "-m", "pinned source"],
        check=True,
    )
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return path, result.stdout.strip()


def test_preprocess_applies_orientation_rgb_518_crop_grid_and_inverse() -> None:
    result = preprocess_image(FIXTURE_ROOT / "preprocess" / "orientation-6-rgba.png")

    assert result.source_size == (20, 40)
    assert result.model_size == (518, 518)
    assert result.rgb.shape == (518, 518, 3)
    assert result.normalized.shape == result.rgb.shape
    assert result.rgb.dtype == result.normalized.dtype == np.float32
    assert result.model_to_source.dtype == np.float64
    assert result.model_size[0] % 14 == result.model_size[1] % 14 == 0

    model_corner = np.array(
        [result.model_size[0] - 1.0, result.model_size[1] - 1.0, 1.0]
    )
    source_corner = result.model_to_source @ model_corner
    round_trip = np.linalg.inv(result.model_to_source) @ source_corner
    np.testing.assert_allclose(round_trip, model_corner, atol=1e-9)


def test_rgba_is_composited_over_white_before_bicubic_resize() -> None:
    result = preprocess_image(FIXTURE_ROOT / "preprocess" / "orientation-6-rgba.png")

    np.testing.assert_allclose(result.rgb[150, 259], np.ones(3), atol=0.02)
    np.testing.assert_allclose(
        result.rgb[259, 259], np.array([0.5, 1.0, 0.5]), atol=0.05
    )


def test_crop_mode_fixes_width_and_uses_python_rounding(tmp_path: Path) -> None:
    source = tmp_path / "bankers-rounding.png"
    Image.new("RGB", (74, 41), (12, 34, 56)).save(source)

    result = preprocess_image(source)

    # 41 * 518 / 74 / 14 == 20.5. The pinned source uses Python round(),
    # therefore the even patch count is 20 rather than 21.
    assert result.source_size == (74, 41)
    assert result.model_size == (518, 280)

    scale_x = 74 / 518
    scale_y = 41 / 280
    expected = np.array(
        [
            [scale_x, 0.0, 0.5 * scale_x - 0.5],
            [0.0, scale_y, 0.5 * scale_y - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result.model_to_source, expected, atol=1e-12)


def test_normalization_matches_pinned_imagenet_constants(tmp_path: Path) -> None:
    source = tmp_path / "solid.png"
    Image.new("RGB", (14, 14), (255, 0, 128)).save(source)

    result = preprocess_image(source)

    expected_rgb = np.array([1.0, 0.0, 128 / 255], dtype=np.float32)
    expected = (expected_rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    np.testing.assert_allclose(result.rgb[200, 200], expected_rgb, rtol=0, atol=0)
    np.testing.assert_allclose(result.normalized[200, 200], expected, rtol=0, atol=0)


def test_preprocess_accepts_exact_source_bounds_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "at-bounds.png"
    _write_png_header(source, (8192, 4096))

    class DecodeReached(RuntimeError):
        pass

    def stop_before_decode(_image: Image.Image) -> Image.Image:
        raise DecodeReached

    monkeypatch.setattr(
        preprocess_module.ImageOps, "exif_transpose", stop_before_decode
    )

    assert preprocess_module.MAX_SOURCE_DIMENSION == 8192
    assert preprocess_module.MAX_SOURCE_PIXELS == 33_554_432
    with pytest.raises(DecodeReached):
        preprocess_image(source)


@pytest.mark.parametrize("size", [(8193, 1), (8192, 4097)])
def test_preprocess_rejects_source_over_dimension_or_pixel_bounds_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    source = tmp_path / f"over-{size[0]}x{size[1]}.png"
    _write_png_header(source, size)

    def decode_must_not_run(_image: Image.Image) -> Image.Image:
        raise AssertionError("pixel decode started before source bounds validation")

    monkeypatch.setattr(
        preprocess_module.ImageOps, "exif_transpose", decode_must_not_run
    )

    with pytest.raises(
        preprocess_module.ImageBoundsError, match="IMAGE_BOUNDS_EXCEEDED"
    ) as captured:
        preprocess_image(source)
    assert captured.value.code == "IMAGE_BOUNDS_EXCEEDED"


@pytest.mark.parametrize("size", [(101, 1), (201, 1)])
def test_preprocess_normalizes_pillow_decompression_bomb_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
) -> None:
    source = tmp_path / f"bomb-{size[0]}x{size[1]}.png"
    _write_png_header(source, size)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(
        preprocess_module.ImageBoundsError, match="IMAGE_BOUNDS_EXCEEDED"
    ) as captured:
        preprocess_image(source)
    assert captured.value.code == "IMAGE_BOUNDS_EXCEEDED"


def test_courthouse_fixture_provenance_verifies_every_file(tmp_path: Path) -> None:
    courthouse = FIXTURE_ROOT / "courthouse"
    paths = verify_courthouse_fixtures(courthouse)
    assert tuple(path.name for path in paths) == tuple(
        f"{index:06d}.png" for index in range(9)
    )

    tampered = tmp_path / "courthouse"
    shutil.copytree(courthouse, tampered)
    victim = tampered / "000004.png"
    victim.write_bytes(victim.read_bytes() + b"tampered")
    with pytest.raises(FixtureProvenanceError, match="000004.png SHA-256 mismatch"):
        verify_courthouse_fixtures(tampered)


def test_courthouse_fixture_matches_pinned_reference_pixels() -> None:
    reference_path = FIXTURE_ROOT / "parity" / "generated" / "preprocess.safetensors"
    if not reference_path.exists():
        pytest.skip("reference fixture not generated")

    actual = preprocess_image(FIXTURE_ROOT / "courthouse" / "000000.png")
    expected = load_file(reference_path)["frame.0.normalized"]
    np.testing.assert_allclose(actual.normalized, expected, rtol=0, atol=1 / 255)


def test_reference_exporter_rejects_checkout_at_another_commit(
    tmp_path: Path,
) -> None:
    exporter = _load_reference_exporter()
    checkout = tmp_path / "lingbot-map"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", checkout], check=True)
    subprocess.run(
        ["git", "-C", checkout, "config", "user.name", "CloudPoint Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", checkout, "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (checkout / "README.md").write_text("wrong source\n", encoding="utf-8")
    subprocess.run(["git", "-C", checkout, "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", checkout, "commit", "-q", "-m", "wrong source"],
        check=True,
    )

    with pytest.raises(exporter.ReferenceExportError, match="source commit"):
        exporter.verify_upstream_checkout(checkout)


def test_reference_exporter_rejects_modified_tracked_upstream_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_reference_exporter()
    checkout, commit = _create_clean_upstream_checkout(tmp_path / "lingbot-map")
    monkeypatch.setattr(exporter, "SOURCE_COMMIT", commit)
    source = checkout / "lingbot_map" / "utils" / "load_fn.py"
    source.write_text("modified executable source\n", encoding="utf-8")

    with pytest.raises(exporter.ReferenceExportError, match="dirty content"):
        exporter.verify_upstream_checkout(checkout)


def test_reference_exporter_rejects_untracked_upstream_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_reference_exporter()
    checkout, commit = _create_clean_upstream_checkout(tmp_path / "lingbot-map")
    monkeypatch.setattr(exporter, "SOURCE_COMMIT", commit)
    injected = checkout / "lingbot_map" / "models" / "injected.py"
    injected.write_text("raise RuntimeError('must never import')\n", encoding="utf-8")

    with pytest.raises(exporter.ReferenceExportError, match="dirty content"):
        exporter.verify_upstream_checkout(checkout)


def test_reference_exporter_rejects_ignored_extension_module_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_reference_exporter()
    checkout, commit = _create_clean_upstream_checkout(tmp_path / "lingbot-map")
    monkeypatch.setattr(exporter, "SOURCE_COMMIT", commit)
    shadow = checkout / "lingbot_map" / "models" / "gct_stream.so"
    shadow.write_bytes(b"ignored executable shadow")

    status = subprocess.run(
        ["git", "-C", checkout, "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    with pytest.raises(exporter.ReferenceExportError, match="dirty content"):
        exporter.verify_upstream_checkout(checkout)


def test_reference_exporter_rejects_assume_unchanged_tracked_source_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_reference_exporter()
    checkout, commit = _create_clean_upstream_checkout(tmp_path / "lingbot-map")
    monkeypatch.setattr(exporter, "SOURCE_COMMIT", commit)
    relative = "lingbot_map/utils/load_fn.py"
    source = checkout / relative
    subprocess.run(
        ["git", "-C", checkout, "update-index", "--assume-unchanged", relative],
        check=True,
    )
    try:
        source.write_text("modified but hidden from status\n", encoding="utf-8")
        status = subprocess.run(
            [
                "git",
                "-C",
                checkout,
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert status.stdout == ""
        index_before = (checkout / ".git" / "index").read_bytes()

        with pytest.raises(exporter.ReferenceExportError, match="content mismatch"):
            exporter.verify_upstream_checkout(checkout)

        assert (checkout / ".git" / "index").read_bytes() == index_before
    finally:
        subprocess.run(
            [
                "git",
                "-C",
                checkout,
                "update-index",
                "--no-assume-unchanged",
                relative,
            ],
            check=True,
        )
    flags = subprocess.run(
        ["git", "-C", checkout, "ls-files", "-v", "--", relative],
        check=True,
        capture_output=True,
        text=True,
    )
    assert flags.stdout.startswith("H ")


def test_reference_exporter_pins_cpu_parity_model_arguments() -> None:
    exporter = _load_reference_exporter()

    assert exporter.MODEL_ARGUMENTS == {
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


def test_reference_exporter_uses_hardened_checkpoint_loader_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_reference_exporter()
    from cloudpoint_worker.model_prep import convert

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test does not load this file")
    expected_state = {"weight": object()}
    calls: list[tuple[Path, object]] = []

    def hardened_loader(
        path: Path, *, progress: object = None
    ) -> tuple[dict[str, object], bool]:
        calls.append((path, progress))
        return expected_state, False

    monkeypatch.setattr(convert, "load_checkpoint", hardened_loader)

    state, fallback_used = exporter.load_reference_checkpoint(checkpoint)

    assert state is expected_state
    assert fallback_used is False
    assert calls == [(checkpoint, None)]


def test_reference_manifest_hashes_and_inventories_every_tensor_file(
    tmp_path: Path,
) -> None:
    exporter = _load_reference_exporter()
    output = tmp_path / "generated"
    output.mkdir()
    for index, name in enumerate(exporter.TENSOR_FILENAMES):
        save_file(
            {
                f"tensor.{index}": np.full(
                    (index + 1, 2), index + 0.25, dtype=np.float32
                )
            },
            output / name,
        )

    manifest_path = exporter.write_reference_manifest(
        output,
        frame_paths=verify_courthouse_fixtures(FIXTURE_ROOT / "courthouse"),
        restricted_load_attempted=True,
        unsafe_fallback_used=False,
        missing_keys=("missing.weight",),
        unexpected_keys=(),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schemaVersion"] == 1
    assert manifest["source"]["commit"] == exporter.SOURCE_COMMIT
    assert manifest["checkpoint"]["sha256"] == exporter.MODEL_SHA256
    assert manifest["load"] == {
        "restrictedLoadAttempted": True,
        "unsafeFallbackUsed": False,
        "missingKeys": ["missing.weight"],
        "unexpectedKeys": [],
    }
    assert list(manifest["files"]) == sorted(exporter.TENSOR_FILENAMES)
    for index, name in enumerate(exporter.TENSOR_FILENAMES):
        path = output / name
        row = manifest["files"][name]
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert row["size"] == path.stat().st_size
        assert row["tensors"] == [
            {
                "dtype": "F32",
                "name": f"tensor.{index}",
                "shape": [index + 1, 2],
            }
        ]
