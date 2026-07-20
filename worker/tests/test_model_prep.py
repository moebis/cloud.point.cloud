from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.model.config import ModelConfig
from cloudpoint_worker.model.weight_specs import WeightSpec, build_weight_specs
from cloudpoint_worker.model_prep.cli import model_prepare_parser
from cloudpoint_worker.model_prep.convert import (
    _atomic_write_bytes,
    _restricted_environment,
    convert_state_dict,
    load_checkpoint,
    prepare_model,
)
from cloudpoint_worker.model_prep.provenance import VerifiedArtifact, verify_artifact


def test_checkpoint_must_match_size_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "lingbot-map-long.pt"
    path.write_bytes(b"wrong")
    with pytest.raises(WorkerFault, match="MODEL_CHECKSUM_MISMATCH"):
        verify_artifact(path, expected_size=5, expected_sha256="00" * 32)


def test_checkpoint_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.pt"
    target.write_bytes(b"model")
    link = tmp_path / "lingbot-map-long.pt"
    link.symlink_to(target)
    with pytest.raises(WorkerFault, match="MODEL_CHECKSUM_MISMATCH"):
        verify_artifact(
            link,
            expected_size=5,
            expected_sha256=hashlib.sha256(b"model").hexdigest(),
        )


def test_layouts_are_explicit_and_coverage_is_bijective() -> None:
    state = {
        "conv.weight": torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(
            2, 3, 2, 2
        ),
        "up.weight": torch.arange(3 * 2 * 2 * 2, dtype=torch.float32).reshape(
            3, 2, 2, 2
        ),
    }
    specs = [
        WeightSpec(
            "conv.weight",
            "conv.weight",
            (2, 3, 2, 2),
            (2, 2, 2, 3),
            "conv2d",
        ),
        WeightSpec(
            "up.weight",
            "up.weight",
            (3, 2, 2, 2),
            (2, 2, 2, 3),
            "conv_transpose2d",
        ),
    ]
    converted, rows = convert_state_dict(state, specs)
    assert converted["conv.weight"].shape == (2, 2, 2, 3)
    assert converted["up.weight"].shape == (2, 2, 2, 3)
    assert [row["destinationKey"] for row in rows] == ["conv.weight", "up.weight"]
    assert all(row["destinationDtype"] == "float16" for row in rows)
    with pytest.raises(WorkerFault, match="MODEL_EXTRA_TENSOR"):
        convert_state_dict({**state, "surprise": torch.zeros(1)}, specs)


def test_converter_rejects_missing_duplicate_and_shape_mismatches() -> None:
    tensor = torch.ones((2, 3), dtype=torch.float32)
    spec = WeightSpec("weight", "weight", (2, 3), (2, 3), "identity")
    with pytest.raises(WorkerFault, match="MODEL_MISSING_TENSOR"):
        convert_state_dict({}, [spec])
    with pytest.raises(WorkerFault, match="MODEL_DUPLICATE_DESTINATION"):
        convert_state_dict(
            {"weight": tensor, "other": tensor},
            [spec, WeightSpec("other", "weight", (2, 3), (2, 3), "identity")],
        )
    with pytest.raises(WorkerFault, match="MODEL_SOURCE_SHAPE_MISMATCH"):
        convert_state_dict(
            {"weight": tensor},
            [WeightSpec("weight", "weight", (3, 2), (3, 2), "identity")],
        )
    with pytest.raises(WorkerFault, match="MODEL_DESTINATION_SHAPE_MISMATCH"):
        convert_state_dict(
            {"weight": tensor},
            [WeightSpec("weight", "weight", (2, 3), (3, 2), "identity")],
        )


def test_converter_rejects_nonfloating_tensors_except_identity_scalars() -> None:
    vector = torch.ones(2, dtype=torch.int64)
    spec = WeightSpec("count", "count", (2,), (2,), "identity")
    with pytest.raises(WorkerFault, match="MODEL_UNSUPPORTED_DTYPE"):
        convert_state_dict({"count": vector}, [spec])

    scalar_spec = WeightSpec("count", "count", (), (), "identity")
    converted, rows = convert_state_dict({"count": torch.tensor(3)}, [scalar_spec])
    assert converted["count"].item() == 3
    assert rows[0]["destinationDtype"] == "int64"


def test_load_checkpoint_rejects_unpinned_artifact_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "arbitrary.pt"
    torch.save({"weight": torch.ones(1)}, path)
    calls: list[str] = []

    def unexpected_load(*args: object, **kwargs: object) -> object:
        calls.append("restricted")
        raise AssertionError("restricted loader must not run")

    def unexpected_child(*args: object, **kwargs: object) -> object:
        calls.append("unsafe")
        raise AssertionError("unsafe loader must not run")

    monkeypatch.setattr(torch, "load", unexpected_load)
    monkeypatch.setattr(
        "cloudpoint_worker.model_prep.convert._trusted_load_in_child",
        unexpected_child,
    )

    with pytest.raises(WorkerFault, match="MODEL_CHECKSUM_MISMATCH"):
        load_checkpoint(path)
    assert calls == []


def test_trusted_child_reverifies_pinned_provenance_in_subprocess(
    tmp_path: Path,
) -> None:
    path = tmp_path / "arbitrary.pt"
    torch.save({"weight": torch.ones(1)}, path)
    output = tmp_path / "unsafe.safetensors"

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "cloudpoint_worker.model_prep.convert",
            "--trusted-child",
            str(path),
            str(output),
        ],
        check=False,
        capture_output=True,
        env=_restricted_environment(),
    )

    assert process.returncode != 0
    assert b"MODEL_CHECKSUM_MISMATCH" in process.stderr
    assert not output.exists()


def test_load_checkpoint_uses_restricted_mode_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"weight": torch.ones(1)}, path)
    checkpoint_bytes = path.read_bytes()
    monkeypatch.setattr(
        "cloudpoint_worker.model_prep.convert.MODEL_SIZE", len(checkpoint_bytes)
    )
    monkeypatch.setattr(
        "cloudpoint_worker.model_prep.convert.MODEL_SHA256",
        hashlib.sha256(checkpoint_bytes).hexdigest(),
    )
    calls: list[bool] = []
    original_load = torch.load

    def recording_load(*args: object, **kwargs: object) -> object:
        calls.append(bool(kwargs["weights_only"]))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    state, trusted = load_checkpoint(path)
    assert calls == [True]
    assert trusted is False
    assert set(state) == {"weight"}


def test_trusted_fallback_scrubs_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint")
    monkeypatch.setattr("cloudpoint_worker.model_prep.convert.MODEL_SIZE", 10)
    monkeypatch.setattr(
        "cloudpoint_worker.model_prep.convert.MODEL_SHA256",
        hashlib.sha256(b"checkpoint").hexdigest(),
    )
    captured: dict[str, object] = {}

    def restricted_failure(*args: object, **kwargs: object) -> object:
        args[0].read(1)
        raise RuntimeError("unsupported restricted pickle")

    def fake_child(
        checkpoint: Path,
        environment: dict[str, str],
    ) -> dict[str, torch.Tensor]:
        captured.update(
            checkpoint=checkpoint,
            environment=environment,
        )
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(torch, "load", restricted_failure)
    monkeypatch.setattr(
        "cloudpoint_worker.model_prep.convert._trusted_load_in_child", fake_child
    )
    monkeypatch.setenv("HOME", "/secret/home")
    monkeypatch.setenv("TOKEN", "secret")
    state, trusted = load_checkpoint(path)
    assert set(state) == {"weight"}
    assert trusted is True
    assert captured["environment"] == {
        "LC_ALL": "C",
        "PATH": os.environ["PATH"],
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }


def test_prepare_writes_atomic_deterministic_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "model": {
                "z.weight": torch.tensor([3.0, 4.0]),
                "a.weight": torch.tensor([1.0, 2.0]),
            },
            "metadata": {"ignored": True},
        },
        checkpoint,
    )
    destination = tmp_path / "converted"
    destination.mkdir()
    specs = (
        WeightSpec("z.weight", "z.weight", (2,), (2,), "identity"),
        WeightSpec("a.weight", "a.weight", (2,), (2,), "identity"),
    )
    source_bytes = checkpoint.read_bytes()
    monkeypatch.setattr(
        "cloudpoint_worker.model_prep.convert.MODEL_SIZE", len(source_bytes)
    )
    monkeypatch.setattr(
        "cloudpoint_worker.model_prep.convert.MODEL_SHA256",
        hashlib.sha256(source_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        "cloudpoint_worker.model_prep.convert.verify_checkpoint",
        lambda path: VerifiedArtifact(
            path.resolve(), len(source_bytes), hashlib.sha256(source_bytes).hexdigest()
        ),
    )

    manifest = prepare_model(checkpoint, destination, specs)

    weights_path = destination / "lingbot-map-long-f16.safetensors"
    rows = json.loads((destination / "weights-manifest.json").read_text())
    model_manifest = json.loads((destination / "model-manifest.json").read_text())
    assert set(load_file(weights_path)) == {"a.weight", "z.weight"}
    assert [row["destinationKey"] for row in rows] == ["a.weight", "z.weight"]
    assert model_manifest["tensorCount"] == 2
    assert model_manifest["convertedSha256"] == hashlib.sha256(
        weights_path.read_bytes()
    ).hexdigest()
    assert manifest.tensor_count == 2
    assert not list(destination.glob("*.partial"))


def test_atomic_replace_failure_removes_partial_and_never_publishes_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        _atomic_write_bytes(destination, b"content")

    assert not destination.exists()
    assert not destination.with_name("manifest.json.partial").exists()


def test_weight_specs_are_deterministic_and_unique() -> None:
    first = build_weight_specs(ModelConfig())
    second = build_weight_specs(ModelConfig())
    assert first == second
    assert first
    assert len({spec.source_key for spec in first}) == len(first)
    assert len({spec.destination_key for spec in first}) == len(first)
    assert {spec.transform for spec in first} == {
        "identity",
        "conv2d",
        "conv_transpose2d",
    }
    assert len(first) == 1342
    assert any(
        spec.source_key.startswith("aggregator.patch_embed.") for spec in first
    )
    assert any(
        spec.source_key.startswith("aggregator.frame_blocks.23.") for spec in first
    )
    assert any(
        spec.source_key.startswith("aggregator.global_blocks.23.") for spec in first
    )
    assert any(spec.source_key.startswith("depth_head.") for spec in first)
    assert any(spec.source_key.startswith("camera_head.") for spec in first)


def test_prepare_has_no_network_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket, "socket", lambda *args, **kwargs: pytest.fail("network used")
    )
    assert set(
        model_prepare_parser()
        .parse_args(
            [
                "prepare",
                "--checkpoint",
                "/tmp/source.pt",
                "--destination",
                "/tmp/converted",
            ]
        )
        .__dict__
    ) == {"command", "checkpoint", "destination"}


def test_cli_requires_absolute_paths() -> None:
    parser = model_prepare_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["prepare", "--checkpoint", "source.pt", "--destination", "/tmp/output"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["verify", "--checkpoint", "source.pt"])
