from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from plyfile import PlyData

from cloudpoint_worker.model._vendor.ml_sharp.utils.gaussians import Gaussians3D
from cloudpoint_worker.sharp.session import (
    SharpSessionError,
    reconstruct,
)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "Scene.cloudpoint"
    (project / "Frames").mkdir(parents=True)
    (project / "Outputs" / "Gaussians").mkdir(parents=True)
    Image.new("RGB", (8, 6), (120, 80, 40)).save(project / "Frames" / "00000000.jpg")
    return project


def _gaussians(depth: float = 2.0) -> Gaussians3D:
    return Gaussians3D(
        mean_vectors=torch.tensor([[[0.0, 0.0, depth], [0.2, 0.1, depth + 0.3]]]),
        singular_values=torch.full((1, 2, 3), 0.02),
        quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
        colors=torch.tensor([[[0.5, 0.25, 0.1], [0.1, 0.3, 0.7]]]),
        opacities=torch.tensor([[0.8, 0.6]]),
    )


def test_fake_predictor_writes_exact_validated_ply_and_provenance_atomically(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    checkpoint = tmp_path / "sharp.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    stages: list[dict[str, object]] = []

    result = reconstruct(
        project_root=project,
        checkpoint=checkpoint,
        input_relative_path="Frames/00000000.jpg",
        output_relative_path="Outputs/Gaussians/00000000.ply",
        prefer_mps=True,
        checkpoint_sha256="a" * 64,
        source_commit="fixture-commit",
        predictor=lambda image, focal_px, device, emit: _gaussians(),
        mps_available=lambda: True,
        emit=stages.append,
    )

    assert result.gaussian_count == 2
    assert result.device == "mps"
    assert result.used_cpu_fallback is False
    assert result.ply_relative_path == "Outputs/Gaussians/00000000.ply"
    assert result.provenance_relative_path == "Outputs/Gaussians/00000000.json"
    assert [event["stage"] for event in stages if event["type"] == "progress"] == [
        "loading",
        "inference",
        "validating",
        "committing",
    ]

    ply = PlyData.read(project / result.ply_relative_path)
    vertex = ply["vertex"]
    assert vertex.count == 2
    assert vertex.properties
    assert [property.name for property in vertex.properties] == [
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
    ]
    assert np.isfinite(np.asarray(vertex.data.tolist(), dtype=np.float64)).all()
    assert (np.asarray(vertex["z"]) > 0).all()

    provenance = json.loads((project / result.provenance_relative_path).read_text())
    assert provenance["checkpointSHA256"] == "a" * 64
    assert provenance["sourceCommit"] == "fixture-commit"
    assert provenance["gaussianCount"] == 2
    assert provenance["device"] == "mps"
    assert not list((project / "Outputs" / "Gaussians").glob("*.partial*"))


def test_recoverable_mps_failure_retries_once_on_cpu(tmp_path: Path) -> None:
    project = _project(tmp_path)
    checkpoint = tmp_path / "sharp.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    calls: list[str] = []
    events: list[dict[str, object]] = []

    def predictor(image, focal_px, device, emit):
        del image, focal_px, emit
        calls.append(device)
        if device == "mps":
            raise RuntimeError("MPS backend out of memory")
        return _gaussians()

    result = reconstruct(
        project_root=project,
        checkpoint=checkpoint,
        input_relative_path="Frames/00000000.jpg",
        output_relative_path="Outputs/Gaussians/00000000.ply",
        prefer_mps=True,
        checkpoint_sha256="b" * 64,
        source_commit="fixture",
        predictor=predictor,
        mps_available=lambda: True,
        emit=events.append,
    )

    assert calls == ["mps", "cpu"]
    assert result.device == "cpu"
    assert result.used_cpu_fallback is True
    assert any(event["type"] == "warning" for event in events)


@pytest.mark.parametrize("depth", [float("nan"), -1.0, 0.0])
def test_invalid_gaussians_never_publish(depth: float, tmp_path: Path) -> None:
    project = _project(tmp_path)
    checkpoint = tmp_path / "sharp.pt"
    checkpoint.write_bytes(b"fixture checkpoint")

    with pytest.raises(SharpSessionError, match="invalid Gaussian"):
        reconstruct(
            project_root=project,
            checkpoint=checkpoint,
            input_relative_path="Frames/00000000.jpg",
            output_relative_path="Outputs/Gaussians/00000000.ply",
            prefer_mps=False,
            checkpoint_sha256="c" * 64,
            source_commit="fixture",
            predictor=lambda image, focal_px, device, emit: _gaussians(depth),
            mps_available=lambda: False,
            emit=lambda event: None,
        )

    assert not (project / "Outputs" / "Gaussians" / "00000000.ply").exists()
    assert not list((project / "Outputs" / "Gaussians").glob("*.partial*"))


@pytest.mark.parametrize(
    "relative_path",
    ["../escape.jpg", "/tmp/escape.jpg", "Frames/link.jpg"],
)
def test_input_path_must_be_canonical_regular_file(
    relative_path: str,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    checkpoint = tmp_path / "sharp.pt"
    checkpoint.write_bytes(b"fixture")
    if relative_path == "Frames/link.jpg":
        (project / relative_path).symlink_to(project / "Frames" / "00000000.jpg")

    with pytest.raises(SharpSessionError):
        reconstruct(
            project_root=project,
            checkpoint=checkpoint,
            input_relative_path=relative_path,
            output_relative_path="Outputs/Gaussians/00000000.ply",
            prefer_mps=False,
            checkpoint_sha256="d" * 64,
            source_commit="fixture",
            predictor=lambda image, focal_px, device, emit: _gaussians(),
            mps_available=lambda: False,
            emit=lambda event: None,
        )
