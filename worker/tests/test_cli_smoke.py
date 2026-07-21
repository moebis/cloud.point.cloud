"""Subprocess smoke coverage for the independently runnable worker CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cloudpoint_worker.cpc import read_cpc


def test_base_dependency_cli_help_does_not_import_optional_torch() -> None:
    uv = shutil.which("uv")
    assert uv is not None

    result = subprocess.run(
        [uv, "run", "--isolated", "--frozen", "cloudpoint-worker", "--help"],
        cwd=Path(__file__).parents[1],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: cloudpoint-worker" in result.stdout
    assert "torch" not in result.stderr


def _make_project(root: Path) -> Path:
    project = root / "RealSmoke.cloudpoint"
    project.mkdir()
    for name in ("Frames", "Predictions", "Points", "Logs"):
        (project / name).mkdir()
    manifest = {
        "completedWindows": [],
        "createdAt": "2026-07-21T00:00:00.0Z",
        "engineConfiguration": {
            "cameraRefinementIterations": 4,
            "confidenceThreshold": 1.5,
            "keyframeInterval": 1,
            "scaleFrames": 8,
            "voxelSize": 0.01,
            "windowOverlap": 8,
            "windowSize": 32,
        },
        "formatVersion": 2,
        "frames": [],
        "projectID": "33333333-3333-3333-3333-333333333333",
        "sessionState": {
            "capturedCount": 0,
            "currentWindow": None,
            "failedCount": 0,
            "isCapturing": False,
            "phase": "empty",
            "processedCount": 0,
            "queuedCount": 0,
        },
        "updatedAt": "2026-07-21T00:00:00.0Z",
    }
    (project / "Manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return project


def test_health_failure_is_machine_readable(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cloudpoint_worker.cli",
            "health",
            "--model",
            str(tmp_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert json.loads(result.stdout)["error"]["code"] == "MODEL_UNAVAILABLE"
    assert result.stderr == ""


def test_run_rejects_non_project_directory(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cloudpoint_worker.cli",
            "run",
            "--project",
            str(tmp_path),
            "--model",
            str(model),
            "--frames",
            str(tmp_path / "missing.jpg"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert json.loads(result.stdout)["error"]["code"] == "PROJECT_INVALID_PATH"
    assert result.stderr == ""


def test_serve_setup_failure_never_writes_unframed_stdout(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cloudpoint_worker.cli",
            "serve",
            "--project",
            str(tmp_path),
            "--model",
            str(model),
        ],
        check=False,
        capture_output=True,
    )

    assert result.returncode == 2
    assert result.stdout == b""
    assert b"PROJECT_INVALID_PATH" in result.stderr


def test_protocol_fixture_subcommand_is_byte_stable(tmp_path: Path) -> None:
    output = tmp_path / "protocol-v1.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cloudpoint_worker.cli",
            "protocol-fixture",
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert (
        output.read_bytes()
        == (Path(__file__).parent / "fixtures/protocol-v1.json").read_bytes()
    )
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.real_model
def test_real_run_writes_scene_geometry_from_all_configured_frames(
    tmp_path: Path,
) -> None:
    model_value = os.environ.get("CLOUDPOINT_MODEL_DIR")
    frames_value = os.environ.get("CLOUDPOINT_REAL_FRAMES")
    if not model_value or not frames_value:
        pytest.skip("CLOUDPOINT_MODEL_DIR and CLOUDPOINT_REAL_FRAMES are required")
    model = Path(model_value)
    frame_root = Path(frames_value)
    frames = sorted(frame_root.glob("*.jpg")) or sorted(frame_root.glob("*.png"))
    if not 1 <= len(frames) <= 32:
        pytest.skip("real smoke requires 1 through 32 ordered frames")
    project = _make_project(tmp_path)
    manifest_before = (project / "Manifest.json").read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cloudpoint_worker.cli",
            "run",
            "--project",
            str(project),
            "--model",
            str(model),
            "--frames",
            *(str(path) for path in frames),
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["type"] == "sessionCompleted"
    assert events[-1]["payload"]["processedFrames"] == len(frames)
    assert len(list((project / "Predictions").glob("*.geometry.json"))) == len(frames)
    chunk = read_cpc(project / "Points/window-00000000.cpc")
    assert chunk.point_count > 1_000
    assert (chunk.descriptor.frame_start, chunk.descriptor.frame_end) == (
        0,
        len(frames) - 1,
    )
    assert len({vertex.source_frame for vertex in chunk.vertices}) == len(frames)
    assert (project / "Manifest.json").read_bytes() == manifest_before
