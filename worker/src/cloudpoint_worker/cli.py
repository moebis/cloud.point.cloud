"""Independent command-line entry point for CloudPoint's MLX worker."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import signal
import stat
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

from cloudpoint_worker import ENGINE_VERSION
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.model.lingbot import CONVERTED_MODEL_SHA256, LingbotMap
from cloudpoint_worker.model_prep.provenance import MODEL_REPO, MODEL_REVISION
from cloudpoint_worker.protocol.fixtures import write_protocol_fixture
from cloudpoint_worker.protocol.framing import encode_canonical_json
from cloudpoint_worker.protocol.schema import (
    BeginSessionPayload,
    ConfigurePayload,
    ModelProgress,
    ModelProgressPayload,
    Ready,
    ReadyPayload,
)
from cloudpoint_worker.server import WorkerServer
from cloudpoint_worker.session import (
    PersistedFrame,
    SessionRunner,
    validate_project_root,
)

EXIT_SUCCESS = 0
EXIT_SETUP = 2
EXIT_PROTOCOL = 3
EXIT_ENGINE = 4
EXIT_CANCELLED = 130


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cloudpoint-worker")
    subcommands = parser.add_subparsers(dest="command", required=True)

    health = subcommands.add_parser("health")
    health.add_argument("--model", required=True, type=_absolute_path)

    serve = subcommands.add_parser("serve")
    serve.add_argument("--project", required=True, type=_absolute_path)
    serve.add_argument("--model", required=True, type=_absolute_path)

    run = subcommands.add_parser("run")
    run.add_argument("--project", required=True, type=_absolute_path)
    run.add_argument("--model", required=True, type=_absolute_path)
    run.add_argument("--frames", required=True, nargs="+", type=_absolute_path)

    fixture = subcommands.add_parser("protocol-fixture")
    fixture.add_argument("--output", required=True, type=_absolute_path)
    return parser


def _write_json_line(value: object, stream: BinaryIO | None = None) -> None:
    destination = stream or sys.stdout.buffer
    destination.write(encode_canonical_json(value) + b"\n")
    destination.flush()


def _error_value(fault: WorkerFault) -> dict[str, object]:
    return {
        "error": {
            "code": fault.code,
            "details": dict(fault.details),
            "message": fault.message,
            "recoverable": fault.recoverable,
        }
    }


def _setup_fault(code: str, message: str) -> WorkerFault:
    return WorkerFault(code, message, True)


def _validate_runtime() -> None:
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or sys.version_info[:2] != (3, 12)
        or importlib.metadata.version("mlx") != "0.32.0"
    ):
        raise WorkerFault(
            "RUNTIME_INCOMPATIBLE",
            "CloudPoint requires macOS on Apple Silicon, Python 3.12, and MLX 0.32.0",
            False,
        )


def _load_model(model_directory: Path) -> LingbotMap:
    _validate_runtime()
    return LingbotMap.load(model_directory)


def _health(model_directory: Path) -> int:
    try:
        _load_model(model_directory)
    except WorkerFault as fault:
        unavailable = WorkerFault(
            "MODEL_UNAVAILABLE",
            "the prepared Lingbot MLX model is unavailable or invalid",
            True,
            {"reason": fault.code},
        )
        _write_json_line(_error_value(unavailable))
        return EXIT_SETUP
    _write_json_line(
        {
            "engineVersion": ENGINE_VERSION,
            "modelIdentifier": MODEL_REPO,
            "modelRevision": MODEL_REVISION,
            "convertedWeightsSHA256": CONVERTED_MODEL_SHA256,
            "status": "ready",
        }
    )
    return EXIT_SUCCESS


def _validate_source_frame(path: Path) -> None:
    if not path.is_absolute() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise _setup_fault(
            "FRAME_INVALID_PATH", "input frames must be absolute JPEG or PNG files"
        )
    try:
        info = path.lstat()
    except OSError as error:
        raise _setup_fault(
            "FRAME_INVALID_PATH", "input frame does not exist"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise _setup_fault("FRAME_INVALID_PATH", "input frame must be a regular file")


def _persist_cli_frames(
    project_root: Path, sources: Sequence[Path]
) -> tuple[PersistedFrame, ...]:
    frames: list[PersistedFrame] = []
    for index, source in enumerate(sources):
        _validate_source_frame(source)
        suffix = source.suffix.lower()
        relative = f"Frames/{index:08d}{suffix}"
        destination = project_root / relative
        # Direct run is intentionally no-clobber. Existing frames may be native-
        # persisted inputs when their canonical path is supplied directly.
        if source == destination:
            frames.append(PersistedFrame(index, float(index), relative))
            continue
        try:
            with source.open("rb") as input_stream, destination.open("xb") as output:
                shutil.copyfileobj(input_stream, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        except FileExistsError as error:
            raise WorkerFault(
                "OUTPUT_ALREADY_EXISTS",
                f"direct-run frame already exists: {relative}",
                True,
            ) from error
        except OSError as error:
            raise WorkerFault(
                "FRAME_IMPORT_FAILED", "input frame could not be persisted", True
            ) from error
        frames.append(PersistedFrame(index, float(index), relative))
    return tuple(frames)


def _run(project_root: Path, model_directory: Path, sources: Sequence[Path]) -> int:
    # Project validation intentionally precedes model loading so setup failures are
    # specific and do not allocate a 2.2 GB model unnecessarily.
    snapshot = validate_project_root(project_root)
    frames = _persist_cli_frames(project_root, sources)
    try:
        _write_json_line(
            ModelProgress(
                1,
                uuid.uuid4(),
                snapshot.project_id,
                "modelProgress",
                ModelProgressPayload("validating", 0, 1),
            )
        )
        model = _load_model(model_directory)
        _write_json_line(
            ModelProgress(
                1,
                uuid.uuid4(),
                snapshot.project_id,
                "modelProgress",
                ModelProgressPayload("loading", 1, 1),
            )
        )
        _write_json_line(
            Ready(
                1,
                uuid.uuid4(),
                snapshot.project_id,
                "ready",
                ReadyPayload(
                    ENGINE_VERSION,
                    MODEL_REPO,
                    MODEL_REVISION,
                    CONVERTED_MODEL_SHA256,
                ),
            )
        )
        runner = SessionRunner(
            project_root,
            model,
            _write_json_line,
            project_id=snapshot.project_id,
            cleanup_orphans=False,
        )
        runner.configure(ConfigurePayload(8, 32, 8, 1, 4, 1.5, 0.01))
        runner.begin(BeginSessionPayload(None))
        for frame in frames:
            runner.enqueue(frame)
        runner.finish_input()

        interrupted = False
        previous_handler = signal.getsignal(signal.SIGINT)

        def interrupt(_signum: int, _frame: object) -> None:
            nonlocal interrupted
            if interrupted:
                raise KeyboardInterrupt
            interrupted = True
            runner.cancel()

        signal.signal(signal.SIGINT, interrupt)
        try:
            runner.process()
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        return EXIT_CANCELLED if interrupted else EXIT_SUCCESS
    except KeyboardInterrupt:
        return EXIT_CANCELLED
    except WorkerFault as fault:
        _write_json_line(_error_value(fault))
        return _exit_code(fault)


def _exit_code(fault: WorkerFault) -> int:
    if fault.code in {
        "PROJECT_INVALID_PATH",
        "PROJECT_INVALID_MANIFEST",
        "PROJECT_UNSUPPORTED_FORMAT",
        "PROJECT_UNSUPPORTED_MODE",
        "FRAME_INVALID_PATH",
        "MODEL_UNAVAILABLE",
        "MODEL_INVALID_PATH",
        "OUTPUT_ALREADY_EXISTS",
    }:
        return EXIT_SETUP
    if fault.code in {
        "PATH_OUTSIDE_PROJECT",
        "PROJECT_ID_MISMATCH",
        "INVALID_RESUME_CHECKPOINT",
        "REPLAY_ORDER_VIOLATION",
    }:
        return EXIT_PROTOCOL
    return EXIT_ENGINE


def main(argv: Sequence[str] | None = None) -> int:
    arguments = worker_parser().parse_args(argv)
    try:
        if arguments.command == "health":
            return _health(arguments.model)
        if arguments.command == "protocol-fixture":
            write_protocol_fixture(arguments.output)
            return EXIT_SUCCESS
        if arguments.command == "serve":
            server = WorkerServer(
                arguments.project,
                model_loader=lambda: _load_model(arguments.model),
            )
            return server.serve()
        if arguments.command == "run":
            return _run(arguments.project, arguments.model, arguments.frames)
        return EXIT_SETUP
    except WorkerFault as fault:
        # Serve cannot write unframed JSON after transport starts. All failures
        # before construction are diagnostics, never protocol-lookalike stdout.
        if getattr(arguments, "command", None) == "serve":
            print(f"{fault.code}: {fault.message}", file=sys.stderr, flush=True)
            return _exit_code(fault)
        _write_json_line(_error_value(fault))
        return _exit_code(fault)


if __name__ == "__main__":  # pragma: no cover - subprocess exercised.
    raise SystemExit(main())


__all__ = ["main", "worker_parser"]
