"""Network-free command-line interface for pinned model preparation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.model.config import ModelConfig
from cloudpoint_worker.model.weight_specs import build_weight_specs
from cloudpoint_worker.model_prep.convert import prepare_model
from cloudpoint_worker.model_prep.provenance import verify_checkpoint


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def model_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cloudpoint-model")
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--checkpoint", required=True, type=_absolute_path)
    prepare.add_argument("--destination", required=True, type=_absolute_path)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--checkpoint", required=True, type=_absolute_path)
    return parser


def _write_json_line(value: object, *, stream: object = sys.stdout) -> None:
    stream.write(  # type: ignore[attr-defined]
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    stream.flush()  # type: ignore[attr-defined]


def _progress(phase: str) -> None:
    _write_json_line({"phase": phase})


def _exit_code(fault: WorkerFault) -> int:
    if fault.code in {"MODEL_CHECKSUM_MISMATCH", "MODEL_INVALID_PATH"}:
        return 2
    return 4


def main(argv: Sequence[str] | None = None) -> int:
    arguments = model_prepare_parser().parse_args(argv)
    try:
        if arguments.command == "verify":
            _progress("verifying")
            artifact = verify_checkpoint(arguments.checkpoint)
            _write_json_line(
                {
                    "path": str(artifact.path),
                    "phase": "ready",
                    "sha256": artifact.sha256,
                    "size": artifact.size,
                }
            )
            return 0
        prepare_model(
            arguments.checkpoint,
            arguments.destination,
            build_weight_specs(ModelConfig()),
            progress=_progress,
        )
        return 0
    except WorkerFault as fault:
        _write_json_line(
            {
                "code": fault.code,
                "message": fault.message,
                "recoverable": fault.recoverable,
            },
            stream=sys.stderr,
        )
        return _exit_code(fault)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "model_prepare_parser"]
