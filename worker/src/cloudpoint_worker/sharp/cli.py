"""One-shot JSON-lines entry point for Apple SHARP reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from .session import reconstruct, result_payload

PROTOCOL_VERSION = 1
ENGINE_VERSION = "0.1.0-sharp"


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="cloudpoint-sharp")
    result.add_argument("--project", required=True, type=_absolute_path)
    result.add_argument("--checkpoint", required=True, type=_absolute_path)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--input-relative-path", required=True)
    result.add_argument("--output-relative-path", required=True)
    result.add_argument(
        "--prefer-mps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return result


def _writer():
    lock = threading.Lock()

    def write(event: dict[str, object]) -> None:
        value = {"protocolVersion": PROTOCOL_VERSION, **event}
        encoded = json.dumps(
            value,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        with lock:
            sys.stdout.write(encoded + "\n")
            sys.stdout.flush()

    return write


def main(arguments: Sequence[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    write = _writer()
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(5):
            write({"type": "heartbeat", "monotonicSeconds": time.monotonic()})

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        result = reconstruct(
            project_root=options.project,
            checkpoint=options.checkpoint,
            input_relative_path=options.input_relative_path,
            output_relative_path=options.output_relative_path,
            prefer_mps=options.prefer_mps,
            checkpoint_sha256=options.checkpoint_sha256,
            source_commit=options.source_commit,
            emit=write,
        )
        write({"type": "completed", **result_payload(result)})
        return 0
    except KeyboardInterrupt:
        write({"type": "cancelled"})
        return 130
    except Exception as error:
        write(
            {
                "type": "failed",
                "code": "SHARP_INFERENCE_FAILED",
                "message": str(error) or type(error).__name__,
                "recoverable": True,
            }
        )
        return 4
    finally:
        stopped.set()
        heartbeat_thread.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
