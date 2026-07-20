"""Deterministic Swift/Python protocol-v1 compatibility corpus."""

from __future__ import annotations

import argparse
import uuid
from decimal import Decimal
from pathlib import Path

from cloudpoint_worker import ENGINE_VERSION, PROTOCOL_VERSION
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.protocol.framing import (
    MAX_MESSAGE_BYTES,
    RawJSONNumber,
    encode_canonical_json,
)
from cloudpoint_worker.protocol.schema import (
    ErrorMessage,
    ErrorPayload,
    ack,
    command_error,
    decode_command,
    decode_event,
)


def _uuid(value: int) -> uuid.UUID:
    return uuid.UUID(int=value)


def _command(
    identifier: uuid.UUID,
    project_id: uuid.UUID,
    kind: str,
    payload: object,
) -> object:
    return decode_command(
        {
            "protocolVersion": PROTOCOL_VERSION,
            "id": str(identifier),
            "projectId": str(project_id),
            "type": kind,
            "payload": payload,
        }
    )


def _event(
    identifier: uuid.UUID,
    project_id: uuid.UUID,
    kind: str,
    payload: object,
    *,
    command_id: uuid.UUID | None | object = ...,
) -> object:
    value: dict[str, object] = {
        "protocolVersion": PROTOCOL_VERSION,
        "id": str(identifier),
        "projectId": str(project_id),
        "type": kind,
        "payload": payload,
    }
    if command_id is not ...:
        value["commandId"] = None if command_id is None else str(command_id)
    return decode_event(value)


def _message_row(name: str, message: object) -> dict[str, object]:
    body = encode_canonical_json(message)
    framed = len(body).to_bytes(4, "big") + body
    return {"name": name, "json": body.decode("utf-8"), "framedBytes": list(framed)}


def _rejection_row(
    name: str,
    framed: bytes,
    expected_disposition: str,
    *,
    body: bytes | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "json": None if body is None else body.decode("utf-8", errors="replace"),
        "framedBytes": list(framed),
        "expectedDisposition": expected_disposition,
    }


def build_protocol_fixture() -> dict[str, object]:
    """Build the complete deterministic compatibility corpus in wire order."""

    project_id = _uuid(1)
    next_identifier = 2

    def identifier() -> uuid.UUID:
        nonlocal next_identifier
        result = _uuid(next_identifier)
        next_identifier += 1
        return result

    configuration = {
        "scaleFrames": 8,
        "windowSize": 32,
        "windowOverlap": 8,
        "keyframeInterval": 1,
        "cameraRefinementIterations": 4,
        "confidenceThreshold": 1.5,
        "voxelSize": 0.01,
    }
    checkpoint = {
        "lastCommittedFrameIndex": 44,
        "replayFromFrameIndex": 31,
        "nextWindowIndex": 2,
    }

    commands = [
        (
            "hello",
            _command(
                identifier(),
                project_id,
                "hello",
                {"clientVersion": "0.1.0", "supportedProtocolVersions": [1]},
            ),
        ),
        ("configure", _command(identifier(), project_id, "configure", configuration)),
        (
            "beginSession-null",
            _command(
                identifier(), project_id, "beginSession", {"resumeCheckpoint": None}
            ),
        ),
        (
            "beginSession-full",
            _command(
                identifier(),
                project_id,
                "beginSession",
                {"resumeCheckpoint": checkpoint},
            ),
        ),
        (
            "enqueueFrame",
            _command(
                identifier(),
                project_id,
                "enqueueFrame",
                {
                    "frameIndex": 7,
                    "sourceTimestamp": 1e100,
                    "relativePath": "Frames/00000007.jpg",
                },
            ),
        ),
        ("finishInput", _command(identifier(), project_id, "finishInput", {})),
        ("pause", _command(identifier(), project_id, "pause", {})),
        ("resume", _command(identifier(), project_id, "resume", {})),
        ("cancel", _command(identifier(), project_id, "cancel", {})),
        ("shutdown", _command(identifier(), project_id, "shutdown", {})),
    ]

    hello = commands[0][1]
    pause = commands[6][1]
    messages = [_message_row(f"command.{name}", command) for name, command in commands]
    messages.extend(
        [
            _message_row("response.ack", ack(hello, id=identifier())),
            _message_row(
                "response.command-error",
                command_error(
                    pause,
                    WorkerFault(
                        "PAUSE_FAILED",
                        "pause could not reach a boundary",
                        True,
                        {"attempt": 2},
                    ),
                    id=identifier(),
                ),
            ),
            _message_row(
                "response.asynchronous-error",
                ErrorMessage(
                    PROTOCOL_VERSION,
                    identifier(),
                    project_id,
                    "error",
                    None,
                    ErrorPayload(
                        "INFERENCE_FAILED",
                        "inference failed",
                        False,
                        {
                            "decimal": Decimal("2.500"),
                            "raw": RawJSONNumber("1.2300e+04"),
                            "signed": -(2**63),
                            "unsigned": 2**64 - 1,
                        },
                    ),
                ),
            ),
        ]
    )

    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    error_payload = {
        "code": "LOW_CONFIDENCE",
        "message": "some points were filtered",
        "recoverable": True,
        "details": {"threshold": RawJSONNumber("1.500")},
    }
    event_specs: list[tuple[str, str, object, object]] = [
        (
            "ready",
            "ready",
            {
                "engineVersion": ENGINE_VERSION,
                "modelIdentifier": "robbyant/lingbot-map",
                "modelRevision": "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9",
                "convertedWeightsSHA256": "a" * 64,
            },
            ...,
        ),
        (
            "modelProgress-validating",
            "modelProgress",
            {"phase": "validating", "completed": 1, "total": 2},
            ...,
        ),
        (
            "modelProgress-loading",
            "modelProgress",
            {"phase": "loading", "completed": 2, "total": 2},
            ...,
        ),
        ("frameStarted", "frameStarted", {"frameIndex": 7, "windowIndex": 2}, ...),
        (
            "frameCompleted",
            "frameCompleted",
            {
                "frameIndex": 7,
                "windowIndex": 2,
                "depthPath": "Predictions/00000007.depth-f16",
                "confidencePath": "Predictions/00000007.confidence-f16",
                "geometryPath": "Predictions/00000007.geometry.json",
                "durationSeconds": 0.25,
            },
            ...,
        ),
        (
            "windowCompleted",
            "windowCompleted",
            {
                "windowIndex": 2,
                "inferenceFrameStart": 31,
                "frameStart": 40,
                "frameEnd": 44,
                "pointChunkPath": "Points/window-00000002.cpc",
                "alignmentTransform": identity,
                "lastProcessedFrameIndex": 44,
                "inlierCount": 99,
                "durationSeconds": 1,
            },
            ...,
        ),
        (
            "sessionCompleted",
            "sessionCompleted",
            {"processedFrames": 5, "windowCount": 1, "durationSeconds": 2.5},
            ...,
        ),
        ("paused", "paused", {"queuedFrames": 8, "processedFrames": 7}, ...),
        ("cancelled-null", "cancelled", {"lastCompletedWindowIndex": None}, ...),
        ("cancelled-full", "cancelled", {"lastCompletedWindowIndex": 2}, ...),
        ("warning", "warning", error_payload, ...),
        (
            "heartbeat-null",
            "heartbeat",
            {
                "busy": False,
                "monotonicSeconds": -0.0,
                "queuedFrames": 0,
                "processedFrames": 0,
                "currentWindow": None,
            },
            ...,
        ),
        (
            "heartbeat-full",
            "heartbeat",
            {
                "busy": True,
                "monotonicSeconds": 1.25,
                "queuedFrames": 8,
                "processedFrames": 7,
                "currentWindow": 2,
            },
            ...,
        ),
    ]
    for name, kind, payload, command_id in event_specs:
        messages.append(
            _message_row(
                f"event.{name}",
                _event(
                    identifier(),
                    project_id,
                    kind,
                    payload,
                    command_id=command_id,
                ),
            )
        )

    rejection_values = [
        (
            "nested-checkpoint-missing",
            {
                "protocolVersion": 1,
                "id": str(identifier()),
                "projectId": str(project_id),
                "type": "beginSession",
                "payload": {
                    "resumeCheckpoint": {
                        "lastCommittedFrameIndex": 44,
                        "replayFromFrameIndex": 31,
                    }
                },
            },
            "commandErrorThenContinue",
        ),
        (
            "nested-checkpoint-unknown",
            {
                "protocolVersion": 1,
                "id": str(identifier()),
                "projectId": str(project_id),
                "type": "beginSession",
                "payload": {"resumeCheckpoint": {**checkpoint, "extra": 0}},
            },
            "commandErrorThenContinue",
        ),
        (
            "nested-configuration-missing",
            {
                "protocolVersion": 1,
                "id": str(identifier()),
                "projectId": str(project_id),
                "type": "configure",
                "payload": {
                    key: value
                    for key, value in configuration.items()
                    if key != "voxelSize"
                },
            },
            "commandErrorThenContinue",
        ),
        (
            "nested-configuration-unknown",
            {
                "protocolVersion": 1,
                "id": str(identifier()),
                "projectId": str(project_id),
                "type": "configure",
                "payload": {**configuration, "extra": 0},
            },
            "commandErrorThenContinue",
        ),
        (
            "uppercase-command-uuid",
            {
                "protocolVersion": 1,
                "id": str(identifier()).upper(),
                "projectId": str(project_id),
                "type": "pause",
                "payload": {},
            },
            "asynchronousErrorThenClose",
        ),
        (
            "unsupported-version",
            {
                "protocolVersion": 2,
                "id": str(identifier()),
                "projectId": str(project_id),
                "type": "pause",
                "payload": {},
            },
            "commandErrorThenClose",
        ),
    ]
    rejections: list[dict[str, object]] = []
    for name, value, disposition in rejection_values:
        body = encode_canonical_json(value)
        rejections.append(
            _rejection_row(
                name,
                len(body).to_bytes(4, "big") + body,
                disposition,
                body=body,
            )
        )

    rejections.append(
        _rejection_row("zero-length", b"\0\0\0\0", "closeWithoutResponse")
    )
    rejections.append(
        _rejection_row(
            "oversized-length",
            (MAX_MESSAGE_BYTES + 1).to_bytes(4, "big"),
            "closeWithoutResponse",
        )
    )
    valid_frame = bytes(messages[0]["framedBytes"])
    rejections.append(
        _rejection_row("truncated-frame", valid_frame[:-1], "closeWithoutResponse")
    )
    invalid_body = b"{"
    rejections.append(
        _rejection_row(
            "invalid-json",
            len(invalid_body).to_bytes(4, "big") + invalid_body,
            "asynchronousErrorThenClose",
            body=invalid_body,
        )
    )

    return {
        "protocolVersion": PROTOCOL_VERSION,
        "maximumMessageBytes": MAX_MESSAGE_BYTES,
        "messages": messages,
        "rejections": rejections,
    }


def write_protocol_fixture(path: Path) -> None:
    """Atomically replace ``path`` with the canonical compatibility corpus."""

    path.parent.mkdir(parents=True, exist_ok=True)
    body = encode_canonical_json(build_protocol_fixture()) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4()}.partial")
    try:
        with temporary.open("xb") as destination:
            destination.write(body)
            destination.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    write_protocol_fixture(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_protocol_fixture", "main", "write_protocol_fixture"]
