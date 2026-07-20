from __future__ import annotations

import dataclasses
import uuid
from decimal import Decimal

import msgspec
import pytest

from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.protocol.framing import RawJSONNumber, encode_canonical_json
from cloudpoint_worker.protocol.schema import (
    PROTOCOL_VERSION,
    Ack,
    BeginSessionCommand,
    Cancelled,
    CommandIDTracker,
    ConfigureCommand,
    EnqueueFrameCommand,
    ErrorMessage,
    FailureDisposition,
    FrameCompleted,
    FrameStarted,
    Heartbeat,
    HelloCommand,
    ModelProgress,
    Paused,
    ProtocolValidationError,
    Ready,
    SessionCompleted,
    WarningMessage,
    WindowCompleted,
    ack,
    artifact_paths,
    classify_failure,
    command_error,
    decode_command,
    decode_event,
    recover_command_header,
    window_point_path,
)

COMMAND_ID = "00000000-0000-0000-0000-000000000001"
PROJECT_ID = "00000000-0000-0000-0000-000000000002"


def command_value(kind: str, payload: object) -> dict[str, object]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "id": COMMAND_ID,
        "projectId": PROJECT_ID,
        "type": kind,
        "payload": payload,
    }


def event_value(
    kind: str,
    payload: object,
    *,
    command_id: object = dataclasses.MISSING,
) -> dict[str, object]:
    value = command_value(kind, payload)
    if command_id is not dataclasses.MISSING:
        value["commandId"] = command_id
    return value


def test_all_nine_commands_decode_to_strict_typed_structs() -> None:
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
    cases = [
        (
            "hello",
            {"clientVersion": "0.1.0", "supportedProtocolVersions": [1]},
            HelloCommand,
        ),
        ("configure", configuration, ConfigureCommand),
        ("beginSession", {"resumeCheckpoint": None}, BeginSessionCommand),
        ("beginSession", {"resumeCheckpoint": checkpoint}, BeginSessionCommand),
        (
            "enqueueFrame",
            {
                "frameIndex": 7,
                "sourceTimestamp": 1.25,
                "relativePath": "Frames/00000007.jpg",
            },
            EnqueueFrameCommand,
        ),
        ("finishInput", {}, object),
        ("pause", {}, object),
        ("resume", {}, object),
        ("cancel", {}, object),
        ("shutdown", {}, object),
    ]

    for kind, payload, expected_type in cases:
        command = decode_command(command_value(kind, payload))
        assert command.type == kind
        assert command.id == uuid.UUID(COMMAND_ID)
        assert command.project_id == uuid.UUID(PROJECT_ID)
        if expected_type is not object:
            assert isinstance(command, expected_type)


def test_command_requires_exact_envelope_version_uuid_project_and_payload() -> None:
    valid = command_value("pause", {})
    variants = [
        {**valid, "protocolVersion": 2},
        {**valid, "id": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"},
        {**valid, "id": COMMAND_ID.replace("-", "")},
        {**valid, "projectId": "BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB"},
        {**valid, "extra": 1},
        {key: value for key, value in valid.items() if key != "payload"},
        {**valid, "payload": {"extra": 1}},
        {**valid, "type": "future"},
    ]
    for variant in variants:
        with pytest.raises(msgspec.ValidationError):
            decode_command(variant)


def test_nested_payloads_reject_missing_unknown_and_wrong_width_values() -> None:
    configure = {
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
    invalid_payloads = [
        ("configure", {**configure, "extra": 0}),
        (
            "configure",
            {key: value for key, value in configure.items() if key != "voxelSize"},
        ),
        ("configure", {**configure, "scaleFrames": 2**32}),
        ("beginSession", {"resumeCheckpoint": {**checkpoint, "extra": 0}}),
        (
            "beginSession",
            {
                "resumeCheckpoint": {
                    key: value
                    for key, value in checkpoint.items()
                    if key != "nextWindowIndex"
                }
            },
        ),
    ]
    for kind, payload in invalid_payloads:
        with pytest.raises(msgspec.ValidationError):
            decode_command(command_value(kind, payload))


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("hello", {"clientVersion": "native", "supportedProtocolVersions": [2]}),
        (
            "configure",
            {
                "scaleFrames": 0,
                "windowSize": 32,
                "windowOverlap": 8,
                "keyframeInterval": 1,
                "cameraRefinementIterations": 4,
                "confidenceThreshold": 1.5,
                "voxelSize": 0.01,
            },
        ),
        (
            "configure",
            {
                "scaleFrames": 8,
                "windowSize": 8,
                "windowOverlap": 8,
                "keyframeInterval": 1,
                "cameraRefinementIterations": 4,
                "confidenceThreshold": 1.5,
                "voxelSize": 0.01,
            },
        ),
        (
            "beginSession",
            {
                "resumeCheckpoint": {
                    "lastCommittedFrameIndex": 4,
                    "replayFromFrameIndex": 5,
                    "nextWindowIndex": 1,
                }
            },
        ),
        (
            "enqueueFrame",
            {"frameIndex": 1, "sourceTimestamp": -1, "relativePath": "Frames/1.jpg"},
        ),
        (
            "enqueueFrame",
            {"frameIndex": 1, "sourceTimestamp": 0, "relativePath": "/tmp/1.jpg"},
        ),
        (
            "enqueueFrame",
            {"frameIndex": 1, "sourceTimestamp": 0, "relativePath": "Frames/../1.jpg"},
        ),
    ],
)
def test_semantically_invalid_commands_are_rejected(kind: str, payload: object) -> None:
    with pytest.raises(msgspec.ValidationError):
        decode_command(command_value(kind, payload))


def test_ack_and_command_error_retain_command_ownership() -> None:
    command = decode_command(command_value("pause", {}))
    response = ack(command, id=uuid.UUID("00000000-0000-0000-0000-000000000003"))
    failure = command_error(
        command,
        WorkerFault(
            "PAUSE_FAILED", "pause could not reach a boundary", True, {"attempt": 2}
        ),
        id=uuid.UUID("00000000-0000-0000-0000-000000000004"),
    )

    assert isinstance(response, Ack)
    assert response.command_id == command.id
    assert response.payload.command == "pause"
    assert isinstance(failure, ErrorMessage)
    assert failure.command_id == command.id
    assert failure.payload.code == "PAUSE_FAILED"
    assert failure.payload.details == {"attempt": 2}


def test_all_event_payloads_decode_and_validate() -> None:
    error_payload = {
        "code": "probe",
        "message": "probe",
        "recoverable": True,
        "details": {"measurement": RawJSONNumber("1.2300e+04")},
    }
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    cases = [
        ("ack", {"command": "hello"}, COMMAND_ID, Ack),
        ("error", error_payload, None, ErrorMessage),
        (
            "ready",
            {
                "engineVersion": "0.1.0",
                "modelIdentifier": "robbyant/lingbot-map",
                "modelRevision": "revision",
                "convertedWeightsSHA256": "a" * 64,
            },
            dataclasses.MISSING,
            Ready,
        ),
        (
            "modelProgress",
            {"phase": "loading", "completed": 1, "total": 2},
            dataclasses.MISSING,
            ModelProgress,
        ),
        (
            "frameStarted",
            {"frameIndex": 7, "windowIndex": 2},
            dataclasses.MISSING,
            FrameStarted,
        ),
        (
            "frameCompleted",
            {
                "frameIndex": 7,
                "windowIndex": 2,
                "depthPath": "Predictions/00000007.depth-f16",
                "confidencePath": "Predictions/00000007.confidence-f16",
                "geometryPath": "Predictions/00000007.geometry.json",
                "durationSeconds": 0.25,
            },
            dataclasses.MISSING,
            FrameCompleted,
        ),
        (
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
            dataclasses.MISSING,
            WindowCompleted,
        ),
        (
            "sessionCompleted",
            {"processedFrames": 5, "windowCount": 1, "durationSeconds": 2},
            dataclasses.MISSING,
            SessionCompleted,
        ),
        (
            "paused",
            {"queuedFrames": 8, "processedFrames": 7},
            dataclasses.MISSING,
            Paused,
        ),
        (
            "cancelled",
            {"lastCompletedWindowIndex": None},
            dataclasses.MISSING,
            Cancelled,
        ),
        ("warning", error_payload, dataclasses.MISSING, WarningMessage),
        (
            "heartbeat",
            {
                "busy": True,
                "monotonicSeconds": 0,
                "queuedFrames": 8,
                "processedFrames": 7,
                "currentWindow": 2,
            },
            dataclasses.MISSING,
            Heartbeat,
        ),
    ]

    for kind, payload, command_id, expected_type in cases:
        decoded = decode_event(event_value(kind, payload, command_id=command_id))
        assert isinstance(decoded, expected_type)
        assert decoded.type == kind


def test_event_validation_rejects_bad_counters_paths_bounds_and_numbers() -> None:
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    invalid = [
        event_value("modelProgress", {"phase": "loading", "completed": 3, "total": 2}),
        event_value("paused", {"queuedFrames": 3, "processedFrames": 4}),
        event_value(
            "heartbeat",
            {
                "busy": False,
                "monotonicSeconds": 0,
                "queuedFrames": 0,
                "processedFrames": 1,
                "currentWindow": None,
            },
        ),
        event_value(
            "frameCompleted",
            {
                "frameIndex": 7,
                "windowIndex": 2,
                "depthPath": "/Predictions/00000007.depth-f16",
                "confidencePath": "Predictions/00000007.confidence-f16",
                "geometryPath": "Predictions/00000007.geometry.json",
                "durationSeconds": 0.25,
            },
        ),
        event_value(
            "windowCompleted",
            {
                "windowIndex": 2,
                "inferenceFrameStart": 41,
                "frameStart": 40,
                "frameEnd": 44,
                "pointChunkPath": "Points/window-00000002.cpc",
                "alignmentTransform": identity,
                "lastProcessedFrameIndex": 44,
                "inlierCount": 99,
                "durationSeconds": 1,
            },
        ),
    ]
    for value in invalid:
        with pytest.raises(msgspec.ValidationError):
            decode_event(value)


def test_canonical_artifact_paths_use_minimum_width_eight_unsigned_decimal() -> None:
    assert artifact_paths(7) == (
        "Predictions/00000007.depth-f16",
        "Predictions/00000007.confidence-f16",
        "Predictions/00000007.geometry.json",
    )
    assert artifact_paths(2**32 - 1)[0] == "Predictions/4294967295.depth-f16"
    assert window_point_path(7) == "Points/window-00000007.cpc"


def test_duplicate_command_ids_are_bounded_and_oldest_is_evicted() -> None:
    tracker = CommandIDTracker(capacity=2)
    first = uuid.UUID(COMMAND_ID)
    second = uuid.UUID("00000000-0000-0000-0000-000000000002")
    third = uuid.UUID("00000000-0000-0000-0000-000000000003")
    tracker.insert(first)
    with pytest.raises(WorkerFault, match="DUPLICATE_COMMAND_ID"):
        tracker.insert(first)
    tracker.insert(second)
    tracker.insert(third)
    tracker.insert(first)
    assert len(tracker) == 2


def test_failure_disposition_matches_the_frozen_transport_table() -> None:
    header = recover_command_header(command_value("pause", {}))
    assert header is not None
    assert (
        classify_failure(WorkerFault("INVALID_MESSAGE_LENGTH", "bad", False), header)
        is FailureDisposition.CLOSE_WITHOUT_RESPONSE
    )
    assert (
        classify_failure(WorkerFault("INVALID_JSON", "bad", False), None)
        is FailureDisposition.ASYNC_ERROR_THEN_CLOSE
    )
    assert (
        classify_failure(WorkerFault("INVALID_COMMAND", "bad", True), header)
        is FailureDisposition.COMMAND_ERROR_THEN_CONTINUE
    )
    assert (
        classify_failure(
            WorkerFault("UNSUPPORTED_PROTOCOL_VERSION", "bad", False), header
        )
        is FailureDisposition.COMMAND_ERROR_THEN_CLOSE
    )


def test_real_unsupported_version_error_selects_flush_then_close() -> None:
    value = command_value("pause", {})
    value["protocolVersion"] = 2
    header = recover_command_header(value)
    assert header is not None

    with pytest.raises(ProtocolValidationError) as caught:
        decode_command(value)

    assert caught.value.code == "UNSUPPORTED_PROTOCOL_VERSION"
    assert (
        classify_failure(caught.value, header)
        is FailureDisposition.COMMAND_ERROR_THEN_CLOSE
    )


@pytest.mark.parametrize("token", ["1e10000", "1" + ("0" * 5_000)])
def test_overflowing_double_is_payload_error_with_recoverable_header(
    token: str,
) -> None:
    value = command_value(
        "enqueueFrame",
        {
            "frameIndex": 1,
            "sourceTimestamp": RawJSONNumber(token),
            "relativePath": "Frames/00000001.jpg",
        },
    )
    header = recover_command_header(value)
    assert header is not None

    with pytest.raises(ProtocolValidationError) as caught:
        decode_command(value)

    assert caught.value.code == "INVALID_COMMAND"
    assert (
        classify_failure(caught.value, header)
        is FailureDisposition.COMMAND_ERROR_THEN_CONTINUE
    )


def test_raw_error_detail_numbers_are_the_only_noncanonical_number_exception() -> None:
    event = decode_event(
        event_value(
            "error",
            {
                "code": "probe",
                "message": "probe",
                "recoverable": True,
                "details": {
                    "raw": RawJSONNumber("1.2300e+04"),
                    "decimal": Decimal("2.500"),
                    "unsigned": 2**64 - 1,
                },
            },
            command_id=None,
        )
    )
    encoded = encode_canonical_json(event)
    assert b'"raw":1.2300e+04' in encoded
    assert b'"decimal":2.500' in encoded
    assert b'"unsigned":18446744073709551615' in encoded
