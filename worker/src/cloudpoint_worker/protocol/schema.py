"""Strict protocol-v1 command, response, and event schemas."""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict
from enum import Enum
from typing import Annotated, Literal

import msgspec

from cloudpoint_worker import PROTOCOL_VERSION
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.protocol.framing import RawJSONNumber

UINT32_MAX = 2**32 - 1
UINT64_MAX = 2**64 - 1

type UInt32 = Annotated[int, msgspec.Meta(ge=0, le=UINT32_MAX)]
type UInt64 = Annotated[int, msgspec.Meta(ge=0, le=UINT64_MAX)]
type CommandType = Literal[
    "hello",
    "configure",
    "beginSession",
    "enqueueFrame",
    "finishInput",
    "pause",
    "resume",
    "cancel",
    "shutdown",
]

COMMAND_TYPES = frozenset(
    {
        "hello",
        "configure",
        "beginSession",
        "enqueueFrame",
        "finishInput",
        "pause",
        "resume",
        "cancel",
        "shutdown",
    }
)


class _StrictStruct(
    msgspec.Struct,
    rename="camel",
    frozen=True,
    forbid_unknown_fields=True,
):
    pass


class EmptyPayload(_StrictStruct):
    pass


class HelloPayload(_StrictStruct):
    client_version: str
    supported_protocol_versions: list[UInt32]


class ConfigurePayload(_StrictStruct):
    scale_frames: UInt32
    window_size: UInt32
    window_overlap: UInt32
    keyframe_interval: UInt32
    camera_refinement_iterations: UInt32
    confidence_threshold: float
    voxel_size: float


class ResumeCheckpoint(_StrictStruct):
    last_committed_frame_index: UInt32
    replay_from_frame_index: UInt32
    next_window_index: UInt32


class BeginSessionPayload(_StrictStruct):
    resume_checkpoint: ResumeCheckpoint | None


class EnqueueFramePayload(_StrictStruct):
    frame_index: UInt32
    source_timestamp: float
    relative_path: str


class _CommandEnvelope(_StrictStruct):
    protocol_version: Literal[1]
    id: uuid.UUID
    project_id: uuid.UUID


class HelloCommand(_CommandEnvelope):
    type: Literal["hello"]
    payload: HelloPayload


class ConfigureCommand(_CommandEnvelope):
    type: Literal["configure"]
    payload: ConfigurePayload


class BeginSessionCommand(_CommandEnvelope):
    type: Literal["beginSession"]
    payload: BeginSessionPayload


class EnqueueFrameCommand(_CommandEnvelope):
    type: Literal["enqueueFrame"]
    payload: EnqueueFramePayload


class FinishInputCommand(_CommandEnvelope):
    type: Literal["finishInput"]
    payload: EmptyPayload


class PauseCommand(_CommandEnvelope):
    type: Literal["pause"]
    payload: EmptyPayload


class ResumeCommand(_CommandEnvelope):
    type: Literal["resume"]
    payload: EmptyPayload


class CancelCommand(_CommandEnvelope):
    type: Literal["cancel"]
    payload: EmptyPayload


class ShutdownCommand(_CommandEnvelope):
    type: Literal["shutdown"]
    payload: EmptyPayload


type Command = (
    HelloCommand
    | ConfigureCommand
    | BeginSessionCommand
    | EnqueueFrameCommand
    | FinishInputCommand
    | PauseCommand
    | ResumeCommand
    | CancelCommand
    | ShutdownCommand
)


class AckPayload(_StrictStruct):
    command: str


class ErrorPayload(_StrictStruct):
    code: str
    message: str
    recoverable: bool
    details: dict[str, object]


class ReadyPayload(_StrictStruct):
    engine_version: str
    model_identifier: str
    model_revision: str
    converted_weights_sha256: str = msgspec.field(name="convertedWeightsSHA256")


class ModelProgressPayload(_StrictStruct):
    phase: Literal["validating", "loading"]
    completed: UInt64
    total: UInt64


class FrameStartedPayload(_StrictStruct):
    frame_index: UInt32
    window_index: UInt32


class FrameCompletedPayload(_StrictStruct):
    frame_index: UInt32
    window_index: UInt32
    depth_path: str
    confidence_path: str
    geometry_path: str
    duration_seconds: float


class WindowCompletedPayload(_StrictStruct):
    window_index: UInt32
    inference_frame_start: UInt32
    frame_start: UInt32
    frame_end: UInt32
    point_chunk_path: str
    alignment_transform: list[float]
    last_processed_frame_index: UInt32
    inlier_count: UInt64
    duration_seconds: float


class SessionCompletedPayload(_StrictStruct):
    processed_frames: UInt64
    window_count: UInt32
    duration_seconds: float


class PausedPayload(_StrictStruct):
    queued_frames: UInt64
    processed_frames: UInt64


class CancelledPayload(_StrictStruct):
    last_completed_window_index: UInt32 | None


class HeartbeatPayload(_StrictStruct):
    busy: bool
    monotonic_seconds: float
    queued_frames: UInt64
    processed_frames: UInt64
    current_window: UInt32 | None


class _EventEnvelope(_StrictStruct):
    protocol_version: Literal[1]
    id: uuid.UUID
    project_id: uuid.UUID


class Ack(_EventEnvelope):
    type: Literal["ack"]
    command_id: uuid.UUID
    payload: AckPayload


class ErrorMessage(_EventEnvelope):
    type: Literal["error"]
    command_id: uuid.UUID | None
    payload: ErrorPayload


class Ready(_EventEnvelope):
    type: Literal["ready"]
    payload: ReadyPayload


class ModelProgress(_EventEnvelope):
    type: Literal["modelProgress"]
    payload: ModelProgressPayload


class FrameStarted(_EventEnvelope):
    type: Literal["frameStarted"]
    payload: FrameStartedPayload


class FrameCompleted(_EventEnvelope):
    type: Literal["frameCompleted"]
    payload: FrameCompletedPayload


class WindowCompleted(_EventEnvelope):
    type: Literal["windowCompleted"]
    payload: WindowCompletedPayload


class SessionCompleted(_EventEnvelope):
    type: Literal["sessionCompleted"]
    payload: SessionCompletedPayload


class Paused(_EventEnvelope):
    type: Literal["paused"]
    payload: PausedPayload


class Cancelled(_EventEnvelope):
    type: Literal["cancelled"]
    payload: CancelledPayload


class WarningMessage(_EventEnvelope):
    type: Literal["warning"]
    payload: ErrorPayload


class Heartbeat(_EventEnvelope):
    type: Literal["heartbeat"]
    payload: HeartbeatPayload


type Event = (
    Ack
    | ErrorMessage
    | Ready
    | ModelProgress
    | FrameStarted
    | FrameCompleted
    | WindowCompleted
    | SessionCompleted
    | Paused
    | Cancelled
    | WarningMessage
    | Heartbeat
)


class CommandHeader(_StrictStruct):
    protocol_version: int
    id: uuid.UUID
    project_id: uuid.UUID
    type: str


class ProtocolValidationError(msgspec.ValidationError):
    """A schema error carrying the stable fault needed by transport policy."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "INVALID_COMMAND",
        recoverable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable

    def worker_fault(self) -> WorkerFault:
        return WorkerFault(self.code, str(self), self.recoverable)


def _invalid(
    message: str,
    *,
    code: str = "INVALID_COMMAND",
    recoverable: bool = True,
) -> ProtocolValidationError:
    return ProtocolValidationError(message, code=code, recoverable=recoverable)


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _invalid(f"Expected object at {path}")
    return value


def _exact_keys(value: dict[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise _invalid(
            f"Invalid fields at {path}: missing={missing}, unknown={unknown}"
        )


def _string(value: object, path: str) -> str:
    if type(value) is not str:
        raise _invalid(f"Expected string at {path}")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise _invalid(f"Expected boolean at {path}")
    return value


def _uint(value: object, maximum: int, path: str) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise _invalid(f"Expected unsigned integer no greater than {maximum} at {path}")
    return value


def _double(
    value: object,
    path: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, RawJSONNumber):
        token_value: object = value.token
    else:
        token_value = value
    if type(value) not in {int, float} and not isinstance(value, RawJSONNumber):
        raise _invalid(f"Expected Double at {path}")
    try:
        result = float(token_value)
    except (OverflowError, ValueError) as error:
        raise _invalid(f"Expected finite Double at {path}") from error
    if not math.isfinite(result):
        raise _invalid(f"Expected finite Double at {path}")
    if nonnegative and result < 0:
        raise _invalid(f"Expected nonnegative Double at {path}")
    if positive and result <= 0:
        raise _invalid(f"Expected positive Double at {path}")
    return result


def _canonical_uuid(value: object, path: str) -> uuid.UUID:
    if type(value) is not str or value != value.lower():
        raise _invalid(f"Expected lowercase canonical UUID at {path}")
    try:
        result = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise _invalid(f"Expected lowercase canonical UUID at {path}") from error
    if str(result) != value:
        raise _invalid(f"Expected lowercase canonical UUID at {path}")
    return result


def _nullable_uuid(value: object, path: str) -> uuid.UUID | None:
    return None if value is None else _canonical_uuid(value, path)


def _safe_relative_path(value: object, path: str) -> str:
    result = _string(value, path)
    if (
        not result
        or result.startswith(("/", "~"))
        or "\\" in result
        or "\0" in result
        or any(component in {"", ".", ".."} for component in result.split("/"))
    ):
        raise _invalid(f"Expected safe package-relative path at {path}")
    return result


def _message_root(
    value: object, *, command_id: bool = False
) -> tuple[dict[str, object], int, uuid.UUID, uuid.UUID, str]:
    root = _object(value, "$")
    keys = {"protocolVersion", "id", "projectId", "type", "payload"}
    if command_id:
        keys.add("commandId")
    _exact_keys(root, keys, "$")

    version = _uint(root["protocolVersion"], UINT32_MAX, "$.protocolVersion")
    if version != PROTOCOL_VERSION:
        raise _invalid(
            f"Unsupported protocol version {version}",
            code="UNSUPPORTED_PROTOCOL_VERSION",
            recoverable=False,
        )
    identifier = _canonical_uuid(root["id"], "$.id")
    project_id = _canonical_uuid(root["projectId"], "$.projectId")
    kind = _string(root["type"], "$.type")
    return root, version, identifier, project_id, kind


def _empty_payload(value: object, path: str = "$.payload") -> EmptyPayload:
    payload = _object(value, path)
    _exact_keys(payload, set(), path)
    return EmptyPayload()


def decode_command(value: object) -> Command:
    """Decode and semantically validate one strict protocol-v1 command."""

    root, version, identifier, project_id, kind = _message_root(value)
    payload = _object(root["payload"], "$.payload")

    if kind == "hello":
        _exact_keys(
            payload, {"clientVersion", "supportedProtocolVersions"}, "$.payload"
        )
        versions_value = payload["supportedProtocolVersions"]
        if not isinstance(versions_value, list):
            raise _invalid("Expected array at $.payload.supportedProtocolVersions")
        versions = [
            _uint(item, UINT32_MAX, f"$.payload.supportedProtocolVersions[{index}]")
            for index, item in enumerate(versions_value)
        ]
        if PROTOCOL_VERSION not in versions:
            raise _invalid("supportedProtocolVersions must include 1")
        return HelloCommand(
            version,
            identifier,
            project_id,
            kind,
            HelloPayload(
                _string(payload["clientVersion"], "$.payload.clientVersion"), versions
            ),
        )

    if kind == "configure":
        expected = {
            "scaleFrames",
            "windowSize",
            "windowOverlap",
            "keyframeInterval",
            "cameraRefinementIterations",
            "confidenceThreshold",
            "voxelSize",
        }
        _exact_keys(payload, expected, "$.payload")
        scale_frames = _uint(
            payload["scaleFrames"], UINT32_MAX, "$.payload.scaleFrames"
        )
        window_size = _uint(payload["windowSize"], UINT32_MAX, "$.payload.windowSize")
        overlap = _uint(payload["windowOverlap"], UINT32_MAX, "$.payload.windowOverlap")
        keyframe_interval = _uint(
            payload["keyframeInterval"], UINT32_MAX, "$.payload.keyframeInterval"
        )
        refinements = _uint(
            payload["cameraRefinementIterations"],
            UINT32_MAX,
            "$.payload.cameraRefinementIterations",
        )
        confidence = _double(
            payload["confidenceThreshold"],
            "$.payload.confidenceThreshold",
            positive=True,
        )
        voxel = _double(payload["voxelSize"], "$.payload.voxelSize", positive=True)
        if not 1 <= window_size <= 1024:
            raise _invalid("windowSize must be in 1...1024")
        if not 1 <= scale_frames <= window_size:
            raise _invalid("scaleFrames must be in 1...windowSize")
        if overlap >= window_size:
            raise _invalid("windowOverlap must be less than windowSize")
        if keyframe_interval == 0 or refinements == 0:
            raise _invalid(
                "keyframeInterval and cameraRefinementIterations must be positive"
            )
        return ConfigureCommand(
            version,
            identifier,
            project_id,
            kind,
            ConfigurePayload(
                scale_frames,
                window_size,
                overlap,
                keyframe_interval,
                refinements,
                confidence,
                voxel,
            ),
        )

    if kind == "beginSession":
        _exact_keys(payload, {"resumeCheckpoint"}, "$.payload")
        checkpoint_value = payload["resumeCheckpoint"]
        checkpoint: ResumeCheckpoint | None
        if checkpoint_value is None:
            checkpoint = None
        else:
            checkpoint_object = _object(checkpoint_value, "$.payload.resumeCheckpoint")
            _exact_keys(
                checkpoint_object,
                {"lastCommittedFrameIndex", "replayFromFrameIndex", "nextWindowIndex"},
                "$.payload.resumeCheckpoint",
            )
            last = _uint(
                checkpoint_object["lastCommittedFrameIndex"],
                UINT32_MAX,
                "$.payload.resumeCheckpoint.lastCommittedFrameIndex",
            )
            replay = _uint(
                checkpoint_object["replayFromFrameIndex"],
                UINT32_MAX,
                "$.payload.resumeCheckpoint.replayFromFrameIndex",
            )
            next_window = _uint(
                checkpoint_object["nextWindowIndex"],
                UINT32_MAX,
                "$.payload.resumeCheckpoint.nextWindowIndex",
            )
            if replay > last:
                raise _invalid(
                    "replayFromFrameIndex must not exceed lastCommittedFrameIndex"
                )
            checkpoint = ResumeCheckpoint(last, replay, next_window)
        return BeginSessionCommand(
            version,
            identifier,
            project_id,
            kind,
            BeginSessionPayload(checkpoint),
        )

    if kind == "enqueueFrame":
        _exact_keys(
            payload, {"frameIndex", "sourceTimestamp", "relativePath"}, "$.payload"
        )
        frame_index = _uint(payload["frameIndex"], UINT32_MAX, "$.payload.frameIndex")
        timestamp = _double(
            payload["sourceTimestamp"], "$.payload.sourceTimestamp", nonnegative=True
        )
        relative_path = _safe_relative_path(
            payload["relativePath"], "$.payload.relativePath"
        )
        return EnqueueFrameCommand(
            version,
            identifier,
            project_id,
            kind,
            EnqueueFramePayload(frame_index, timestamp, relative_path),
        )

    empty_types: dict[
        str,
        type[
            FinishInputCommand
            | PauseCommand
            | ResumeCommand
            | CancelCommand
            | ShutdownCommand
        ],
    ] = {
        "finishInput": FinishInputCommand,
        "pause": PauseCommand,
        "resume": ResumeCommand,
        "cancel": CancelCommand,
        "shutdown": ShutdownCommand,
    }
    if kind in empty_types:
        return empty_types[kind](
            version, identifier, project_id, kind, _empty_payload(payload)
        )
    raise _invalid(f"Unknown command type {kind!r}", code="UNKNOWN_MESSAGE_TYPE")


def _validate_detail_value(value: object, path: str) -> object:
    if (
        value is None
        or type(value) in {bool, str, int, float}
        or isinstance(value, RawJSONNumber)
    ):
        if isinstance(value, float) and not math.isfinite(value):
            raise _invalid(f"Expected finite JSON number at {path}")
        return value
    # Decimal values are accepted without importing Decimal into every schema user.
    if (
        value.__class__.__module__ == "decimal"
        and value.__class__.__name__ == "Decimal"
    ):
        if not value.is_finite():  # type: ignore[attr-defined]
            raise _invalid(f"Expected finite Decimal at {path}")
        return value
    if isinstance(value, list):
        return [
            _validate_detail_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {
            key: _validate_detail_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    raise _invalid(f"Expected JSON value at {path}")


def _error_payload(value: object, path: str = "$.payload") -> ErrorPayload:
    payload = _object(value, path)
    _exact_keys(payload, {"code", "message", "recoverable", "details"}, path)
    details = _object(payload["details"], f"{path}.details")
    return ErrorPayload(
        _string(payload["code"], f"{path}.code"),
        _string(payload["message"], f"{path}.message"),
        _boolean(payload["recoverable"], f"{path}.recoverable"),
        {
            key: _validate_detail_value(item, f"{path}.details.{key}")
            for key, item in details.items()
        },
    )


def artifact_paths(frame_index: int) -> tuple[str, str, str]:
    frame = _uint(frame_index, UINT32_MAX, "frameIndex")
    token = f"{frame:08d}"
    return (
        f"Predictions/{token}.depth-f16",
        f"Predictions/{token}.confidence-f16",
        f"Predictions/{token}.geometry.json",
    )


def window_point_path(window_index: int) -> str:
    window = _uint(window_index, UINT32_MAX, "windowIndex")
    return f"Points/window-{window:08d}.cpc"


def decode_event(value: object) -> Event:
    """Decode and semantically validate one worker response or event."""

    rough = _object(value, "$")
    kind = _string(rough.get("type"), "$.type")
    has_command_id = kind in {"ack", "error"}
    root, version, identifier, project_id, kind = _message_root(
        value, command_id=has_command_id
    )
    payload = _object(root["payload"], "$.payload")

    if kind == "ack":
        _exact_keys(payload, {"command"}, "$.payload")
        command = _string(payload["command"], "$.payload.command")
        if command not in COMMAND_TYPES:
            raise _invalid("ack command must name a protocol-v1 command")
        return Ack(
            version,
            identifier,
            project_id,
            kind,
            _canonical_uuid(root["commandId"], "$.commandId"),
            AckPayload(command),
        )

    if kind == "error":
        return ErrorMessage(
            version,
            identifier,
            project_id,
            kind,
            _nullable_uuid(root["commandId"], "$.commandId"),
            _error_payload(payload),
        )

    if kind == "ready":
        _exact_keys(
            payload,
            {
                "engineVersion",
                "modelIdentifier",
                "modelRevision",
                "convertedWeightsSHA256",
            },
            "$.payload",
        )
        return Ready(
            version,
            identifier,
            project_id,
            kind,
            ReadyPayload(
                _string(payload["engineVersion"], "$.payload.engineVersion"),
                _string(payload["modelIdentifier"], "$.payload.modelIdentifier"),
                _string(payload["modelRevision"], "$.payload.modelRevision"),
                _string(
                    payload["convertedWeightsSHA256"],
                    "$.payload.convertedWeightsSHA256",
                ),
            ),
        )

    if kind == "modelProgress":
        _exact_keys(payload, {"phase", "completed", "total"}, "$.payload")
        phase = _string(payload["phase"], "$.payload.phase")
        if phase not in {"validating", "loading"}:
            raise _invalid("modelProgress phase must be validating or loading")
        completed = _uint(payload["completed"], UINT64_MAX, "$.payload.completed")
        total = _uint(payload["total"], UINT64_MAX, "$.payload.total")
        if completed > total:
            raise _invalid("modelProgress completed must not exceed total")
        return ModelProgress(
            version,
            identifier,
            project_id,
            kind,
            ModelProgressPayload(phase, completed, total),
        )

    if kind == "frameStarted":
        _exact_keys(payload, {"frameIndex", "windowIndex"}, "$.payload")
        return FrameStarted(
            version,
            identifier,
            project_id,
            kind,
            FrameStartedPayload(
                _uint(payload["frameIndex"], UINT32_MAX, "$.payload.frameIndex"),
                _uint(payload["windowIndex"], UINT32_MAX, "$.payload.windowIndex"),
            ),
        )

    if kind == "frameCompleted":
        expected = {
            "frameIndex",
            "windowIndex",
            "depthPath",
            "confidencePath",
            "geometryPath",
            "durationSeconds",
        }
        _exact_keys(payload, expected, "$.payload")
        frame_index = _uint(payload["frameIndex"], UINT32_MAX, "$.payload.frameIndex")
        window_index = _uint(
            payload["windowIndex"], UINT32_MAX, "$.payload.windowIndex"
        )
        depth, confidence, geometry = artifact_paths(frame_index)
        actual_paths = (
            _safe_relative_path(payload["depthPath"], "$.payload.depthPath"),
            _safe_relative_path(payload["confidencePath"], "$.payload.confidencePath"),
            _safe_relative_path(payload["geometryPath"], "$.payload.geometryPath"),
        )
        if actual_paths != (depth, confidence, geometry):
            raise _invalid("frameCompleted paths must be canonical for frameIndex")
        return FrameCompleted(
            version,
            identifier,
            project_id,
            kind,
            FrameCompletedPayload(
                frame_index,
                window_index,
                depth,
                confidence,
                geometry,
                _double(
                    payload["durationSeconds"],
                    "$.payload.durationSeconds",
                    nonnegative=True,
                ),
            ),
        )

    if kind == "windowCompleted":
        expected = {
            "windowIndex",
            "inferenceFrameStart",
            "frameStart",
            "frameEnd",
            "pointChunkPath",
            "alignmentTransform",
            "lastProcessedFrameIndex",
            "inlierCount",
            "durationSeconds",
        }
        _exact_keys(payload, expected, "$.payload")
        window_index = _uint(
            payload["windowIndex"], UINT32_MAX, "$.payload.windowIndex"
        )
        inference_start = _uint(
            payload["inferenceFrameStart"], UINT32_MAX, "$.payload.inferenceFrameStart"
        )
        frame_start = _uint(payload["frameStart"], UINT32_MAX, "$.payload.frameStart")
        frame_end = _uint(payload["frameEnd"], UINT32_MAX, "$.payload.frameEnd")
        last_processed = _uint(
            payload["lastProcessedFrameIndex"],
            UINT32_MAX,
            "$.payload.lastProcessedFrameIndex",
        )
        if not inference_start <= frame_start <= frame_end <= last_processed:
            raise _invalid(
                "windowCompleted bounds must be ordered from inference start "
                "through last processed frame"
            )
        point_path = _safe_relative_path(
            payload["pointChunkPath"], "$.payload.pointChunkPath"
        )
        if point_path != window_point_path(window_index):
            raise _invalid("pointChunkPath must be canonical for windowIndex")
        transform_value = payload["alignmentTransform"]
        if not isinstance(transform_value, list) or len(transform_value) != 16:
            raise _invalid("alignmentTransform must contain exactly 16 Doubles")
        transform = [
            _double(item, f"$.payload.alignmentTransform[{index}]")
            for index, item in enumerate(transform_value)
        ]
        return WindowCompleted(
            version,
            identifier,
            project_id,
            kind,
            WindowCompletedPayload(
                window_index,
                inference_start,
                frame_start,
                frame_end,
                point_path,
                transform,
                last_processed,
                _uint(payload["inlierCount"], UINT64_MAX, "$.payload.inlierCount"),
                _double(
                    payload["durationSeconds"],
                    "$.payload.durationSeconds",
                    nonnegative=True,
                ),
            ),
        )

    if kind == "sessionCompleted":
        _exact_keys(
            payload, {"processedFrames", "windowCount", "durationSeconds"}, "$.payload"
        )
        return SessionCompleted(
            version,
            identifier,
            project_id,
            kind,
            SessionCompletedPayload(
                _uint(
                    payload["processedFrames"], UINT64_MAX, "$.payload.processedFrames"
                ),
                _uint(payload["windowCount"], UINT32_MAX, "$.payload.windowCount"),
                _double(
                    payload["durationSeconds"],
                    "$.payload.durationSeconds",
                    nonnegative=True,
                ),
            ),
        )

    if kind == "paused":
        _exact_keys(payload, {"queuedFrames", "processedFrames"}, "$.payload")
        queued = _uint(payload["queuedFrames"], UINT64_MAX, "$.payload.queuedFrames")
        processed = _uint(
            payload["processedFrames"], UINT64_MAX, "$.payload.processedFrames"
        )
        if processed > queued:
            raise _invalid("paused processedFrames must not exceed queuedFrames")
        return Paused(
            version, identifier, project_id, kind, PausedPayload(queued, processed)
        )

    if kind == "cancelled":
        _exact_keys(payload, {"lastCompletedWindowIndex"}, "$.payload")
        value = payload["lastCompletedWindowIndex"]
        last_window = (
            None
            if value is None
            else _uint(value, UINT32_MAX, "$.payload.lastCompletedWindowIndex")
        )
        return Cancelled(
            version, identifier, project_id, kind, CancelledPayload(last_window)
        )

    if kind == "warning":
        return WarningMessage(
            version, identifier, project_id, kind, _error_payload(payload)
        )

    if kind == "heartbeat":
        expected = {
            "busy",
            "monotonicSeconds",
            "queuedFrames",
            "processedFrames",
            "currentWindow",
        }
        _exact_keys(payload, expected, "$.payload")
        queued = _uint(payload["queuedFrames"], UINT64_MAX, "$.payload.queuedFrames")
        processed = _uint(
            payload["processedFrames"], UINT64_MAX, "$.payload.processedFrames"
        )
        if processed > queued:
            raise _invalid("heartbeat processedFrames must not exceed queuedFrames")
        current_value = payload["currentWindow"]
        current = (
            None
            if current_value is None
            else _uint(current_value, UINT32_MAX, "$.payload.currentWindow")
        )
        return Heartbeat(
            version,
            identifier,
            project_id,
            kind,
            HeartbeatPayload(
                _boolean(payload["busy"], "$.payload.busy"),
                _double(
                    payload["monotonicSeconds"],
                    "$.payload.monotonicSeconds",
                    nonnegative=True,
                ),
                queued,
                processed,
                current,
            ),
        )

    raise _invalid(f"Unknown event type {kind!r}")


def ack(command: Command, *, id: uuid.UUID | None = None) -> Ack:
    """Build the sole immediate success response for a decoded command."""

    return Ack(
        PROTOCOL_VERSION,
        id or uuid.uuid4(),
        command.project_id,
        "ack",
        command.id,
        AckPayload(command.type),
    )


def command_error(
    command: Command,
    fault: WorkerFault,
    *,
    id: uuid.UUID | None = None,
) -> ErrorMessage:
    """Build a command-owned structured error response."""

    return ErrorMessage(
        PROTOCOL_VERSION,
        id or uuid.uuid4(),
        command.project_id,
        "error",
        command.id,
        ErrorPayload(fault.code, fault.message, fault.recoverable, dict(fault.details)),
    )


class CommandIDTracker:
    """A bounded insertion-ordered set for recently decoded command UUIDs."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._ids: OrderedDict[uuid.UUID, None] = OrderedDict()

    def insert(self, identifier: uuid.UUID) -> None:
        if identifier in self._ids:
            raise WorkerFault(
                "DUPLICATE_COMMAND_ID",
                f"command {identifier} was already received",
                True,
                {"commandId": str(identifier)},
            )
        self._ids[identifier] = None
        if len(self._ids) > self.capacity:
            self._ids.popitem(last=False)

    def __len__(self) -> int:
        return len(self._ids)


def recover_command_header(value: object) -> CommandHeader | None:
    """Recover the exact response-routing fields without accepting the payload."""

    try:
        root = _object(value, "$")
        version_value = root.get("protocolVersion")
        if type(version_value) is not int:
            return None
        kind = _string(root.get("type"), "$.type")
        if not kind:
            return None
        return CommandHeader(
            version_value,
            _canonical_uuid(root.get("id"), "$.id"),
            _canonical_uuid(root.get("projectId"), "$.projectId"),
            kind,
        )
    except msgspec.ValidationError:
        return None


class FailureDisposition(Enum):
    CLOSE_WITHOUT_RESPONSE = "closeWithoutResponse"
    ASYNC_ERROR_THEN_CLOSE = "asynchronousErrorThenClose"
    COMMAND_ERROR_THEN_CONTINUE = "commandErrorThenContinue"
    COMMAND_ERROR_THEN_CLOSE = "commandErrorThenClose"


def classify_failure(
    fault: WorkerFault | ProtocolValidationError,
    recoverable_header: CommandHeader | None,
) -> FailureDisposition:
    """Apply the frozen malformed-input response and transport policy."""

    if fault.code in {
        "INVALID_MESSAGE_LENGTH",
        "MESSAGE_TOO_LARGE",
        "TRUNCATED_MESSAGE",
    }:
        return FailureDisposition.CLOSE_WITHOUT_RESPONSE
    if fault.code in {"INVALID_JSON", "MALFORMED_ENVELOPE"}:
        return FailureDisposition.ASYNC_ERROR_THEN_CLOSE
    if fault.code == "UNSUPPORTED_PROTOCOL_VERSION":
        if recoverable_header is not None:
            return FailureDisposition.COMMAND_ERROR_THEN_CLOSE
        return FailureDisposition.ASYNC_ERROR_THEN_CLOSE
    if recoverable_header is not None:
        return FailureDisposition.COMMAND_ERROR_THEN_CONTINUE
    return FailureDisposition.ASYNC_ERROR_THEN_CLOSE


__all__ = [
    "PROTOCOL_VERSION",
    "Ack",
    "AckPayload",
    "BeginSessionCommand",
    "BeginSessionPayload",
    "CancelCommand",
    "Cancelled",
    "CancelledPayload",
    "Command",
    "CommandHeader",
    "CommandIDTracker",
    "ConfigureCommand",
    "ConfigurePayload",
    "EmptyPayload",
    "EnqueueFrameCommand",
    "EnqueueFramePayload",
    "ErrorMessage",
    "ErrorPayload",
    "Event",
    "FailureDisposition",
    "FinishInputCommand",
    "FrameCompleted",
    "FrameCompletedPayload",
    "FrameStarted",
    "FrameStartedPayload",
    "Heartbeat",
    "HeartbeatPayload",
    "HelloCommand",
    "HelloPayload",
    "ModelProgress",
    "ModelProgressPayload",
    "PauseCommand",
    "Paused",
    "PausedPayload",
    "ProtocolValidationError",
    "Ready",
    "ReadyPayload",
    "ResumeCheckpoint",
    "ResumeCommand",
    "SessionCompleted",
    "SessionCompletedPayload",
    "ShutdownCommand",
    "WarningMessage",
    "WindowCompleted",
    "WindowCompletedPayload",
    "ack",
    "artifact_paths",
    "classify_failure",
    "command_error",
    "decode_command",
    "decode_event",
    "recover_command_header",
    "window_point_path",
]
