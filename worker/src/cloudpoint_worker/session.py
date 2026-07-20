"""Disk-backed, direct-mode reconstruction session supervision.

Version 1 deliberately ships one honest inference mode: a complete sequence of at
most 32 frames is buffered as package-relative descriptors and reconstructed when
input closes. Longer jobs fail with a recoverable capability error until the
overlapping-window alignment path is available; they are never reported complete.
"""

from __future__ import annotations

import json
import math
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import numpy as np

from cloudpoint_worker import ENGINE_VERSION, PROTOCOL_VERSION
from cloudpoint_worker.cpc import CPCVertex, reduce_vertices, write_cpc
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.geometry import filter_and_reduce_points
from cloudpoint_worker.model.lingbot import FrameBatchPrediction
from cloudpoint_worker.model_prep.provenance import MODEL_REPO, MODEL_REVISION
from cloudpoint_worker.outputs import write_frame_outputs
from cloudpoint_worker.preprocess import (
    ImageBoundsError,
    PreprocessedFrame,
    preprocess_image,
)
from cloudpoint_worker.protocol.schema import (
    BeginSessionPayload,
    Cancelled,
    CancelledPayload,
    ConfigurePayload,
    Event,
    FrameCompleted,
    FrameCompletedPayload,
    FrameStarted,
    FrameStartedPayload,
    Paused,
    PausedPayload,
    ResumeCheckpoint,
    SessionCompleted,
    SessionCompletedPayload,
    WindowCompleted,
    WindowCompletedPayload,
    artifact_paths,
    window_point_path,
)

_UINT32_MAX = 2**32 - 1
_DIRECT_FRAME_LIMIT = 32
_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_CANONICAL_PREDICTION = re.compile(
    r"^[0-9]{8,10}\.(?:depth-f16|confidence-f16|geometry\.json)$"
)
_CANONICAL_POINTS = re.compile(r"^window-[0-9]{8,10}\.cpc$")
_PARTIAL = re.compile(
    r"^\.(?:[0-9]{8,10}\.(?:depth-f16|confidence-f16|geometry\.json)"
    r"|window-[0-9]{8,10}\.cpc)\.[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.partial$"
)


class ReconstructionModel(Protocol):
    def infer_direct(
        self, images: mx.array, scale_frames: int = 8
    ) -> FrameBatchPrediction: ...


type EventSink = Callable[[Event], None]


class SessionState(StrEnum):
    READY = "ready"
    PROCESSING = "processing"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PersistedFrame:
    index: int
    source_timestamp: float
    relative_path: str
    replay: bool = False


@dataclass(frozen=True, slots=True)
class _ProjectSnapshot:
    project_id: uuid.UUID
    manifest: dict[str, object]
    frames: dict[int, PersistedFrame]
    referenced_artifacts: frozenset[str]
    completed_windows: tuple[dict[str, object], ...]


def _fault(
    code: str,
    message: str,
    recoverable: bool = False,
    **details: object,
) -> WorkerFault:
    return WorkerFault(code, message, recoverable, dict(details))


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or (
        not value
        or value.startswith(("/", "~"))
        or "\\" in value
        or "\0" in value
        or any(component in {"", ".", ".."} for component in value.split("/"))
    ):
        raise _fault("PATH_OUTSIDE_PROJECT", "artifact path is not package-relative")
    return value


def _canonical_uuid(
    value: object, *, code: str = "PROJECT_INVALID_MANIFEST"
) -> uuid.UUID:
    if not isinstance(value, str) or value != value.lower():
        raise _fault(code, "project ID is not a lowercase canonical UUID")
    try:
        result = uuid.UUID(value)
    except ValueError as error:
        raise _fault(code, "project ID is not a lowercase canonical UUID") from error
    if str(result) != value:
        raise _fault(code, "project ID is not a lowercase canonical UUID")
    return result


def _read_manifest(project_root: Path) -> _ProjectSnapshot:
    manifest_path = project_root / "Manifest.json"
    try:
        info = manifest_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise OSError("manifest is not a regular file")
        if info.st_size > _MANIFEST_MAX_BYTES:
            raise OSError("manifest exceeds the worker bound")

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate key {key}")
                result[key] = value
            return result

        decoded = json.loads(
            manifest_path.read_text("utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise _fault(
            "PROJECT_INVALID_MANIFEST", "Manifest.json is unavailable or invalid"
        ) from error
    if not isinstance(decoded, dict) or decoded.get("formatVersion") != 2:
        raise _fault(
            "PROJECT_UNSUPPORTED_FORMAT",
            "CloudPoint project format version 2 is required",
        )
    manifest = cast(dict[str, object], decoded)
    project_id = _canonical_uuid(manifest.get("projectID"))

    frames_value = manifest.get("frames")
    windows_value = manifest.get("completedWindows")
    if not isinstance(frames_value, list) or not isinstance(windows_value, list):
        raise _fault(
            "PROJECT_INVALID_MANIFEST",
            "manifest frames and completedWindows must be arrays",
        )
    frames: dict[int, PersistedFrame] = {}
    previous_index = -1
    for raw in frames_value:
        if not isinstance(raw, dict):
            raise _fault("PROJECT_INVALID_MANIFEST", "manifest frame is not an object")
        index = raw.get("index")
        timestamp = raw.get("sourceTimestamp")
        if (
            type(index) is not int
            or not 0 <= index <= _UINT32_MAX
            or index <= previous_index
            or type(timestamp) not in {int, float}
            or not math.isfinite(float(timestamp))
            or float(timestamp) < 0
        ):
            raise _fault("PROJECT_INVALID_MANIFEST", "manifest frame is invalid")
        relative = _safe_relative(raw.get("relativePath"))
        frames[index] = PersistedFrame(index, float(timestamp), relative)
        previous_index = index

    completed: list[dict[str, object]] = []
    referenced: set[str] = set()
    previous_window = -1
    for raw_window in windows_value:
        if not isinstance(raw_window, dict):
            raise _fault(
                "PROJECT_INVALID_MANIFEST", "completed window is not an object"
            )
        window = cast(dict[str, object], raw_window)
        index = window.get("index")
        frame_start = window.get("frameStart")
        frame_end = window.get("frameEnd")
        if (
            type(index) is not int
            or not 0 <= index <= _UINT32_MAX
            or index <= previous_window
            or type(frame_start) is not int
            or type(frame_end) is not int
            or not 0 <= frame_start <= frame_end <= _UINT32_MAX
        ):
            raise _fault(
                "PROJECT_INVALID_MANIFEST", "completed window bounds are invalid"
            )
        point_path = _safe_relative(window.get("pointChunkRelativePath"))
        if point_path != window_point_path(index):
            raise _fault(
                "PROJECT_INVALID_MANIFEST", "completed CPC path is not canonical"
            )
        referenced.add(point_path)
        artifacts = window.get("frameArtifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise _fault(
                "PROJECT_INVALID_MANIFEST", "completed window has no artifacts"
            )
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise _fault("PROJECT_INVALID_MANIFEST", "frame artifact is invalid")
            frame_index = artifact.get("frameIndex")
            if type(frame_index) is not int or frame_index not in frames:
                raise _fault(
                    "PROJECT_INVALID_MANIFEST", "frame artifact index is invalid"
                )
            expected = artifact_paths(frame_index)
            actual = (
                _safe_relative(artifact.get("depthRelativePath")),
                _safe_relative(artifact.get("confidenceRelativePath")),
                _safe_relative(artifact.get("geometryRelativePath")),
            )
            if actual != expected:
                raise _fault(
                    "PROJECT_INVALID_MANIFEST", "frame artifact paths are not canonical"
                )
            referenced.update(actual)
        completed.append(window)
        previous_window = index
    return _ProjectSnapshot(
        project_id,
        manifest,
        frames,
        frozenset(referenced),
        tuple(completed),
    )


def validate_project_root(project_root: Path) -> _ProjectSnapshot:
    """Validate one existing project package without following symlinks."""

    if not project_root.is_absolute() or project_root.suffix != ".cloudpoint":
        raise _fault(
            "PROJECT_INVALID_PATH",
            "project must be an absolute existing .cloudpoint package",
            True,
        )
    try:
        info = project_root.lstat()
    except OSError as error:
        raise _fault(
            "PROJECT_INVALID_PATH", "project package does not exist", True
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _fault(
            "PROJECT_INVALID_PATH", "project package must be a real directory", True
        )
    for name in ("Frames", "Predictions", "Points", "Logs"):
        path = project_root / name
        try:
            child = path.lstat()
        except OSError as error:
            raise _fault(
                "PROJECT_INVALID_PATH", f"project directory {name} is missing", True
            ) from error
        if stat.S_ISLNK(child.st_mode) or not stat.S_ISDIR(child.st_mode):
            raise _fault(
                "PROJECT_INVALID_PATH", f"project directory {name} is unsafe", True
            )
    return _read_manifest(project_root)


def _real_project_file(project_root: Path, relative: str) -> Path:
    relative = _safe_relative(relative)
    components = relative.split("/")
    candidate = project_root
    try:
        for ordinal, component in enumerate(components):
            candidate = candidate / component
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise OSError("symlink component")
            if ordinal < len(components) - 1 and not stat.S_ISDIR(info.st_mode):
                raise OSError("non-directory component")
        if not stat.S_ISREG(info.st_mode):
            raise OSError("not a regular file")
    except OSError as error:
        raise _fault(
            "PATH_OUTSIDE_PROJECT",
            "frame path must resolve to a regular package file without symlinks",
            True,
            relativePath=relative,
        ) from error
    return candidate


def _validate_frame_path(project_root: Path, frame: PersistedFrame) -> Path:
    components = frame.relative_path.split("/")
    expected_stem = f"{frame.index:08d}"
    if (
        len(components) != 2
        or components[0] != "Frames"
        or Path(components[1]).stem != expected_stem
        or Path(components[1]).suffix.lower() not in {".jpg", ".jpeg", ".png"}
    ):
        raise _fault(
            "PATH_OUTSIDE_PROJECT",
            "frame path must be canonical beneath Frames",
            True,
            relativePath=frame.relative_path,
        )
    return _real_project_file(project_root, frame.relative_path)


def _remove_regular_file(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        path.unlink()


def _cleanup_orphans(project_root: Path, referenced: frozenset[str]) -> None:
    for directory_name, canonical_pattern in (
        ("Predictions", _CANONICAL_PREDICTION),
        ("Points", _CANONICAL_POINTS),
    ):
        directory = project_root / directory_name
        for path in directory.iterdir():
            relative = f"{directory_name}/{path.name}"
            if _PARTIAL.fullmatch(path.name) or (
                canonical_pattern.fullmatch(path.name) and relative not in referenced
            ):
                _remove_regular_file(path)


class SessionRunner:
    """Own one finite, recoverable reconstruction invocation."""

    def __init__(
        self,
        project_root: Path,
        model: ReconstructionModel,
        event_sink: EventSink,
        resume_checkpoint: ResumeCheckpoint | None = None,
        *,
        project_id: uuid.UUID | None = None,
        cleanup_orphans: bool = True,
    ) -> None:
        self.project_root = project_root
        self.model = model
        self._event_sink = event_sink
        self._requested_project_id = project_id
        self._initial_checkpoint = resume_checkpoint
        self._cleanup_orphans_enabled = cleanup_orphans
        self._snapshot: _ProjectSnapshot | None = None
        self._configuration: ConfigurePayload | None = None
        self._checkpoint: ResumeCheckpoint | None = None
        self._frames: list[PersistedFrame] = []
        self._input_finished = False
        self._begun = False
        self._processing_active = False
        self._pause_requested = False
        self._pause_emitted = False
        self._cancel_requested = False
        self._cancelled_emitted = False
        self._created_outputs: list[Path] = []
        self._condition = threading.Condition()
        self.state = SessionState.READY
        self.queued_frames = 0
        self.processed_frames = 0
        self.window_count = 0
        self.current_window: int | None = None
        self.last_completed_window_index: int | None = None

    @property
    def project_id(self) -> uuid.UUID:
        if self._snapshot is None:
            raise _fault("SESSION_NOT_BEGUN", "session has not validated its project")
        return self._snapshot.project_id

    def configure(self, configuration: ConfigurePayload) -> None:
        with self._condition:
            if self._begun or self._processing_active or self._input_finished:
                raise _fault("INVALID_STATE", "configuration is already frozen", True)
            self._configuration = configuration

    def begin(self, payload: BeginSessionPayload | None = None) -> None:
        with self._condition:
            if self._begun:
                raise _fault("INVALID_STATE", "session has already begun", True)
            if self._configuration is None:
                raise _fault(
                    "INVALID_STATE", "configure must precede beginSession", True
                )
            snapshot = validate_project_root(self.project_root)
            if (
                self._requested_project_id is not None
                and snapshot.project_id != self._requested_project_id
            ):
                raise _fault(
                    "PROJECT_ID_MISMATCH", "command project ID is incorrect", True
                )
            checkpoint = (
                payload.resume_checkpoint
                if payload is not None
                else self._initial_checkpoint
            )
            self._validate_checkpoint(snapshot, checkpoint)
            if self._cleanup_orphans_enabled:
                _cleanup_orphans(self.project_root, snapshot.referenced_artifacts)
            self._snapshot = snapshot
            self._checkpoint = checkpoint
            self.last_completed_window_index = (
                checkpoint.next_window_index - 1 if checkpoint is not None else None
            )
            self._begun = True

    def _validate_checkpoint(
        self, snapshot: _ProjectSnapshot, checkpoint: ResumeCheckpoint | None
    ) -> None:
        if checkpoint is None:
            if snapshot.completed_windows:
                raise _fault(
                    "INVALID_RESUME_CHECKPOINT",
                    "completed project state requires a resume checkpoint",
                    True,
                )
            return
        if not snapshot.completed_windows:
            raise _fault(
                "INVALID_RESUME_CHECKPOINT",
                "resume checkpoint has no committed manifest window",
                True,
            )
        final = snapshot.completed_windows[-1]
        if (
            final.get("index") != checkpoint.next_window_index - 1
            or final.get("frameEnd") != checkpoint.last_committed_frame_index
            or checkpoint.replay_from_frame_index not in snapshot.frames
            or checkpoint.last_committed_frame_index not in snapshot.frames
        ):
            raise _fault(
                "INVALID_RESUME_CHECKPOINT",
                "resume checkpoint does not match committed manifest state",
                True,
            )
        artifact_indices = {
            artifact.get("frameIndex")
            for window in snapshot.completed_windows
            for artifact in cast(list[dict[str, object]], window["frameArtifacts"])
        }
        replay_indices = [
            index
            for index in snapshot.frames
            if checkpoint.replay_from_frame_index
            <= index
            <= checkpoint.last_committed_frame_index
        ]
        if not replay_indices or any(
            index not in artifact_indices for index in replay_indices
        ):
            raise _fault(
                "MISSING_REPLAY_ARTIFACTS",
                "resume context is not fully committed in the manifest",
                True,
            )
        for relative in snapshot.referenced_artifacts:
            _real_project_file(self.project_root, relative)

    def enqueue(self, frame: PersistedFrame) -> None:
        with self._condition:
            if not self._begun or self._input_finished or self._cancel_requested:
                raise _fault("INVALID_STATE", "session is not accepting frames", True)
            if (
                type(frame.index) is not int
                or not 0 <= frame.index <= _UINT32_MAX
                or not math.isfinite(float(frame.source_timestamp))
                or frame.source_timestamp < 0
            ):
                raise _fault("INVALID_FRAME", "frame descriptor is invalid", True)
            _validate_frame_path(self.project_root, frame)
            if self._frames and frame.index <= self._frames[-1].index:
                raise _fault(
                    "FRAME_ORDER_VIOLATION", "frame indices must increase", True
                )

            replay = False
            checkpoint = self._checkpoint
            if (
                checkpoint is not None
                and frame.index <= checkpoint.last_committed_frame_index
            ):
                if (
                    not self._frames
                    and frame.index != checkpoint.replay_from_frame_index
                ):
                    raise _fault(
                        "REPLAY_ORDER_VIOLATION",
                        "replay must begin at the exact checkpoint frame",
                        True,
                    )
                committed = cast(_ProjectSnapshot, self._snapshot).frames.get(
                    frame.index
                )
                if committed is None or (
                    committed.relative_path != frame.relative_path
                    or committed.source_timestamp != frame.source_timestamp
                ):
                    raise _fault(
                        "MISSING_REPLAY_ARTIFACTS",
                        "replay descriptor differs from committed manifest state",
                        True,
                    )
                replay = True
            elif checkpoint is not None:
                replay_seen = [
                    candidate.index for candidate in self._frames if candidate.replay
                ]
                if (
                    not replay_seen
                    or replay_seen[-1] != checkpoint.last_committed_frame_index
                ):
                    raise _fault(
                        "REPLAY_ORDER_VIOLATION",
                        "new output cannot begin before the committed replay boundary",
                        True,
                    )
            admitted = PersistedFrame(
                frame.index, frame.source_timestamp, frame.relative_path, replay
            )
            self._frames.append(admitted)
            if not replay:
                self.queued_frames += 1

    def finish_input(self) -> None:
        with self._condition:
            if not self._begun or self._input_finished or self._cancel_requested:
                raise _fault("INVALID_STATE", "input cannot be finished now", True)
            if self._checkpoint is not None:
                replay_indices = [frame.index for frame in self._frames if frame.replay]
                if (
                    not replay_indices
                    or replay_indices[0] != self._checkpoint.replay_from_frame_index
                    or replay_indices[-1] != self._checkpoint.last_committed_frame_index
                ):
                    raise _fault(
                        "REPLAY_ORDER_VIOLATION",
                        "finishInput requires the complete replay boundary",
                        True,
                    )
            self._input_finished = True

    def pause(self, *, emit_event: bool = True) -> None:
        with self._condition:
            if not self._begun or self.state in {
                SessionState.COMPLETED,
                SessionState.CANCELLED,
                SessionState.FAILED,
            }:
                raise _fault("INVALID_STATE", "session cannot be paused", True)
            self._pause_requested = True
            if not self._processing_active:
                self.state = SessionState.PAUSED
                if emit_event:
                    self._emit_paused_locked()

    def publish_paused_if_quiescent(self) -> None:
        """Publish the deferred paused event after its command ACK is flushed."""

        with self._condition:
            if self._pause_requested and not self._processing_active:
                self._emit_paused_locked()

    def resume(self) -> None:
        with self._condition:
            if not self._pause_requested and self.state != SessionState.PAUSED:
                raise _fault("INVALID_STATE", "session is not paused", True)
            self._pause_requested = False
            self._pause_emitted = False
            self.state = (
                SessionState.PROCESSING
                if self._processing_active
                else SessionState.READY
            )
            self._condition.notify_all()

    def cancel(self, *, emit_event: bool = True) -> None:
        emit_now = False
        with self._condition:
            if self.state in {SessionState.COMPLETED, SessionState.CANCELLED}:
                return
            self._cancel_requested = True
            self._pause_requested = False
            self._condition.notify_all()
            emit_now = not self._processing_active
            if emit_now:
                self.state = SessionState.CANCELLED
        if emit_now and emit_event:
            self._cleanup_uncommitted_outputs()
            self._emit_cancelled()

    def publish_cancelled_if_quiescent(self) -> None:
        """Publish deferred idle cancellation after its command ACK."""

        with self._condition:
            should_emit = (
                self.state == SessionState.CANCELLED and not self._processing_active
            )
        if should_emit:
            self._cleanup_uncommitted_outputs()
            self._emit_cancelled()

    def _emit(self, event: Event) -> None:
        self._event_sink(event)

    def _emit_paused_locked(self) -> None:
        if self._pause_emitted:
            return
        self._pause_emitted = True
        self._emit(
            Paused(
                PROTOCOL_VERSION,
                uuid.uuid4(),
                self.project_id,
                "paused",
                PausedPayload(self.queued_frames, self.processed_frames),
            )
        )

    def _emit_cancelled(self) -> None:
        with self._condition:
            if self._cancelled_emitted:
                return
            self._cancelled_emitted = True
        self._emit(
            Cancelled(
                PROTOCOL_VERSION,
                uuid.uuid4(),
                self.project_id,
                "cancelled",
                CancelledPayload(self.last_completed_window_index),
            )
        )

    def _cooperative_boundary(self) -> None:
        with self._condition:
            while self._pause_requested and not self._cancel_requested:
                self.state = SessionState.PAUSED
                self._emit_paused_locked()
                self._condition.wait()
            if self._cancel_requested:
                raise _fault("CANCELLED", "session was cancelled", True)
            if self._processing_active:
                self.state = SessionState.PROCESSING

    def _cleanup_uncommitted_outputs(self) -> None:
        for path in reversed(self._created_outputs):
            _remove_regular_file(path)
        self._created_outputs.clear()
        for directory in (
            self.project_root / "Predictions",
            self.project_root / "Points",
        ):
            for path in directory.iterdir():
                if _PARTIAL.fullmatch(path.name):
                    _remove_regular_file(path)

    def process(self) -> None:
        """Run one complete direct reconstruction on the calling thread."""

        started = time.monotonic()
        with self._condition:
            if not self._begun or not self._input_finished or self._processing_active:
                raise _fault(
                    "INVALID_STATE", "finishInput must precede processing", True
                )
            if self.state in {SessionState.CANCELLED, SessionState.COMPLETED}:
                raise _fault("INVALID_STATE", "session is already terminal", True)
            configuration = cast(ConfigurePayload, self._configuration)
            maximum = min(_DIRECT_FRAME_LIMIT, configuration.window_size)
            if len(self._frames) > maximum:
                self.state = SessionState.FAILED
                raise _fault(
                    "WINDOWING_UNAVAILABLE",
                    "direct reconstruction supports at most "
                    f"{maximum} inference frames",
                    True,
                    frameCount=len(self._frames),
                    maximum=maximum,
                )
            self._processing_active = True
            self.state = SessionState.PROCESSING
            window_index = (
                self._checkpoint.next_window_index
                if self._checkpoint is not None
                else 0
            )
            self.current_window = window_index

        try:
            unique_frames = [frame for frame in self._frames if not frame.replay]
            if not unique_frames:
                self.state = SessionState.COMPLETED
                self._emit(
                    SessionCompleted(
                        PROTOCOL_VERSION,
                        uuid.uuid4(),
                        self.project_id,
                        "sessionCompleted",
                        SessionCompletedPayload(0, 0, time.monotonic() - started),
                    )
                )
                return

            prepared: list[PreprocessedFrame] = []
            for frame in self._frames:
                self._cooperative_boundary()
                try:
                    prepared.append(
                        preprocess_image(_validate_frame_path(self.project_root, frame))
                    )
                except ImageBoundsError as error:
                    raise _fault(
                        error.code, "source frame exceeds supported image bounds", True
                    ) from error
                except Exception as error:
                    raise _fault(
                        "INVALID_FRAME_IMAGE",
                        "source frame could not be decoded",
                        True,
                        frameIndex=frame.index,
                    ) from error
            model_sizes = {frame.model_size for frame in prepared}
            if len(model_sizes) != 1:
                raise _fault(
                    "FRAME_SIZE_MISMATCH",
                    "all frames in one direct reconstruction must share "
                    "model dimensions",
                    True,
                )
            images = mx.array(np.stack([frame.rgb for frame in prepared]))
            prediction_batch = self.model.infer_direct(
                images, scale_frames=min(configuration.scale_frames, len(prepared))
            )
            predictions = tuple(prediction_batch.frames)
            if len(predictions) != len(prepared):
                raise _fault(
                    "INVALID_MODEL_OUTPUT", "model prediction count differs from input"
                )
            self._cooperative_boundary()

            all_vertices: list[CPCVertex] = []
            window_started = time.monotonic()
            for ordinal, (frame, preprocessed, prediction) in enumerate(
                zip(self._frames, prepared, predictions, strict=True)
            ):
                if frame.replay:
                    continue
                self._cooperative_boundary()
                frame_started = time.monotonic()
                self._emit(
                    FrameStarted(
                        PROTOCOL_VERSION,
                        uuid.uuid4(),
                        self.project_id,
                        "frameStarted",
                        FrameStartedPayload(frame.index, window_index),
                    )
                )
                expected_artifacts = artifact_paths(frame.index)
                self._created_outputs.extend(
                    self.project_root / relative for relative in expected_artifacts
                )
                artifacts = write_frame_outputs(
                    self.project_root,
                    frame,
                    prediction,
                    preprocessed,
                    confidence_floor=configuration.confidence_threshold,
                    engine_version=ENGINE_VERSION,
                    model_identifier=MODEL_REPO,
                    model_revision=MODEL_REVISION,
                )
                flags = 0
                if ordinal % configuration.keyframe_interval == 0:
                    flags |= 1
                if frame.index == unique_frames[0].index:
                    flags |= 2
                vertices = filter_and_reduce_points(
                    np.asarray(prediction.depth, dtype=np.float32),
                    np.asarray(prediction.confidence, dtype=np.float32),
                    preprocessed.rgb,
                    np.asarray(prediction.intrinsics, dtype=np.float32),
                    np.asarray(prediction.camera_to_world, dtype=np.float32),
                    frame.index,
                    configuration.confidence_threshold,
                    configuration.voxel_size,
                    flags,
                )
                all_vertices.extend(vertices)
                self.processed_frames += 1
                self._emit(
                    FrameCompleted(
                        PROTOCOL_VERSION,
                        uuid.uuid4(),
                        self.project_id,
                        "frameCompleted",
                        FrameCompletedPayload(
                            frame.index,
                            window_index,
                            artifacts.depth_path,
                            artifacts.confidence_path,
                            artifacts.geometry_path,
                            time.monotonic() - frame_started,
                        ),
                    )
                )

            self._cooperative_boundary()
            vertices = reduce_vertices(
                all_vertices, voxel_size=configuration.voxel_size
            )
            if not vertices:
                raise _fault(
                    "EMPTY_POINT_CLOUD",
                    "no points survived the configured confidence threshold",
                    True,
                )
            self.state = SessionState.FINALIZING
            point_relative = window_point_path(window_index)
            descriptor = write_cpc(
                self.project_root / point_relative,
                unique_frames[0].index,
                unique_frames[-1].index,
                vertices,
            )
            self._created_outputs.append(self.project_root / point_relative)
            self.window_count = 1
            self.last_completed_window_index = window_index
            self._emit(
                WindowCompleted(
                    PROTOCOL_VERSION,
                    uuid.uuid4(),
                    self.project_id,
                    "windowCompleted",
                    WindowCompletedPayload(
                        window_index,
                        self._frames[0].index,
                        descriptor.frame_start,
                        descriptor.frame_end,
                        descriptor.relative_path,
                        [
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                        descriptor.frame_end,
                        descriptor.point_count,
                        time.monotonic() - window_started,
                    ),
                )
            )
            # A windowCompleted event transfers ownership to native's pending-window
            # transaction. Cancellation must not remove those artifacts afterward.
            self._created_outputs.clear()
            self.current_window = None
            self.state = SessionState.COMPLETED
            self._emit(
                SessionCompleted(
                    PROTOCOL_VERSION,
                    uuid.uuid4(),
                    self.project_id,
                    "sessionCompleted",
                    SessionCompletedPayload(
                        self.processed_frames,
                        self.window_count,
                        time.monotonic() - started,
                    ),
                )
            )
        except WorkerFault as fault:
            if fault.code == "CANCELLED" or self._cancel_requested:
                self._cleanup_uncommitted_outputs()
                self.state = SessionState.CANCELLED
                self.current_window = None
                self._emit_cancelled()
                return
            self._cleanup_uncommitted_outputs()
            self.state = SessionState.FAILED
            self.current_window = None
            raise
        except MemoryError as error:
            self._cleanup_uncommitted_outputs()
            self.state = SessionState.FAILED
            self.current_window = None
            raise _fault(
                "ALLOCATION_FAILED",
                "MLX could not allocate reconstruction memory",
                True,
            ) from error
        except Exception as error:
            self._cleanup_uncommitted_outputs()
            self.state = SessionState.FAILED
            self.current_window = None
            error_module = type(error).__module__
            error_text = str(error).lower()
            if error_module.startswith("mlx") and (
                "alloc" in error_text or "memory" in error_text
            ):
                raise _fault(
                    "ALLOCATION_FAILED",
                    "MLX could not allocate reconstruction memory",
                    True,
                ) from error
            raise _fault(
                "RECONSTRUCTION_FAILED", "reconstruction failed before commit"
            ) from error
        finally:
            with self._condition:
                self._processing_active = False
                self._condition.notify_all()


__all__ = [
    "EventSink",
    "PersistedFrame",
    "ReconstructionModel",
    "SessionRunner",
    "SessionState",
    "validate_project_root",
]
