"""Disk-backed, direct-mode reconstruction session supervision.

Version 1 deliberately ships one honest inference mode: a complete sequence of at
most 32 frames is buffered as package-relative descriptors and reconstructed when
input closes. Longer jobs fail with a recoverable capability error until the
overlapping-window alignment path is available; they are never reported complete.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import numpy as np

from cloudpoint_worker import ENGINE_VERSION, PROTOCOL_VERSION
from cloudpoint_worker.cpc import (
    CPCVertex,
    _open_directory_fd,
    reduce_vertices,
    write_cpc,
)
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
_UINT64_MAX = 2**64 - 1
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
_LINGBOT_MODE_ID = "cloudpoint.lingbot.point-cloud.v1"


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
    root_device: int
    root_inode: int
    manifest: dict[str, object]
    frames: dict[int, PersistedFrame]
    referenced_artifacts: frozenset[str]
    completed_windows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _OwnedOutput:
    relative_path: str
    device: int
    inode: int


def _fault(
    code: str,
    message: str,
    recoverable: bool = False,
    **details: object,
) -> WorkerFault:
    return WorkerFault(code, message, recoverable, dict(details))


def _finite_float(value: object, *, nonnegative: bool = False) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(result) or (nonnegative and result < 0):
        return None
    return result


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


def _manifest_relative(value: object) -> str:
    try:
        return _safe_relative(value)
    except WorkerFault as error:
        raise _fault(
            "PROJECT_INVALID_MANIFEST",
            "manifest contains an unsafe project-relative path",
        ) from error


def _validated_lingbot_manifest(decoded: object) -> dict[str, object]:
    if not isinstance(decoded, dict):
        raise _fault(
            "PROJECT_UNSUPPORTED_FORMAT",
            "CloudPoint project format version 2 or 3 is required",
        )
    manifest = cast(dict[str, object], decoded)
    version = manifest.get("formatVersion")
    if version == 2:
        return manifest
    if version != 3:
        raise _fault(
            "PROJECT_UNSUPPORTED_FORMAT",
            "CloudPoint project format version 2 or 3 is required",
        )

    plan = manifest.get("reconstructionPlan")
    if not isinstance(plan, dict):
        raise _fault(
            "PROJECT_INVALID_MANIFEST",
            "manifest reconstructionPlan is invalid",
        )
    mode_id = plan.get("modeID")
    if not isinstance(mode_id, str):
        raise _fault(
            "PROJECT_INVALID_MANIFEST",
            "manifest reconstruction mode ID is invalid",
        )
    if mode_id != _LINGBOT_MODE_ID:
        raise _fault(
            "PROJECT_UNSUPPORTED_MODE",
            "the LingBot worker cannot run this reconstruction mode",
            modeID=mode_id,
        )

    configuration = plan.get("configuration")
    output = manifest.get("outputState")
    if (
        not isinstance(configuration, dict)
        or configuration.get("type") != "lingbotPointCloud"
        or not isinstance(configuration.get("settings"), dict)
        or not isinstance(output, dict)
        or output.get("type") != "pointCloud"
    ):
        raise _fault(
            "PROJECT_INVALID_MANIFEST",
            "manifest LingBot plan or output state is invalid",
        )
    return manifest


def _directory_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if no_follow == 0 or directory == 0:
        raise OSError("safe descriptor-relative traversal is unavailable")
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _file_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow == 0:
        raise OSError("safe descriptor-relative traversal is unavailable")
    return os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)


def _open_project_fd(
    project_root: Path, expected_identity: tuple[int, int] | None = None
) -> int:
    descriptor = _open_directory_fd(project_root)
    try:
        info = os.fstat(descriptor)
        identity = (info.st_dev, info.st_ino)
        if expected_identity is not None and identity != expected_identity:
            raise OSError("project package identity changed")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_child_directory(parent_fd: int, name: str) -> int:
    return os.open(name, _directory_flags(), dir_fd=parent_fd)


def _read_manifest(root_fd: int, root_identity: tuple[int, int]) -> _ProjectSnapshot:
    descriptor = -1
    try:
        descriptor = os.open("Manifest.json", _file_flags(), dir_fd=root_fd)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
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

        def reject_constant(token: str) -> object:
            raise ValueError(f"nonstandard JSON number {token}")

        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            payload = stream.read(_MANIFEST_MAX_BYTES + 1)
            if len(payload) > _MANIFEST_MAX_BYTES:
                raise OSError("manifest exceeds the worker bound")
            decoded = json.loads(
                payload.decode("utf-8", errors="strict"),
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise _fault(
            "PROJECT_INVALID_MANIFEST", "Manifest.json is unavailable or invalid"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    manifest = _validated_lingbot_manifest(decoded)
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
        timestamp_value = _finite_float(timestamp, nonnegative=True)
        if (
            type(index) is not int
            or not 0 <= index <= _UINT32_MAX
            or index <= previous_index
            or timestamp_value is None
        ):
            raise _fault("PROJECT_INVALID_MANIFEST", "manifest frame is invalid")
        relative = _manifest_relative(raw.get("relativePath"))
        frames[index] = PersistedFrame(index, timestamp_value, relative)
        previous_index = index

    completed: list[dict[str, object]] = []
    referenced: set[str] = set()
    previous_window = -1
    previous_output_end = -1
    for raw_window in windows_value:
        if not isinstance(raw_window, dict):
            raise _fault(
                "PROJECT_INVALID_MANIFEST", "completed window is not an object"
            )
        window = cast(dict[str, object], raw_window)
        index = window.get("index")
        inference_start = window.get("inferenceFrameStart")
        frame_start = window.get("frameStart")
        frame_end = window.get("frameEnd")
        last_processed = window.get("lastProcessedFrameIndex")
        alignment = window.get("alignmentRowMajor")
        inlier_count = window.get("inlierCount")
        duration = window.get("durationSeconds")
        alignment_is_valid = (
            isinstance(alignment, list)
            and len(alignment) == 16
            and all(_finite_float(value) is not None for value in alignment)
        )
        duration_value = _finite_float(duration, nonnegative=True)
        if (
            type(index) is not int
            or not 0 <= index <= _UINT32_MAX
            or index <= previous_window
            or type(inference_start) is not int
            or type(frame_start) is not int
            or type(frame_end) is not int
            or type(last_processed) is not int
            or not 0
            <= inference_start
            <= frame_start
            <= frame_end
            <= last_processed
            <= _UINT32_MAX
            or frame_start <= previous_output_end
            or not alignment_is_valid
            or type(inlier_count) is not int
            or not 0 <= inlier_count <= _UINT64_MAX
            or duration_value is None
        ):
            raise _fault(
                "PROJECT_INVALID_MANIFEST", "completed window bounds are invalid"
            )
        point_path = _manifest_relative(window.get("pointChunkRelativePath"))
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
        artifact_indices: list[int] = []
        previous_artifact = -1
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise _fault("PROJECT_INVALID_MANIFEST", "frame artifact is invalid")
            frame_index = artifact.get("frameIndex")
            artifact_window = artifact.get("windowIndex")
            artifact_duration = artifact.get("durationSeconds")
            artifact_duration_value = _finite_float(artifact_duration, nonnegative=True)
            if (
                type(frame_index) is not int
                or frame_index not in frames
                or not frame_start <= frame_index <= frame_end
                or frame_index <= previous_artifact
                or type(artifact_window) is not int
                or artifact_window != index
                or artifact_duration_value is None
            ):
                raise _fault(
                    "PROJECT_INVALID_MANIFEST", "frame artifact index is invalid"
                )
            expected = artifact_paths(frame_index)
            actual = (
                _manifest_relative(artifact.get("depthRelativePath")),
                _manifest_relative(artifact.get("confidenceRelativePath")),
                _manifest_relative(artifact.get("geometryRelativePath")),
            )
            if actual != expected:
                raise _fault(
                    "PROJECT_INVALID_MANIFEST", "frame artifact paths are not canonical"
                )
            referenced.update(actual)
            artifact_indices.append(frame_index)
            previous_artifact = frame_index
        expected_artifact_indices = [
            frame_index
            for frame_index in frames
            if frame_start <= frame_index <= frame_end
        ]
        if (
            artifact_indices[0] != frame_start
            or artifact_indices[-1] != frame_end
            or artifact_indices != expected_artifact_indices
        ):
            raise _fault(
                "PROJECT_INVALID_MANIFEST",
                "completed window artifacts do not cover its ordered frame range",
            )
        completed.append(window)
        previous_window = index
        previous_output_end = frame_end
    return _ProjectSnapshot(
        project_id=project_id,
        root_device=root_identity[0],
        root_inode=root_identity[1],
        manifest=manifest,
        frames=frames,
        referenced_artifacts=frozenset(referenced),
        completed_windows=tuple(completed),
    )


def validate_project_root(project_root: Path) -> _ProjectSnapshot:
    """Validate one existing project package without following symlinks."""

    if not project_root.is_absolute() or project_root.suffix != ".cloudpoint":
        raise _fault(
            "PROJECT_INVALID_PATH",
            "project must be an absolute existing .cloudpoint package",
            True,
        )
    root_fd = -1
    try:
        root_fd = _open_project_fd(project_root)
        root_info = os.fstat(root_fd)
        for name in ("Frames", "Predictions", "Points", "Logs"):
            child_fd = -1
            try:
                child_fd = _open_child_directory(root_fd, name)
            finally:
                if child_fd >= 0:
                    os.close(child_fd)
        return _read_manifest(root_fd, (root_info.st_dev, root_info.st_ino))
    except WorkerFault:
        raise
    except OSError as error:
        raise _fault(
            "PROJECT_INVALID_PATH",
            "project package or one of its required directories is unsafe",
            True,
        ) from error
    finally:
        if root_fd >= 0:
            os.close(root_fd)


def _open_regular_relative(
    project_root: Path,
    relative: str,
    *,
    root_identity: tuple[int, int] | None = None,
) -> int:
    relative = _safe_relative(relative)
    components = relative.split("/")
    directory_fd = -1
    descriptor = -1
    try:
        directory_fd = _open_project_fd(project_root, root_identity)
        for component in components[:-1]:
            next_fd = _open_child_directory(directory_fd, component)
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(components[-1], _file_flags(), dir_fd=directory_fd)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("not a regular file")
        return descriptor
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise _fault(
            "PATH_OUTSIDE_PROJECT",
            "path must resolve to a regular package file without symlinks",
            True,
            relativePath=relative,
        ) from error
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _real_project_file(
    project_root: Path,
    relative: str,
    root_identity: tuple[int, int] | None = None,
) -> None:
    descriptor = _open_regular_relative(
        project_root, relative, root_identity=root_identity
    )
    os.close(descriptor)


def _validate_frame_relative(frame: PersistedFrame) -> None:
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


def _validate_frame_path(
    project_root: Path,
    frame: PersistedFrame,
    root_identity: tuple[int, int] | None = None,
) -> None:
    _validate_frame_relative(frame)
    _real_project_file(project_root, frame.relative_path, root_identity)


@contextmanager
def _opened_frame_path(
    project_root: Path,
    frame: PersistedFrame,
    root_identity: tuple[int, int],
) -> Iterator[Path]:
    _validate_frame_relative(frame)
    descriptor = _open_regular_relative(
        project_root, frame.relative_path, root_identity=root_identity
    )
    try:
        yield Path(f"/dev/fd/{descriptor}")
    finally:
        os.close(descriptor)


def _capture_owned_output(
    project_root: Path, relative: str, root_identity: tuple[int, int]
) -> _OwnedOutput:
    descriptor = _open_regular_relative(
        project_root, relative, root_identity=root_identity
    )
    try:
        info = os.fstat(descriptor)
        return _OwnedOutput(relative, info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)


def _remove_owned_output(
    project_root: Path, owned: _OwnedOutput, root_identity: tuple[int, int]
) -> None:
    components = _safe_relative(owned.relative_path).split("/")
    if len(components) != 2 or components[0] not in {"Predictions", "Points"}:
        return
    root_fd = -1
    directory_fd = -1
    try:
        root_fd = _open_project_fd(project_root, root_identity)
        directory_fd = _open_child_directory(root_fd, components[0])
        try:
            info = os.stat(components[1], dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if (
            stat.S_ISREG(info.st_mode)
            and info.st_dev == owned.device
            and info.st_ino == owned.inode
        ):
            os.unlink(components[1], dir_fd=directory_fd)
    except OSError:
        return
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _cleanup_directory(
    project_root: Path,
    directory_name: str,
    root_identity: tuple[int, int],
    should_remove: Callable[[str], bool],
) -> None:
    root_fd = -1
    directory_fd = -1
    try:
        root_fd = _open_project_fd(project_root, root_identity)
        directory_fd = _open_child_directory(root_fd, directory_name)
        for name in os.listdir(directory_fd):
            if not should_remove(name):
                continue
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                with suppress(FileNotFoundError):
                    os.unlink(name, dir_fd=directory_fd)
    except OSError as error:
        raise _fault(
            "PROJECT_INVALID_PATH",
            f"project directory {directory_name} became unsafe",
            True,
        ) from error
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _cleanup_orphans(
    project_root: Path,
    referenced: frozenset[str],
    root_identity: tuple[int, int],
) -> None:
    for directory_name, canonical_pattern in (
        ("Predictions", _CANONICAL_PREDICTION),
        ("Points", _CANONICAL_POINTS),
    ):

        def should_remove(
            name: str,
            directory: str = directory_name,
            pattern: re.Pattern[str] = canonical_pattern,
        ) -> bool:
            return _PARTIAL.fullmatch(name) is not None or (
                pattern.fullmatch(name) is not None
                and f"{directory}/{name}" not in referenced
            )

        _cleanup_directory(
            project_root,
            directory_name,
            root_identity,
            should_remove,
        )


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
        self._expected_replay: tuple[PersistedFrame, ...] = ()
        self._frames: list[PersistedFrame] = []
        self._input_finished = False
        self._begun = False
        self._processing_active = False
        self._pause_requested = False
        self._pause_emitted = False
        self._cancel_requested = False
        self._cancelled_emitted = False
        self._created_outputs: list[_OwnedOutput] = []
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
            expected_replay = self._validate_checkpoint(snapshot, checkpoint)
            if self._cleanup_orphans_enabled:
                _cleanup_orphans(
                    self.project_root,
                    snapshot.referenced_artifacts,
                    (snapshot.root_device, snapshot.root_inode),
                )
            self._snapshot = snapshot
            self._checkpoint = checkpoint
            self._expected_replay = expected_replay
            self.last_completed_window_index = (
                checkpoint.next_window_index - 1 if checkpoint is not None else None
            )
            self._begun = True

    def _validate_checkpoint(
        self, snapshot: _ProjectSnapshot, checkpoint: ResumeCheckpoint | None
    ) -> tuple[PersistedFrame, ...]:
        if checkpoint is None:
            if snapshot.completed_windows:
                raise _fault(
                    "INVALID_RESUME_CHECKPOINT",
                    "completed project state requires a resume checkpoint",
                    True,
                )
            return ()
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
        artifact_indices = [
            cast(int, artifact.get("frameIndex"))
            for window in snapshot.completed_windows
            for artifact in cast(list[dict[str, object]], window["frameArtifacts"])
            if checkpoint.replay_from_frame_index
            <= cast(int, artifact.get("frameIndex"))
            <= checkpoint.last_committed_frame_index
        ]
        expected_replay = tuple(
            snapshot.frames[index]
            for index in artifact_indices
            if checkpoint.replay_from_frame_index
            <= index
            <= checkpoint.last_committed_frame_index
        )
        if (
            not expected_replay
            or expected_replay[0].index != checkpoint.replay_from_frame_index
            or expected_replay[-1].index != checkpoint.last_committed_frame_index
        ):
            raise _fault(
                "MISSING_REPLAY_ARTIFACTS",
                "resume context is not fully committed in the manifest",
                True,
            )
        for relative in snapshot.referenced_artifacts:
            _real_project_file(
                self.project_root,
                relative,
                (snapshot.root_device, snapshot.root_inode),
            )
        return expected_replay

    def enqueue(self, frame: PersistedFrame) -> None:
        with self._condition:
            if (
                not self._begun
                or self._input_finished
                or self._cancel_requested
                or self.state == SessionState.FAILED
            ):
                raise _fault("INVALID_STATE", "session is not accepting frames", True)
            configuration = cast(ConfigurePayload, self._configuration)
            maximum = min(_DIRECT_FRAME_LIMIT, configuration.window_size)
            if len(self._frames) >= maximum:
                fault = _fault(
                    "WINDOWING_UNAVAILABLE",
                    "direct reconstruction admission limit has been reached",
                    True,
                    frameCount=len(self._frames),
                    maximum=maximum,
                )
                self.state = SessionState.FAILED
                raise fault
            timestamp_value = _finite_float(frame.source_timestamp, nonnegative=True)
            if (
                type(frame.index) is not int
                or not 0 <= frame.index <= _UINT32_MAX
                or timestamp_value is None
            ):
                raise _fault("INVALID_FRAME", "frame descriptor is invalid", True)
            snapshot = cast(_ProjectSnapshot, self._snapshot)
            _validate_frame_path(
                self.project_root,
                frame,
                (snapshot.root_device, snapshot.root_inode),
            )
            if self._frames and frame.index <= self._frames[-1].index:
                raise _fault(
                    "FRAME_ORDER_VIOLATION", "frame indices must increase", True
                )

            replay = False
            checkpoint = self._checkpoint
            if checkpoint is not None:
                replay_ordinal = len(self._frames)
                if frame.index > checkpoint.last_committed_frame_index:
                    fault = _fault(
                        "WINDOWING_UNAVAILABLE",
                        "resumed output requires real Sim3 window alignment",
                        True,
                    )
                    self.state = SessionState.FAILED
                    raise fault
                if replay_ordinal >= len(self._expected_replay):
                    raise _fault(
                        "REPLAY_ORDER_VIOLATION",
                        "replay contains more descriptors than the manifest boundary",
                        True,
                    )
                committed = self._expected_replay[replay_ordinal]
                if (
                    committed.index != frame.index
                    or committed.relative_path != frame.relative_path
                    or committed.source_timestamp != frame.source_timestamp
                ):
                    raise _fault(
                        "REPLAY_ORDER_VIOLATION",
                        "replay descriptors must exactly match manifest artifact order",
                        True,
                    )
                replay = True
            admitted = PersistedFrame(
                frame.index, timestamp_value, frame.relative_path, replay
            )
            self._frames.append(admitted)
            if not replay:
                self.queued_frames += 1

    def finish_input(self) -> None:
        with self._condition:
            if not self._begun or self._input_finished or self._cancel_requested:
                raise _fault("INVALID_STATE", "input cannot be finished now", True)
            if self.state == SessionState.FAILED:
                raise _fault("INVALID_STATE", "failed session cannot be finished", True)
            if self._checkpoint is not None:
                replay_descriptors = tuple(
                    frame for frame in self._frames if frame.replay
                )
                if replay_descriptors != tuple(
                    PersistedFrame(
                        frame.index,
                        frame.source_timestamp,
                        frame.relative_path,
                        True,
                    )
                    for frame in self._expected_replay
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
        snapshot = self._snapshot
        if snapshot is None:
            return
        root_identity = (snapshot.root_device, snapshot.root_inode)
        for owned in reversed(self._created_outputs):
            _remove_owned_output(self.project_root, owned, root_identity)
        self._created_outputs.clear()
        for directory_name in ("Predictions", "Points"):
            # Cleanup is best effort after a terminal failure; never replace the
            # original reconstruction fault with a path-race diagnostic.
            with suppress(WorkerFault):
                _cleanup_directory(
                    self.project_root,
                    directory_name,
                    root_identity,
                    lambda name: _PARTIAL.fullmatch(name) is not None,
                )

    def process(self) -> None:
        """Run one complete direct reconstruction on the calling thread."""

        started = time.monotonic()
        with self._condition:
            if not self._begun or not self._input_finished or self._processing_active:
                raise _fault(
                    "INVALID_STATE", "finishInput must precede processing", True
                )
            if self.state in {
                SessionState.CANCELLED,
                SessionState.COMPLETED,
                SessionState.FAILED,
            }:
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
                self._cooperative_boundary()
                replay_event = SessionCompleted(
                    PROTOCOL_VERSION,
                    uuid.uuid4(),
                    self.project_id,
                    "sessionCompleted",
                    SessionCompletedPayload(0, 0, time.monotonic() - started),
                )
                with self._condition:
                    if self._cancel_requested:
                        raise _fault("CANCELLED", "session was cancelled", True)
                    self._emit(replay_event)
                    self.state = SessionState.COMPLETED
                return

            snapshot = cast(_ProjectSnapshot, self._snapshot)
            root_identity = (snapshot.root_device, snapshot.root_inode)
            prepared: list[PreprocessedFrame] = []
            for frame in self._frames:
                self._cooperative_boundary()
                try:
                    with _opened_frame_path(
                        self.project_root, frame, root_identity
                    ) as opened_path:
                        prepared.append(preprocess_image(opened_path))
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
                with self._condition:
                    if self._cancel_requested:
                        raise _fault("CANCELLED", "session was cancelled", True)
                    self._emit(
                        FrameStarted(
                            PROTOCOL_VERSION,
                            uuid.uuid4(),
                            self.project_id,
                            "frameStarted",
                            FrameStartedPayload(frame.index, window_index),
                        )
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
                for relative in (
                    artifacts.depth_path,
                    artifacts.confidence_path,
                    artifacts.geometry_path,
                ):
                    self._created_outputs.append(
                        _capture_owned_output(
                            self.project_root, relative, root_identity
                        )
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
                self._cooperative_boundary()
                with self._condition:
                    if self._cancel_requested:
                        raise _fault("CANCELLED", "session was cancelled", True)
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
            self._cooperative_boundary()
            if not vertices:
                raise _fault(
                    "EMPTY_POINT_CLOUD",
                    "no points survived the configured confidence threshold",
                    True,
                )
            self.state = SessionState.FINALIZING
            point_relative = window_point_path(window_index)
            self._cooperative_boundary()
            descriptor = write_cpc(
                self.project_root / point_relative,
                unique_frames[0].index,
                unique_frames[-1].index,
                vertices,
            )
            self._created_outputs.append(
                _capture_owned_output(self.project_root, point_relative, root_identity)
            )
            self._cooperative_boundary()
            window_event = WindowCompleted(
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
            # Cancellation and ownership transfer are serialized. Once this event
            # is visible, native owns a complete window and cleanup must preserve it.
            with self._condition:
                if self._cancel_requested:
                    raise _fault("CANCELLED", "session was cancelled", True)
                self._emit(window_event)
                self._created_outputs.clear()
                self.window_count = 1
                self.last_completed_window_index = window_index
                self.current_window = None
            self._cooperative_boundary()
            session_event = SessionCompleted(
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
            with self._condition:
                if self._cancel_requested:
                    raise _fault("CANCELLED", "session was cancelled", True)
                self._emit(session_event)
                self.state = SessionState.COMPLETED
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
