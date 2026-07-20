"""Protocol-v1 framed-stdio adapter for the reconstruction session."""

from __future__ import annotations

import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import BinaryIO, cast

import msgspec

from cloudpoint_worker import ENGINE_VERSION, PROTOCOL_VERSION
from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.model.lingbot import CONVERTED_MODEL_SHA256
from cloudpoint_worker.model_prep.provenance import MODEL_REPO, MODEL_REVISION
from cloudpoint_worker.protocol.framing import read_json_frame, write_json_frame
from cloudpoint_worker.protocol.schema import (
    BeginSessionCommand,
    CancelCommand,
    Command,
    CommandHeader,
    CommandIDTracker,
    ConfigureCommand,
    EnqueueFrameCommand,
    ErrorMessage,
    ErrorPayload,
    Event,
    FailureDisposition,
    FinishInputCommand,
    Heartbeat,
    HeartbeatPayload,
    HelloCommand,
    ModelProgress,
    ModelProgressPayload,
    PauseCommand,
    ProtocolValidationError,
    Ready,
    ReadyPayload,
    ResumeCommand,
    ShutdownCommand,
    ack,
    classify_failure,
    command_error,
    decode_command,
    recover_command_header,
)
from cloudpoint_worker.session import (
    PersistedFrame,
    ReconstructionModel,
    SessionRunner,
    SessionState,
    validate_project_root,
)

type ModelLoader = Callable[[], ReconstructionModel]
type ServerSink = Callable[[Event], None]


def _fault(code: str, message: str, recoverable: bool = True) -> WorkerFault:
    return WorkerFault(code, message, recoverable)


class _PrefixedReader:
    """Replay a consumed frame byte without buffering the framed body."""

    def __init__(self, prefix: bytes, stream: BinaryIO) -> None:
        self._prefix = prefix
        self._stream = stream

    def read(self, count: int = -1) -> bytes:
        if count == 0:
            return b""
        prefix = self._prefix
        self._prefix = b""
        if count < 0:
            return prefix + self._stream.read()
        if len(prefix) >= count:
            self._prefix = prefix[count:]
            return prefix[:count]
        return prefix + self._stream.read(count - len(prefix))


class WorkerServer:
    """Decode commands, own responses, and keep MLX work off the reader thread."""

    def __init__(
        self,
        project_root: Path,
        *,
        model_loader: ModelLoader,
        input_stream: BinaryIO | None = None,
        output_stream: BinaryIO | None = None,
        event_sink: ServerSink | None = None,
        heartbeat_interval: float = 5.0,
    ) -> None:
        snapshot = validate_project_root(project_root)
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat interval must be positive")
        if event_sink is not None and output_stream is not None:
            raise ValueError("event_sink and output_stream are mutually exclusive")
        self.project_root = project_root
        self.project_id = snapshot.project_id
        self._model_loader = model_loader
        self._input = input_stream or cast(BinaryIO, sys.stdin.buffer)
        self._output = output_stream or cast(BinaryIO, sys.stdout.buffer)
        self._external_sink = event_sink
        self._heartbeat_interval = heartbeat_interval
        self._write_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cloudpoint-mlx"
        )
        self._future: Future[None] | None = None
        self._model_future: Future[None] | None = None
        self._tracker = CommandIDTracker()
        self._hello_received = False
        self._model_loading = False
        self._model_ready = False
        self._shutdown_requested = False
        self._closed = False
        self._model: ReconstructionModel | None = None
        self._configuration = None
        self._runner: SessionRunner | None = None

    def _emit(self, event: Event) -> None:
        with self._write_lock:
            if self._external_sink is not None:
                self._external_sink(event)
            else:
                write_json_frame(self._output, event)

    def _heartbeat(self) -> Heartbeat:
        runner = self._runner
        busy = self._model_loading or (
            self._future is not None and not self._future.done()
        )
        return Heartbeat(
            PROTOCOL_VERSION,
            uuid.uuid4(),
            self.project_id,
            "heartbeat",
            HeartbeatPayload(
                busy,
                time.monotonic(),
                runner.queued_frames if runner is not None else 0,
                runner.processed_frames if runner is not None else 0,
                runner.current_window if runner is not None else None,
            ),
        )

    def _start_heartbeats(self) -> None:
        if self._heartbeat_thread is not None:
            return

        def loop() -> None:
            while not self._heartbeat_stop.wait(self._heartbeat_interval):
                try:
                    self._emit(self._heartbeat())
                except Exception:
                    return

        self._heartbeat_thread = threading.Thread(
            target=loop, name="cloudpoint-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _require_hello(self) -> None:
        with self._state_lock:
            if not self._hello_received or not self._model_ready or self._model is None:
                raise _fault("INVALID_STATE", "hello must complete before this command")

    def _require_runner(self) -> SessionRunner:
        if self._runner is None:
            raise _fault("INVALID_STATE", "beginSession must precede this command")
        return self._runner

    def _asynchronous_error(self, fault: WorkerFault) -> ErrorMessage:
        return ErrorMessage(
            PROTOCOL_VERSION,
            uuid.uuid4(),
            self.project_id,
            "error",
            None,
            ErrorPayload(
                fault.code, fault.message, fault.recoverable, dict(fault.details)
            ),
        )

    def _load_after_hello(self) -> None:
        with self._state_lock:
            if self._shutdown_requested:
                self._model_loading = False
                return
            self._emit(
                ModelProgress(
                    PROTOCOL_VERSION,
                    uuid.uuid4(),
                    self.project_id,
                    "modelProgress",
                    ModelProgressPayload("validating", 0, 1),
                )
            )
        try:
            model = self._model_loader()
        except WorkerFault as fault:
            with self._state_lock:
                self._model_loading = False
                if not self._shutdown_requested:
                    self._emit(self._asynchronous_error(fault))
            return
        except Exception as error:
            should_report = False
            with self._state_lock:
                self._model_loading = False
                if not self._shutdown_requested:
                    should_report = True
                    self._emit(
                        self._asynchronous_error(
                            WorkerFault(
                                "MODEL_LOAD_FAILED",
                                "model could not be loaded",
                                False,
                            )
                        )
                    )
            if should_report:
                print(
                    f"MODEL_LOAD_FAILED: {type(error).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
            return
        with self._state_lock:
            if self._shutdown_requested:
                self._model_loading = False
                return
            self._model = model
            self._emit(
                ModelProgress(
                    PROTOCOL_VERSION,
                    uuid.uuid4(),
                    self.project_id,
                    "modelProgress",
                    ModelProgressPayload("loading", 1, 1),
                )
            )
            self._emit(
                Ready(
                    PROTOCOL_VERSION,
                    uuid.uuid4(),
                    self.project_id,
                    "ready",
                    ReadyPayload(
                        ENGINE_VERSION,
                        MODEL_REPO,
                        MODEL_REVISION,
                        CONVERTED_MODEL_SHA256,
                    ),
                )
            )
            self._model_ready = True
            self._model_loading = False

    def _run_session(self) -> None:
        runner = self._require_runner()
        try:
            runner.process()
        except WorkerFault as fault:
            self._emit(self._asynchronous_error(fault))
        except Exception as error:  # Last-resort transport containment.
            print(
                f"RECONSTRUCTION_FAILED: {type(error).__name__}",
                file=sys.stderr,
                flush=True,
            )
            self._emit(
                self._asynchronous_error(
                    WorkerFault(
                        "RECONSTRUCTION_FAILED",
                        "reconstruction worker failed unexpectedly",
                        False,
                    )
                )
            )

    def handle(self, command: Command) -> bool:
        """Handle one decoded command; return false when transport should close."""

        if self._closed:
            return False
        try:
            self._tracker.insert(command.id)
            if command.project_id != self.project_id:
                raise _fault("PROJECT_ID_MISMATCH", "command project ID is incorrect")
            if isinstance(command, HelloCommand):
                if self._hello_received:
                    raise _fault("INVALID_STATE", "hello was already received")
                self._hello_received = True
                self._model_loading = True
                self._emit(ack(command))
                # Protocol readiness begins with an immediate post-ACK heartbeat.
                self._emit(self._heartbeat())
                self._start_heartbeats()
                # Loading must never own the command-reader thread: shutdown and
                # control commands remain live during multi-second model setup.
                self._model_future = self._executor.submit(self._load_after_hello)
                return True
            if isinstance(command, ShutdownCommand) and self._hello_received:
                with self._state_lock:
                    self._shutdown_requested = True
                runner = self._runner
                if runner is not None and runner.state not in {
                    SessionState.COMPLETED,
                    SessionState.CANCELLED,
                    SessionState.FAILED,
                }:
                    runner.cancel(emit_event=False)
                self._emit(ack(command))
                if runner is not None:
                    runner.publish_cancelled_if_quiescent()
                return False
            self._require_hello()
            with self._write_lock:
                if isinstance(command, ConfigureCommand):
                    if self._runner is not None:
                        raise _fault("INVALID_STATE", "session configuration is frozen")
                    self._configuration = command.payload
                    self._emit(ack(command))
                    return True
                if isinstance(command, BeginSessionCommand):
                    if self._configuration is None or self._runner is not None:
                        raise _fault(
                            "INVALID_STATE", "configure must precede beginSession"
                        )
                    runner = SessionRunner(
                        self.project_root,
                        cast(ReconstructionModel, self._model),
                        self._emit,
                        project_id=self.project_id,
                    )
                    runner.configure(self._configuration)
                    runner.begin(command.payload)
                    self._runner = runner
                    self._emit(ack(command))
                    return True
                if isinstance(command, EnqueueFrameCommand):
                    runner = self._require_runner()
                    runner.enqueue(
                        PersistedFrame(
                            command.payload.frame_index,
                            command.payload.source_timestamp,
                            command.payload.relative_path,
                        )
                    )
                    self._emit(ack(command))
                    return True
                if isinstance(command, FinishInputCommand):
                    runner = self._require_runner()
                    runner.finish_input()
                    self._emit(ack(command))
                    self._future = self._executor.submit(self._run_session)
                    return True
                if isinstance(command, PauseCommand):
                    runner = self._require_runner()
                    runner.pause(emit_event=False)
                    self._emit(ack(command))
                    runner.publish_paused_if_quiescent()
                    return True
                if isinstance(command, ResumeCommand):
                    runner = self._require_runner()
                    runner.resume()
                    self._emit(ack(command))
                    return True
                if isinstance(command, CancelCommand):
                    runner = self._require_runner()
                    runner.cancel(emit_event=False)
                    self._emit(ack(command))
                    runner.publish_cancelled_if_quiescent()
                    return True
                raise _fault("UNKNOWN_MESSAGE_TYPE", "unknown command type")
        except WorkerFault as fault:
            self._emit(command_error(command, fault))
            return True

    def wait_for_idle(self, timeout: float | None = None) -> None:
        future = self._future
        if future is None:
            return
        try:
            future.result(timeout=timeout)
        except TimeoutError as error:
            raise TimeoutError("worker session did not reach a boundary") from error

    def wait_for_model(self, timeout: float | None = None) -> None:
        future = self._model_future
        if future is None:
            return
        try:
            future.result(timeout=timeout)
        except TimeoutError as error:
            raise TimeoutError("worker model did not finish loading") from error

    def _header_error(self, header: CommandHeader, fault: WorkerFault) -> ErrorMessage:
        return ErrorMessage(
            PROTOCOL_VERSION,
            uuid.uuid4(),
            header.project_id,
            "error",
            header.id,
            ErrorPayload(
                fault.code, fault.message, fault.recoverable, dict(fault.details)
            ),
        )

    def _handle_decoding_failure(
        self,
        fault: WorkerFault | ProtocolValidationError,
        raw: object | None,
    ) -> bool:
        header = recover_command_header(raw) if raw is not None else None
        disposition = classify_failure(fault, header)
        worker_fault = (
            fault.worker_fault()
            if isinstance(fault, ProtocolValidationError)
            else fault
        )
        if disposition == FailureDisposition.CLOSE_WITHOUT_RESPONSE:
            return False
        if disposition == FailureDisposition.ASYNC_ERROR_THEN_CLOSE:
            self._emit(self._asynchronous_error(worker_fault))
            return False
        if header is None:
            return False
        self._emit(self._header_error(header, worker_fault))
        return disposition == FailureDisposition.COMMAND_ERROR_THEN_CONTINUE

    def serve(self) -> int:
        """Serve framed stdin/stdout without opening any network endpoint."""

        protocol_error_seen = False
        try:
            while True:
                raw: object | None = None
                try:
                    first_byte = self._input.read(1)
                    if not first_byte:
                        break
                    prefixed = cast(BinaryIO, _PrefixedReader(first_byte, self._input))
                    raw = read_json_frame(prefixed)
                    command = decode_command(raw)
                except (
                    WorkerFault,
                    ProtocolValidationError,
                    msgspec.ValidationError,
                ) as error:
                    protocol_error_seen = True
                    fault: WorkerFault | ProtocolValidationError
                    if isinstance(error, WorkerFault | ProtocolValidationError):
                        fault = error
                    else:
                        fault = ProtocolValidationError(str(error))
                    if not self._handle_decoding_failure(fault, raw):
                        break
                    continue
                if not self.handle(command):
                    break
            return 3 if protocol_error_seen else 0
        finally:
            self.close()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._shutdown_requested = True
        self._heartbeat_stop.set()
        runner = self._runner
        if runner is not None and runner.state not in {
            SessionState.COMPLETED,
            SessionState.CANCELLED,
            SessionState.FAILED,
        }:
            runner.cancel(emit_event=False)
        try:
            self.wait_for_idle(timeout=None)
        finally:
            self.wait_for_model(timeout=None)
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.join(timeout=1)
            self._executor.shutdown(wait=True, cancel_futures=False)


__all__ = ["ModelLoader", "WorkerServer"]
