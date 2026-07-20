"""Deterministic CloudPoint CPC1 point-chunk encoding."""

from __future__ import annotations

import errno
import math
import os
import stat
import struct
import uuid
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from cloudpoint_worker.errors import WorkerFault

_HEADER = struct.Struct("<4sHHQII8s")
_VERTEX = struct.Struct("<fff4BeHI")
_MAGIC = b"CPC1"
_VERSION = 1
_STRIDE = 24
_MAX_POINTS = 50_000_000
_MAX_BYTES = 1_200_000_032
_UINT16_MAX = 2**16 - 1
_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class CPCVertex:
    position: tuple[float, float, float]
    rgba: tuple[int, int, int, int]
    confidence: float
    flags: int
    source_frame: int
    # Deterministic tie-break metadata. It is intentionally not serialized.
    pixel_index: int = field(default=0, compare=True)


@dataclass(frozen=True)
class CPCDescriptor:
    relative_path: str
    point_count: int
    frame_start: int
    frame_end: int


@dataclass(frozen=True)
class CPCContents:
    descriptor: CPCDescriptor
    vertices: tuple[CPCVertex, ...]

    @property
    def point_count(self) -> int:
        return self.descriptor.point_count


def _fault(message: str, *, code: str = "INVALID_POINT_CHUNK") -> WorkerFault:
    return WorkerFault(code, message, False)


def _check_final_path(path: Path) -> None:
    if not path.is_absolute():
        raise _fault("output path must be absolute")
    try:
        target = path.lstat()
    except FileNotFoundError:
        target = None
    if target is not None:
        if stat.S_ISLNK(target.st_mode):
            raise _fault("output path must not be a symlink")
        raise _fault("output already exists", code="OUTPUT_ALREADY_EXISTS")
    try:
        parent = path.parent.lstat()
    except FileNotFoundError as error:
        raise _fault("output parent does not exist") from error
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise _fault("output parent must be a real directory")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write one exclusive sibling partial and promote without clobbering."""

    _check_final_path(path)
    partial = path.with_name(f".{path.name}.{uuid.uuid4()}.partial")
    fd = -1
    try:
        fd = os.open(
            partial,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(fd, "wb", closefd=True) as stream:
            fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(partial, path, follow_symlinks=False)
        except FileExistsError as error:
            raise _fault(
                "output already exists", code="OUTPUT_ALREADY_EXISTS"
            ) from error
        os.unlink(partial)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except WorkerFault:
        raise
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise _fault(
                "output already exists", code="OUTPUT_ALREADY_EXISTS"
            ) from error
        raise _fault(f"atomic output failed: {error}") from error
    finally:
        if fd >= 0:
            os.close(fd)
        with suppress(FileNotFoundError):
            partial.unlink()


def _validated_vertex(vertex: CPCVertex) -> CPCVertex:
    if len(vertex.position) != 3 or not all(
        math.isfinite(float(value)) for value in vertex.position
    ):
        raise _fault("vertex position must contain three finite values")
    if len(vertex.rgba) != 4 or any(
        type(value) is not int or value < 0 or value > 255 for value in vertex.rgba
    ):
        raise _fault("vertex RGBA must contain four bytes")
    if not math.isfinite(float(vertex.confidence)):
        raise _fault("vertex confidence must be finite")
    if type(vertex.flags) is not int or not 0 <= vertex.flags <= _UINT16_MAX:
        raise _fault("vertex flags exceed UInt16")
    if (
        type(vertex.source_frame) is not int
        or not 0 <= vertex.source_frame <= _UINT32_MAX
    ):
        raise _fault("source frame exceeds UInt32")
    return vertex


def reduce_vertices(
    vertices: Iterable[CPCVertex], *, voxel_size: float
) -> tuple[CPCVertex, ...]:
    """Keep one deterministic, highest-confidence vertex per voxel."""

    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise _fault("voxel size must be finite and positive")
    winners: dict[tuple[int, int, int], CPCVertex] = {}
    for candidate in vertices:
        candidate = _validated_vertex(candidate)
        key = tuple(math.floor(axis / voxel_size) for axis in candidate.position)
        current = winners.get(key)
        if current is None or (
            -candidate.confidence,
            candidate.source_frame,
            candidate.pixel_index,
        ) < (
            -current.confidence,
            current.source_frame,
            current.pixel_index,
        ):
            winners[key] = candidate
    return tuple(winners[key] for key in sorted(winners))


def _relative_name(path: Path) -> str:
    if path.parent.name == "Points":
        return f"Points/{path.name}"
    return path.name


def write_cpc(
    path: Path,
    frame_start: int,
    frame_end_inclusive: int,
    vertices: Sequence[CPCVertex],
) -> CPCDescriptor:
    """Atomically write one CPC1 file with exact little-endian layout."""

    if (
        type(frame_start) is not int
        or type(frame_end_inclusive) is not int
        or frame_start < 0
        or frame_end_inclusive < frame_start
        or frame_end_inclusive > _UINT32_MAX
    ):
        raise _fault("invalid inclusive source frame range")
    count = len(vertices)
    total_size = _HEADER.size + count * _STRIDE
    if count > _MAX_POINTS or total_size > _MAX_BYTES:
        raise _fault("point count or encoded size exceeds CPC1 bounds")

    payload = bytearray(total_size)
    _HEADER.pack_into(
        payload,
        0,
        _MAGIC,
        _VERSION,
        _STRIDE,
        count,
        frame_start,
        frame_end_inclusive,
        b"\0" * 8,
    )
    offset = _HEADER.size
    for raw_vertex in vertices:
        vertex = _validated_vertex(raw_vertex)
        try:
            _VERTEX.pack_into(
                payload,
                offset,
                *vertex.position,
                *vertex.rgba,
                vertex.confidence,
                vertex.flags,
                vertex.source_frame,
            )
        except (OverflowError, struct.error) as error:
            raise _fault("vertex cannot be represented by CPC1") from error
        offset += _STRIDE

    atomic_write_bytes(path, payload)
    return CPCDescriptor(
        relative_path=_relative_name(path),
        point_count=count,
        frame_start=frame_start,
        frame_end=frame_end_inclusive,
    )


def read_cpc(path: Path) -> CPCContents:
    """Strictly read a CPC1 file for verification and recovery."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise _fault(f"unable to read point chunk: {error}") from error
    if len(raw) < _HEADER.size:
        raise _fault("truncated CPC1 header")
    (
        magic,
        version,
        stride,
        count,
        frame_start,
        frame_end,
        reserved,
    ) = _HEADER.unpack_from(raw)
    expected_size = _HEADER.size + count * _STRIDE
    if (
        magic != _MAGIC
        or version != _VERSION
        or stride != _STRIDE
        or reserved != b"\0" * 8
        or frame_start > frame_end
        or count > _MAX_POINTS
        or expected_size > _MAX_BYTES
        or len(raw) != expected_size
    ):
        raise _fault("invalid or truncated CPC1 file")

    vertices: list[CPCVertex] = []
    offset = _HEADER.size
    for _ in range(count):
        values = _VERTEX.unpack_from(raw, offset)
        vertex = CPCVertex(
            position=(values[0], values[1], values[2]),
            rgba=(values[3], values[4], values[5], values[6]),
            confidence=values[7],
            flags=values[8],
            source_frame=values[9],
        )
        vertices.append(_validated_vertex(vertex))
        offset += _STRIDE
    descriptor = CPCDescriptor(
        relative_path=_relative_name(path),
        point_count=count,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    return CPCContents(descriptor=descriptor, vertices=tuple(vertices))


__all__ = [
    "CPCContents",
    "CPCDescriptor",
    "CPCVertex",
    "atomic_write_bytes",
    "read_cpc",
    "reduce_vertices",
    "write_cpc",
]
