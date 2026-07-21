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
_UINT32_MAX = 2**32 - 1
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_SUPPORTED_FLAGS = 0b11


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


@dataclass(frozen=True, slots=True)
class AtomicWriteReceipt:
    """Filesystem identity of one exclusively promoted canonical output."""

    device: int
    inode: int


def _fault(message: str, *, code: str = "INVALID_POINT_CHUNK") -> WorkerFault:
    return WorkerFault(code, message, False)


def _directory_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if no_follow == 0 or directory == 0:
        raise OSError(errno.ENOTSUP, "safe directory traversal is unavailable")
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _path_components(path: Path) -> tuple[str, ...]:
    if not path.is_absolute() or path.anchor not in {"/", "//"}:
        raise OSError(errno.EINVAL, "path must be absolute")
    components = path.parts[1:]
    if any(
        component in {"", ".", ".."} or "\0" in component for component in components
    ):
        raise OSError(errno.EINVAL, "path contains an unsafe component")
    return components


def _open_directory_fd(path: Path, *, create_leaf: bool = False) -> int:
    """Open every directory component without following symbolic links."""

    components = _path_components(path)
    flags = _directory_flags()
    descriptor = os.open(path.anchor, flags)
    try:
        for index, component in enumerate(components):
            created = False
            next_descriptor = -1
            try:
                try:
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create_leaf or index != len(components) - 1:
                        raise
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                        created = True
                    except FileExistsError:
                        pass
                    next_descriptor = os.open(component, flags, dir_fd=descriptor)
                if created:
                    os.fsync(descriptor)
            except BaseException:
                if next_descriptor >= 0:
                    os.close(next_descriptor)
                raise
            previous_descriptor = descriptor
            descriptor = next_descriptor
            os.close(previous_descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_parent_directory(path: Path) -> tuple[int, str]:
    components = _path_components(path)
    if not components:
        raise OSError(errno.EINVAL, "output path has no filename")
    return _open_directory_fd(path.parent), components[-1]


def _unlink_if_owned_at(
    parent_fd: int, final_name: str, receipt: AtomicWriteReceipt
) -> bool:
    try:
        info = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_dev != receipt.device
        or info.st_ino != receipt.inode
    ):
        return False
    os.unlink(final_name, dir_fd=parent_fd)
    return True


def rollback_atomic_write(path: Path, receipt: AtomicWriteReceipt) -> None:
    """Remove a promotion only while its exact inode still owns the path."""

    parent_fd = -1
    try:
        parent_fd, final_name = _open_parent_directory(path)
        if _unlink_if_owned_at(parent_fd, final_name, receipt):
            os.fsync(parent_fd)
    except (OSError, ValueError):
        return
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)


def atomic_write_bytes(path: Path, payload: bytes) -> AtomicWriteReceipt:
    """Write one exclusive sibling partial and promote without clobbering."""

    parent_fd = -1
    fd = -1
    partial_name: str | None = None
    final_name = ""
    receipt: AtomicWriteReceipt | None = None
    promoted = False
    completed = False
    try:
        parent_fd, final_name = _open_parent_directory(path)
        try:
            target = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            target = None
        if target is not None:
            if stat.S_ISLNK(target.st_mode):
                raise _fault("output path must not be a symlink")
            raise _fault("output already exists", code="OUTPUT_ALREADY_EXISTS")

        partial_name = f".{final_name}.{uuid.uuid4()}.partial"
        fd = os.open(
            partial_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        stream = os.fdopen(fd, "wb", closefd=True)
        fd = -1
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            partial_info = os.fstat(stream.fileno())
            if not stat.S_ISREG(partial_info.st_mode):
                raise OSError("partial output is not a regular file")
            receipt = AtomicWriteReceipt(partial_info.st_dev, partial_info.st_ino)
        try:
            os.link(
                partial_name,
                final_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            promoted = True
        except FileExistsError as error:
            raise _fault(
                "output already exists", code="OUTPUT_ALREADY_EXISTS"
            ) from error
        final_info = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(final_info.st_mode)
            or final_info.st_dev != receipt.device
            or final_info.st_ino != receipt.inode
        ):
            raise OSError("promoted output identity changed")
        os.unlink(partial_name, dir_fd=parent_fd)
        partial_name = None
        os.fsync(parent_fd)
        completed = True
        return receipt
    except WorkerFault:
        raise
    except (OSError, ValueError) as error:
        if getattr(error, "errno", None) == errno.EEXIST:
            raise _fault(
                "output already exists", code="OUTPUT_ALREADY_EXISTS"
            ) from error
        raise _fault(f"atomic output failed: {error}") from error
    finally:
        if fd >= 0:
            os.close(fd)
        if parent_fd >= 0:
            if promoted and not completed and receipt is not None:
                with suppress(OSError):
                    if _unlink_if_owned_at(parent_fd, final_name, receipt):
                        os.fsync(parent_fd)
            if partial_name is not None:
                with suppress(FileNotFoundError):
                    os.unlink(partial_name, dir_fd=parent_fd)
            os.close(parent_fd)


def _validated_vertex(
    vertex: CPCVertex,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> CPCVertex:
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
    if type(vertex.flags) is not int or not 0 <= vertex.flags <= _SUPPORTED_FLAGS:
        raise _fault("vertex flags contain unsupported reserved bits")
    if (
        type(vertex.source_frame) is not int
        or not 0 <= vertex.source_frame <= _UINT32_MAX
    ):
        raise _fault("source frame exceeds UInt32")
    if (
        frame_start is not None
        and frame_end is not None
        and not frame_start <= vertex.source_frame <= frame_end
    ):
        raise _fault("source frame is outside the inclusive CPC1 range")
    return vertex


def _voxel_key(
    position: tuple[float, float, float], voxel_size: float
) -> tuple[int, int, int]:
    key: list[int] = []
    for axis in position:
        quotient = float(axis) / voxel_size
        if not math.isfinite(quotient):
            raise _fault("voxel coordinate exceeds Int64")
        coordinate = math.floor(quotient)
        if not _INT64_MIN <= coordinate <= _INT64_MAX:
            raise _fault("voxel coordinate exceeds Int64")
        key.append(coordinate)
    return (key[0], key[1], key[2])


def reduce_vertices(
    vertices: Iterable[CPCVertex], *, voxel_size: float
) -> tuple[CPCVertex, ...]:
    """Keep one deterministic, highest-confidence vertex per voxel."""

    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise _fault("voxel size must be finite and positive")
    winners: dict[tuple[int, int, int], CPCVertex] = {}
    for candidate in vertices:
        candidate = _validated_vertex(candidate)
        key = _voxel_key(candidate.position, voxel_size)
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
        vertex = _validated_vertex(
            raw_vertex,
            frame_start=frame_start,
            frame_end=frame_end_inclusive,
        )
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

    parent_fd = -1
    descriptor = -1
    try:
        parent_fd, filename = _open_parent_directory(path)
        descriptor = os.open(
            filename,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise _fault("point chunk must be a regular file")
        if info.st_size > _MAX_BYTES:
            raise _fault("point chunk exceeds CPC1 size bounds")
        if info.st_size < _HEADER.size:
            raise _fault("truncated CPC1 header")
        stream = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        with stream:
            raw = stream.read(info.st_size + 1)
        if len(raw) != info.st_size:
            raise _fault("point chunk changed while being read")
    except WorkerFault:
        raise
    except (OSError, ValueError) as error:
        raise _fault(f"unable to read point chunk: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)

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
        vertices.append(
            _validated_vertex(
                vertex,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        )
        offset += _STRIDE
    descriptor = CPCDescriptor(
        relative_path=_relative_name(path),
        point_count=count,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    return CPCContents(descriptor=descriptor, vertices=tuple(vertices))


__all__ = [
    "AtomicWriteReceipt",
    "CPCContents",
    "CPCDescriptor",
    "CPCVertex",
    "atomic_write_bytes",
    "read_cpc",
    "reduce_vertices",
    "rollback_atomic_write",
    "write_cpc",
]
