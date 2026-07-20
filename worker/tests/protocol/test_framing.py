from __future__ import annotations

import io
import math

import pytest

from cloudpoint_worker.errors import WorkerFault
from cloudpoint_worker.protocol.framing import (
    MAX_MESSAGE_BYTES,
    RawJSONNumber,
    encode_canonical_json,
    read_json_frame,
    write_json_frame,
)


class FragmentedReader:
    def __init__(self, data: bytes, fragment_size: int = 1) -> None:
        self._stream = io.BytesIO(data)
        self._fragment_size = fragment_size

    def read(self, count: int) -> bytes:
        return self._stream.read(min(count, self._fragment_size))


class RecordingReader:
    def __init__(self, data: bytes) -> None:
        self._stream = io.BytesIO(data)
        self.requested: list[int] = []

    def read(self, count: int) -> bytes:
        self.requested.append(count)
        return self._stream.read(count)


class ShortWriter:
    def __init__(self, maximum_write: int) -> None:
        self.maximum_write = maximum_write
        self.bytes = bytearray()
        self.flush_count = 0

    def write(self, data: bytes | memoryview) -> int:
        count = min(len(data), self.maximum_write)
        self.bytes.extend(data[:count])
        return count

    def flush(self) -> None:
        self.flush_count += 1


def test_frame_is_big_endian_canonical_and_round_trips() -> None:
    stream = io.BytesIO()
    write_json_frame(stream, {"type": "hello", "protocolVersion": 1})
    raw = stream.getvalue()

    assert int.from_bytes(raw[:4], "big") == len(raw) - 4
    assert raw[4:] == b'{"protocolVersion":1,"type":"hello"}'
    stream.seek(0)
    assert read_json_frame(stream) == {"protocolVersion": 1, "type": "hello"}


def test_fragmented_header_and_body_are_read_exactly() -> None:
    body = b'{"payload":{},"protocolVersion":1}'
    framed = len(body).to_bytes(4, "big") + body
    assert read_json_frame(FragmentedReader(framed)) == {
        "payload": {},
        "protocolVersion": 1,
    }


def test_frame_rejects_more_than_one_megabyte_before_body_read() -> None:
    stream = RecordingReader((MAX_MESSAGE_BYTES + 1).to_bytes(4, "big"))
    with pytest.raises(WorkerFault, match="MESSAGE_TOO_LARGE"):
        read_json_frame(stream)
    assert stream.requested == [4]


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"\0\0\0\0", "INVALID_MESSAGE_LENGTH"),
        (b"\0\0", "TRUNCATED_MESSAGE"),
        ((3).to_bytes(4, "big") + b"{}", "TRUNCATED_MESSAGE"),
        ((1).to_bytes(4, "big") + b"{", "INVALID_JSON"),
    ],
)
def test_invalid_frames_raise_stable_faults(data: bytes, code: str) -> None:
    with pytest.raises(WorkerFault, match=code):
        read_json_frame(io.BytesIO(data))


def test_write_all_handles_short_writes_and_flushes_once() -> None:
    stream = ShortWriter(maximum_write=3)
    write_json_frame(stream, {"ok": True})

    assert stream.flush_count == 1
    assert bytes(stream.bytes) == b'\x00\x00\x00\x0b{"ok":true}'


def test_write_rejects_closed_peer_and_oversized_body() -> None:
    class ClosedWriter:
        def write(self, data: bytes | memoryview) -> int:
            return 0

        def flush(self) -> None:
            raise AssertionError("flush must not run")

    with pytest.raises(WorkerFault, match="TRUNCATED_MESSAGE"):
        write_json_frame(ClosedWriter(), {})  # type: ignore[arg-type]

    with pytest.raises(WorkerFault, match="MESSAGE_TOO_LARGE"):
        write_json_frame(io.BytesIO(), {"value": "x" * MAX_MESSAGE_BYTES})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.0, b"0"),
        (1.0, b"1"),
        (1.25, b"1.25"),
        (1e100, b"1e100"),
        (1e-7, b"1e-7"),
        (1.2345678901234567, b"1.2345678901234567"),
    ],
)
def test_typed_doubles_use_canonical_tokens(value: float, expected: bytes) -> None:
    assert encode_canonical_json(value) == expected


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_are_never_encoded(value: float) -> None:
    with pytest.raises(WorkerFault, match="INVALID_JSON_NUMBER"):
        encode_canonical_json({"value": value})


def test_error_details_preserve_raw_numeric_lexemes_byte_for_byte() -> None:
    body = (
        b'{"commandId":null,"id":"00000000-0000-0000-0000-000000000001",'
        b'"payload":{"code":"probe","details":{"decimal":1.2300e+04,'
        b'"nested":[18446744073709551615,-0]},"message":"probe",'
        b'"recoverable":true},"projectId":"00000000-0000-0000-0000-000000000002",'
        b'"protocolVersion":1,"type":"error"}'
    )
    value = read_json_frame(io.BytesIO(len(body).to_bytes(4, "big") + body))

    details = value["payload"]["details"]  # type: ignore[index]
    assert details["decimal"] == RawJSONNumber("1.2300e+04")  # type: ignore[index]
    assert details["nested"] == [  # type: ignore[index]
        RawJSONNumber("18446744073709551615"),
        RawJSONNumber("-0"),
    ]
    assert encode_canonical_json(value) == body


def test_exact_number_exceptions_are_rejected_outside_error_details() -> None:
    with pytest.raises(WorkerFault, match="INVALID_JSON_NUMBER"):
        encode_canonical_json({"measurement": RawJSONNumber("1.2300")})


def test_duplicate_object_keys_and_non_object_frames_are_rejected() -> None:
    for body in [b'{"id":1,"id":2}', b"[]", b"NaN"]:
        frame = len(body).to_bytes(4, "big") + body
        with pytest.raises(WorkerFault, match="INVALID_JSON"):
            read_json_frame(io.BytesIO(frame))


@pytest.mark.parametrize("token", ["1e10000", "1" + ("0" * 5_000)])
def test_valid_overflowing_number_remains_available_for_typed_schema_rejection(
    token: str,
) -> None:
    body = (
        b'{"id":"00000000-0000-0000-0000-000000000001",'
        b'"payload":{"frameIndex":1,"relativePath":"Frames/00000001.jpg",'
        + b'"sourceTimestamp":'
        + token.encode("ascii")
        + b"},"
        b'"projectId":"00000000-0000-0000-0000-000000000002",'
        b'"protocolVersion":1,"type":"enqueueFrame"}'
    )
    value = read_json_frame(io.BytesIO(len(body).to_bytes(4, "big") + body))
    assert value["payload"]["sourceTimestamp"] == RawJSONNumber(token)  # type: ignore[index]
