"""Bounded length-prefixed canonical JSON framing for protocol version 1."""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import BinaryIO

import msgspec

from cloudpoint_worker.errors import WorkerFault

MAX_MESSAGE_BYTES = 1_048_576

_JSON_NUMBER_PATTERN = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"
)


@dataclass(frozen=True, slots=True)
class RawJSONNumber:
    """A validated numeric token retained only inside decoded error details."""

    token: str

    def __post_init__(self) -> None:
        if not _JSON_NUMBER_PATTERN.fullmatch(self.token):
            raise ValueError(f"invalid JSON number token: {self.token!r}")


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count - len(chunks))
        if not chunk:
            raise WorkerFault("TRUNCATED_MESSAGE", "peer closed mid-frame", False)
        chunks.extend(chunk)
    return bytes(chunks)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(token: str) -> object:
    raise ValueError(f"nonstandard JSON number: {token}")


def _materialize_number(value: RawJSONNumber) -> int | float | RawJSONNumber:
    token = value.token
    if "." not in token and "e" not in token.lower():
        try:
            return int(token)
        except ValueError:
            # CPython caps decimal-to-int conversion length. The token is
            # still valid JSON, so typed schema validation owns rejection.
            return value
    result = float(token)
    if not math.isfinite(result):
        # The token is valid JSON even when it cannot fit in a typed Double.
        # Preserve it long enough for schema validation to reject the payload
        # while retaining a recoverable command header and open transport.
        return value
    return result


def _materialize_decoded_numbers(value: object, *, preserve: bool = False) -> object:
    if isinstance(value, RawJSONNumber):
        return value if preserve else _materialize_number(value)
    if isinstance(value, list):
        return [_materialize_decoded_numbers(item, preserve=preserve) for item in value]
    if isinstance(value, dict):
        return {
            key: _materialize_decoded_numbers(item, preserve=preserve)
            for key, item in value.items()
        }
    return value


def _decode_json_object(body: bytes) -> dict[str, object]:
    try:
        text = body.decode("utf-8", errors="strict")
        decoded = json.loads(
            text,
            parse_int=RawJSONNumber,
            parse_float=RawJSONNumber,
            parse_constant=_reject_nonstandard_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(decoded, dict):
            raise ValueError("a protocol frame must contain a JSON object")

        preserve_details = decoded.get("type") in {"error", "warning"}
        result: dict[str, object] = {}
        for key, value in decoded.items():
            if key == "payload" and preserve_details and isinstance(value, dict):
                payload: dict[str, object] = {}
                for payload_key, payload_value in value.items():
                    payload[payload_key] = _materialize_decoded_numbers(
                        payload_value,
                        preserve=payload_key == "details",
                    )
                result[key] = payload
            else:
                result[key] = _materialize_decoded_numbers(value)
        return result
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        OverflowError,
    ) as error:
        raise WorkerFault(
            "INVALID_JSON", "frame body is not one unambiguous JSON object", False
        ) from error


def read_json_frame(stream: BinaryIO) -> dict[str, object]:
    """Read exactly one bounded four-byte big-endian JSON frame."""

    length = int.from_bytes(_read_exact(stream, 4), "big")
    if length == 0:
        raise WorkerFault("INVALID_MESSAGE_LENGTH", "zero-length JSON frame", False)
    if length > MAX_MESSAGE_BYTES:
        raise WorkerFault(
            "MESSAGE_TOO_LARGE", f"{length} exceeds {MAX_MESSAGE_BYTES}", False
        )
    return _decode_json_object(_read_exact(stream, length))


def _canonical_float(value: float) -> str:
    if not math.isfinite(value):
        raise WorkerFault("INVALID_JSON_NUMBER", "JSON numbers must be finite", False)
    if value == 0:
        return "0"

    token = repr(value).lower()
    if "e" in token:
        mantissa, exponent = token.split("e", 1)
        if mantissa.endswith(".0"):
            mantissa = mantissa[:-2]
        return f"{mantissa}e{int(exponent)}"
    if token.endswith(".0"):
        return token[:-2]
    return token


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise WorkerFault("INVALID_JSON_NUMBER", "JSON numbers must be finite", False)
    token = str(value)
    if not _JSON_NUMBER_PATTERN.fullmatch(token):
        raise WorkerFault(
            "INVALID_JSON_NUMBER", f"invalid Decimal token: {token}", False
        )
    return token


def _struct_to_builtins(value: msgspec.Struct) -> object:
    # Walking the encoded field names ourselves is intentional. ``msgspec``
    # recursively lowers arbitrary dataclasses, which would turn
    # ``RawJSONNumber`` into ``{"token": ...}`` before the details-only exact
    # number rule can see it.
    return {
        encoded_name: getattr(value, field_name)
        for field_name, encoded_name in zip(
            value.__struct_fields__,
            value.__struct_encode_fields__,
            strict=True,
        )
    }


def _encode_value(
    value: object,
    *,
    exact_detail_numbers: bool = False,
    error_payload: bool = False,
) -> bytes:
    if isinstance(value, msgspec.Struct):
        return _encode_value(
            _struct_to_builtins(value),
            exact_detail_numbers=exact_detail_numbers,
            error_payload=error_payload,
        )
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, RawJSONNumber):
        if not exact_detail_numbers:
            raise WorkerFault(
                "INVALID_JSON_NUMBER",
                "raw numeric tokens are restricted to error details",
                False,
            )
        return value.token.encode("ascii")
    if isinstance(value, uuid.UUID):
        return _encode_value(str(value))
    if isinstance(value, Decimal):
        if not exact_detail_numbers:
            raise WorkerFault(
                "INVALID_JSON_NUMBER",
                "Decimal tokens are restricted to error details",
                False,
            )
        return _canonical_decimal(value).encode("ascii")
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, float):
        return _canonical_float(value).encode("ascii")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise WorkerFault("INVALID_JSON", "JSON object keys must be strings", False)
        is_error_envelope = value.get("type") in {"error", "warning"}
        entries = [
            _encode_value(key)
            + b":"
            + _encode_value(
                value[key],
                exact_detail_numbers=(
                    exact_detail_numbers or (error_payload and key == "details")
                ),
                error_payload=(is_error_envelope and key == "payload"),
            )
            for key in sorted(value)
        ]
        return b"{" + b",".join(entries) + b"}"
    if isinstance(value, Sequence) and not isinstance(
        value, bytes | bytearray | memoryview
    ):
        return (
            b"["
            + b",".join(
                _encode_value(item, exact_detail_numbers=exact_detail_numbers)
                for item in value
            )
            + b"]"
        )
    raise WorkerFault(
        "INVALID_JSON", f"unsupported JSON value: {type(value).__name__}", False
    )


def encode_canonical_json(value: object) -> bytes:
    """Encode sorted, compact protocol JSON with typed-Double normalization."""

    return _encode_value(value)


def _write_all(stream: BinaryIO, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise WorkerFault("TRUNCATED_MESSAGE", "peer closed during write", False)
        remaining = remaining[written:]


def write_json_frame(stream: BinaryIO, value: object) -> None:
    """Write and flush exactly one canonical bounded JSON frame."""

    body = encode_canonical_json(value)
    if len(body) > MAX_MESSAGE_BYTES:
        raise WorkerFault(
            "MESSAGE_TOO_LARGE", f"{len(body)} exceeds {MAX_MESSAGE_BYTES}", False
        )
    _write_all(stream, len(body).to_bytes(4, "big") + body)
    stream.flush()


__all__ = [
    "MAX_MESSAGE_BYTES",
    "RawJSONNumber",
    "encode_canonical_json",
    "read_json_frame",
    "write_json_frame",
]
