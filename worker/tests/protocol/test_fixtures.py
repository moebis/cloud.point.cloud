from __future__ import annotations

import json
from pathlib import Path

from cloudpoint_worker.protocol.fixtures import (
    build_protocol_fixture,
    write_protocol_fixture,
)
from cloudpoint_worker.protocol.framing import MAX_MESSAGE_BYTES


def test_fixture_is_deterministic_and_framed_rows_match_json(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_protocol_fixture(first)
    write_protocol_fixture(second)

    assert first.read_bytes() == second.read_bytes()
    fixture = json.loads(first.read_bytes())
    assert fixture["protocolVersion"] == 1
    assert fixture["maximumMessageBytes"] == MAX_MESSAGE_BYTES
    for row in fixture["messages"]:
        framed = bytes(row["framedBytes"])
        body = row["json"].encode()
        assert int.from_bytes(framed[:4], "big") == len(body)
        assert framed[4:] == body


def test_fixture_covers_every_command_response_event_and_rejection() -> None:
    fixture = build_protocol_fixture()
    names = [row["name"] for row in fixture["messages"]]

    for command in [
        "hello",
        "configure",
        "beginSession-null",
        "beginSession-full",
        "enqueueFrame",
        "finishInput",
        "pause",
        "resume",
        "cancel",
        "shutdown",
    ]:
        assert f"command.{command}" in names
    for response in ["ack", "command-error", "asynchronous-error"]:
        assert f"response.{response}" in names
    for event in [
        "ready",
        "modelProgress-validating",
        "modelProgress-loading",
        "frameStarted",
        "frameCompleted",
        "windowCompleted",
        "sessionCompleted",
        "paused",
        "cancelled-null",
        "cancelled-full",
        "warning",
        "heartbeat-null",
        "heartbeat-full",
    ]:
        assert f"event.{event}" in names

    rejection_names = {row["name"] for row in fixture["rejections"]}
    assert {
        "nested-checkpoint-missing",
        "nested-checkpoint-unknown",
        "nested-configuration-missing",
        "nested-configuration-unknown",
        "uppercase-command-uuid",
        "unsupported-version",
        "zero-length",
        "oversized-length",
        "truncated-frame",
        "invalid-json",
    } <= rejection_names


def test_committed_fixture_matches_generator() -> None:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "protocol-v1.json"
    assert (
        fixture_path.read_bytes()
        == json.dumps(
            build_protocol_fixture(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
