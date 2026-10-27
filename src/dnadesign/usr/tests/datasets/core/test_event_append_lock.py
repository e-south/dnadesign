"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/core/test_event_append_lock.py

Security contracts for the stable USR event-log sidecar lock.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dnadesign.usr.src.events.append import (
    EVENT_LOCK_FILENAME,
    MAX_EVENT_RECORD_BYTES,
    EventAppendFailure,
    EventAppendState,
    append_event_line,
    append_event_payload,
    encode_event_line,
)


def _event_payload_with_size(size_bytes: int) -> bytes:
    prefix = b'{"value":"'
    suffix = b'"}\n'
    return prefix + (b"x" * (size_bytes - len(prefix) - len(suffix))) + suffix


def test_event_append_rejects_symlinked_lock_without_touching_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.lock"
    outside.write_text("keep\n", encoding="utf-8")
    lock_path = tmp_path / EVENT_LOCK_FILENAME
    lock_path.symlink_to(outside)

    with pytest.raises(EventAppendFailure, match="restored") as exc_info:
        append_event_line(tmp_path / ".events.log", "{}")

    assert exc_info.value.state is EventAppendState.RESTORED
    assert outside.read_text(encoding="utf-8") == "keep\n"
    assert not (tmp_path / ".events.log").exists()


def test_event_append_rejects_hard_linked_lock(tmp_path: Path) -> None:
    outside = tmp_path / "outside.lock"
    outside.write_text("keep\n", encoding="utf-8")
    os.link(outside, tmp_path / EVENT_LOCK_FILENAME)

    with pytest.raises(EventAppendFailure, match="restored") as exc_info:
        append_event_line(tmp_path / ".events.log", "{}")

    assert exc_info.value.state is EventAppendState.RESTORED
    assert outside.read_text(encoding="utf-8") == "keep\n"
    assert not (tmp_path / ".events.log").exists()


def test_event_encoder_accepts_a_record_at_the_encoded_size_limit() -> None:
    payload = _event_payload_with_size(MAX_EVENT_RECORD_BYTES)

    assert encode_event_line(payload[:-1].decode("utf-8")) == payload


def test_event_encoder_rejects_a_valid_record_over_the_encoded_size_limit() -> None:
    payload = _event_payload_with_size(MAX_EVENT_RECORD_BYTES + 1)

    with pytest.raises(ValueError, match="must not exceed"):
        encode_event_line(payload[:-1].decode("utf-8"))


def test_prepared_event_append_rejects_an_oversized_record_before_mutation(tmp_path: Path) -> None:
    event_path = tmp_path / ".events.log"

    with pytest.raises(ValueError, match="must not exceed"):
        append_event_payload(event_path, b"x" * (MAX_EVENT_RECORD_BYTES + 1))

    assert not event_path.exists()
    assert not (tmp_path / EVENT_LOCK_FILENAME).exists()
