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

from dnadesign.usr.src.events.append import EVENT_LOCK_FILENAME, append_event_line


def test_event_append_rejects_symlinked_lock_without_touching_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.lock"
    outside.write_text("keep\n", encoding="utf-8")
    lock_path = tmp_path / EVENT_LOCK_FILENAME
    lock_path.symlink_to(outside)

    with pytest.raises(OSError):
        append_event_line(tmp_path / ".events.log", "{}")

    assert outside.read_text(encoding="utf-8") == "keep\n"
    assert not (tmp_path / ".events.log").exists()


def test_event_append_rejects_hard_linked_lock(tmp_path: Path) -> None:
    outside = tmp_path / "outside.lock"
    outside.write_text("keep\n", encoding="utf-8")
    os.link(outside, tmp_path / EVENT_LOCK_FILENAME)

    with pytest.raises(OSError, match="one regular file"):
        append_event_line(tmp_path / ".events.log", "{}")

    assert outside.read_text(encoding="utf-8") == "keep\n"
    assert not (tmp_path / ".events.log").exists()
