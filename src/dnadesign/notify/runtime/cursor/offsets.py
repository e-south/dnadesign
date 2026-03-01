"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/cursor/offsets.py

Cursor offset read/write helpers for notify watch loops.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ...errors import NotifyConfigError


def load_cursor_offset(cursor_path: Path | None) -> int:
    if cursor_path is None or not cursor_path.exists():
        return 0
    raw = cursor_path.read_text(encoding="utf-8").strip()
    if not raw:
        return 0
    try:
        offset = int(raw)
    except ValueError as exc:
        raise NotifyConfigError(f"cursor file must contain an integer byte offset: {cursor_path}") from exc
    if offset < 0:
        raise NotifyConfigError(f"cursor offset must be >= 0: {cursor_path}")
    return offset


def save_cursor_offset(cursor_path: Path | None, offset: int) -> None:
    if cursor_path is None:
        return
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_path.write_text(str(int(offset)), encoding="utf-8")
    try:
        cursor_path.chmod(0o600)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on cursor file: {cursor_path}") from exc
