"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/cursor/iteration.py

Follow-loop line iteration with truncate and idle-timeout handling for notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

from ...errors import NotifyConfigError


def iter_file_lines(
    events_path: Path,
    *,
    start_offset: int,
    on_truncate: str,
    follow: bool,
    wait_for_events: bool = False,
    idle_timeout_seconds: float | None = None,
    poll_interval_seconds: float = 0.2,
) -> Iterable[tuple[int, str]]:
    timeout_value = None
    if idle_timeout_seconds is not None:
        timeout_value = float(idle_timeout_seconds)
        if timeout_value <= 0:
            raise NotifyConfigError("idle_timeout must be > 0 when provided")
    poll_interval = float(poll_interval_seconds)
    if poll_interval <= 0:
        raise NotifyConfigError("poll_interval_seconds must be > 0")
    started_at = time.monotonic()
    if not events_path.exists():
        if not follow or not wait_for_events:
            raise NotifyConfigError(f"events file not found: {events_path}")
        while not events_path.exists():
            if timeout_value is not None and (time.monotonic() - started_at) >= timeout_value:
                return
            time.sleep(poll_interval)
    mode = str(on_truncate or "").strip().lower()
    if mode not in {"error", "restart"}:
        raise NotifyConfigError(f"unsupported on-truncate mode '{on_truncate}'")
    size = int(events_path.stat().st_size)
    offset = int(start_offset)
    if offset > size:
        if mode == "restart":
            offset = 0
        else:
            raise NotifyConfigError(
                f"cursor offset exceeds events file size: offset={offset} size={size}. "
                "If the file was truncated or rotated, pass --on-truncate restart."
            )

    handle = events_path.open("r", encoding="utf-8")
    try:
        last_activity = time.monotonic()
        handle.seek(offset)
        while True:
            line = handle.readline()
            if line:
                last_activity = time.monotonic()
                yield handle.tell(), line
                continue
            if not follow:
                return
            if timeout_value is not None and (time.monotonic() - last_activity) >= timeout_value:
                return
            time.sleep(poll_interval)
            if not events_path.exists():
                if timeout_value is not None and (time.monotonic() - last_activity) >= timeout_value:
                    return
                continue
            try:
                path_stat = events_path.stat()
            except FileNotFoundError:
                if timeout_value is not None and (time.monotonic() - last_activity) >= timeout_value:
                    return
                continue
            handle_stat = os.fstat(handle.fileno())
            handle_pos = int(handle.tell())
            path_changed = (int(path_stat.st_dev), int(path_stat.st_ino)) != (
                int(handle_stat.st_dev),
                int(handle_stat.st_ino),
            )
            truncated = handle_pos > int(path_stat.st_size)
            if not path_changed and not truncated:
                continue
            if mode == "error":
                reason = (
                    "events file was replaced while following"
                    if path_changed
                    else f"events file was truncated while following (pos={handle_pos} size={int(path_stat.st_size)})"
                )
                raise NotifyConfigError(f"{reason}. Pass --on-truncate restart to resume from start.")
            handle.close()
            handle = events_path.open("r", encoding="utf-8")
            handle.seek(0)
    finally:
        if not handle.closed:
            handle.close()
