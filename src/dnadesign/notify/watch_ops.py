"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/watch_ops.py

Event-watch cursor and follow-loop primitives for notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .errors import NotifyConfigError


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


def _cursor_lock_path(cursor_path: Path) -> Path:
    return cursor_path.with_name(f"{cursor_path.name}.lock")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def acquire_cursor_lock(cursor_path: Path | None) -> Iterable[None]:
    if cursor_path is None:
        yield
        return
    lock_path = _cursor_lock_path(cursor_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    acquired = False
    for _attempt in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            holder_pid: int | None = None
            try:
                body = json.loads(lock_path.read_text(encoding="utf-8"))
                if isinstance(body, dict):
                    pid_raw = body.get("pid")
                    if isinstance(pid_raw, int):
                        holder_pid = pid_raw
            except (json.JSONDecodeError, OSError):
                holder_pid = None
            if holder_pid is not None and _pid_is_running(holder_pid):
                raise NotifyConfigError(f"cursor lock already held: {lock_path} (pid={holder_pid})")
            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise NotifyConfigError(f"failed to remove stale cursor lock: {lock_path}") from exc
            continue
        except OSError as exc:
            raise NotifyConfigError(f"failed to create cursor lock: {lock_path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle)
        acquired = True
        break

    if not acquired:
        raise NotifyConfigError(f"failed to acquire cursor lock: {lock_path}")

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise NotifyConfigError(f"failed to release cursor lock: {lock_path}") from exc


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
