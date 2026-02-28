"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/cursor/locking.py

Cursor lock acquisition and stale-lock recovery for notify watch loops.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from ...errors import NotifyConfigError


def cursor_lock_path(cursor_path: Path) -> Path:
    return cursor_path.with_name(f"{cursor_path.name}.lock")


def pid_is_running(pid: int) -> bool:
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
    lock_path = cursor_lock_path(cursor_path)
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
            if holder_pid is not None and pid_is_running(holder_pid):
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
