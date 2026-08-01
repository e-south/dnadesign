"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/events/append.py

Bounded append transactions for USR JSONL event logs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("fcntl is required for event-log append locking") from exc

EVENT_LOCK_FILENAME = ".events.lock"


class EventAppendState(str, Enum):
    """Observable state of an event append that raised."""

    RESTORED = "restored"
    COMMITTED = "committed"
    INDETERMINATE = "indeterminate"


class EventAppendFailure(RuntimeError):
    """Report whether a failed append changed the event log."""

    def __init__(
        self,
        event_path: str | Path,
        *,
        state: EventAppendState,
        detail: str | None = None,
    ) -> None:
        self.event_path = Path(event_path)
        self.state = EventAppendState(state)
        suffix = f" ({detail})" if detail else ""
        super().__init__(f"Event log append {self.state.value}: {self.event_path}{suffix}")


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write one encoded event completely or raise."""

    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("Event log append made no forward progress.")
        remaining = remaining[written:]


def _close_descriptor(descriptor: int) -> None:
    os.close(descriptor)


def _path_matches_descriptor(path: Path, descriptor: int) -> bool:
    try:
        path_stat = path.stat(follow_symlinks=False)
        descriptor_stat = os.fstat(descriptor)
    except OSError:
        return False
    return (path_stat.st_dev, path_stat.st_ino) == (descriptor_stat.st_dev, descriptor_stat.st_ino)


def _restore_prior_length(path: Path, descriptor: int, prior_size: int) -> bool:
    """Restore one failed append while its event-file lock is still held."""

    if not _path_matches_descriptor(path, descriptor):
        return False
    current = os.fstat(descriptor)
    if not stat.S_ISREG(current.st_mode) or current.st_size < prior_size:
        return False
    os.ftruncate(descriptor, prior_size)
    os.fsync(descriptor)
    restored = os.fstat(descriptor)
    return restored.st_size == prior_size and _path_matches_descriptor(path, descriptor)


@contextmanager
def event_log_lock(event_path: str | Path) -> Iterator[None]:
    """Hold the stable sidecar lock shared by appends and replacements."""

    event_path = Path(event_path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = event_path.parent / EVENT_LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    locked = False
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise OSError(f"Event log lock must be one regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        if not _path_matches_descriptor(lock_path, descriptor):
            raise OSError(f"Event log lock identity changed while acquiring it: {lock_path}")
        yield
    finally:
        if locked:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(descriptor)
        except OSError:
            pass


def _append_event_payload_locked(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        try:
            descriptor = os.open(path, flags, 0o600)
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError(f"Event log is not a regular file: {path}")
            prior_size = file_stat.st_size
        except BaseException as prewrite_error:
            raise EventAppendFailure(path, state=EventAppendState.RESTORED) from prewrite_error
        committed = False
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            if not _path_matches_descriptor(path, descriptor):
                raise OSError(f"Event log identity changed during append: {path}")
            committed = True
        except BaseException as append_error:
            if committed:
                raise EventAppendFailure(path, state=EventAppendState.COMMITTED) from append_error
            try:
                restored = _restore_prior_length(path, descriptor, prior_size)
            except BaseException as restore_error:
                raise EventAppendFailure(
                    path,
                    state=EventAppendState.INDETERMINATE,
                    detail=f"rollback failed: {restore_error}",
                ) from append_error
            if not restored:
                raise EventAppendFailure(path, state=EventAppendState.INDETERMINATE) from append_error
            raise EventAppendFailure(path, state=EventAppendState.RESTORED) from append_error
        closing_descriptor = descriptor
        descriptor = -1
        try:
            _close_descriptor(closing_descriptor)
        except BaseException as close_error:
            raise EventAppendFailure(path, state=EventAppendState.COMMITTED) from close_error
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException:
                pass


def append_event_line(event_path: str | Path, encoded: str) -> None:
    """Append one JSONL record or report its committed/restored state.

    Callers that coordinate another mutation may roll that mutation back only
    when a raised :class:`EventAppendFailure` reports ``RESTORED``. A committed
    or indeterminate append must retain the referenced artifact.
    """

    if not encoded or "\n" in encoded or "\r" in encoded:
        raise ValueError("Event log records must be one non-empty JSONL line.")
    payload = f"{encoded}\n".encode("utf-8")
    path = Path(event_path)
    lock_entered = False
    append_completed = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with event_log_lock(path):
            lock_entered = True
            _append_event_payload_locked(path, payload)
            append_completed = True
    except EventAppendFailure:
        raise
    except BaseException as lock_error:
        if not lock_entered:
            state = EventAppendState.RESTORED
        elif append_completed:
            state = EventAppendState.COMMITTED
        else:
            state = EventAppendState.INDETERMINATE
        raise EventAppendFailure(
            path,
            state=state,
            detail=f"event lock failure: {lock_error}",
        ) from lock_error


__all__ = [
    "EventAppendFailure",
    "EventAppendState",
    "EVENT_LOCK_FILENAME",
    "append_event_line",
    "event_log_lock",
]
