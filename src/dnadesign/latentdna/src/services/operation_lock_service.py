"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/services/operation_lock_service.py

Workspace-scoped operation locks for heavy latentdna tasks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

try:
    import fcntl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("fcntl is required for latentdna operation locking") from exc

from ..contracts.errors import OperationLockError


def operation_lock_path(output_root: Path, *, operation: str) -> Path:
    return Path(output_root) / ".locks" / f"{operation}.lock"


def _read_lock_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_lock_payload(handle: TextIO, payload: dict[str, object]) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(payload, indent=2, sort_keys=True))
    handle.flush()
    os.fsync(handle.fileno())


def _lock_payload(*, operation: str, owner_id: str) -> dict[str, object]:
    return {
        "operation": operation,
        "owner_id": owner_id,
        "owner_pid": int(os.getpid()),
        "owner_host": socket.gethostname(),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _lock_held_error(*, operation: str, lock_path: Path) -> OperationLockError:
    payload = _read_lock_payload(lock_path)
    owner_id = str(payload.get("owner_id", "?"))
    owner_pid = str(payload.get("owner_pid", "?"))
    owner_host = str(payload.get("owner_host", "?"))
    acquired_at = str(payload.get("acquired_at_utc", "-") or "-")
    if operation == "projection_fit":
        message = (
            "another projection fit is already in progress for this workspace; "
            "latentdna serializes heavy projection fits to avoid aggregate memory pressure."
        )
    elif operation == "view_materialize":
        message = (
            "another view materialize is already in progress for this workspace; "
            "latentdna serializes heavy view materializations to avoid aggregate memory pressure."
        )
    else:
        message = (
            f"another {operation} operation is already in progress for this workspace; "
            "wait for the active operation to finish before starting a second copy."
        )
    return OperationLockError(
        f"{message} lock={lock_path} owner_id={owner_id} "
        f"owner_pid={owner_pid} owner_host={owner_host} acquired_at_utc={acquired_at}"
    )


@contextmanager
def acquire_workspace_operation_lock(
    output_root: Path,
    *,
    operation: str,
    owner_id: str,
) -> Iterator[Path]:
    lock_path = operation_lock_path(output_root, operation=operation)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    lock_acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise _lock_held_error(operation=operation, lock_path=lock_path) from exc

        lock_acquired = True
        _write_lock_payload(handle, _lock_payload(operation=operation, owner_id=owner_id))
        yield lock_path
    finally:
        if lock_acquired:
            try:
                handle.seek(0)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                pass
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            handle.close()
            lock_path.unlink(missing_ok=True)
        else:
            handle.close()
