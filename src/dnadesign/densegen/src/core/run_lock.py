"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/run_lock.py

Workspace-scoped run lock for DenseGen execution.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

from .run_paths import ensure_run_meta_dir


class RunLockError(RuntimeError):
    pass


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
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_lock_payload(handle: TextIO, payload: dict[str, object]) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(payload, indent=2, sort_keys=True))
    handle.flush()
    os.fsync(handle.fileno())


def _lock_payload(*, run_id: str) -> dict[str, object]:
    return {
        "run_id": str(run_id),
        "owner_pid": int(os.getpid()),
        "owner_host": socket.gethostname(),
        "job_id": str(os.environ.get("JOB_ID", "")).strip() or None,
        "sge_task_id": str(os.environ.get("SGE_TASK_ID", "")).strip() or None,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _lock_held_error(*, lock_path: Path) -> RunLockError:
    payload = _read_lock_payload(lock_path)
    owner_pid = str(payload.get("owner_pid", "?"))
    owner_host = str(payload.get("owner_host", "?"))
    owner_run_id = str(payload.get("run_id", "?"))
    owner_job_id = str(payload.get("job_id", "-") or "-")
    owner_task_id = str(payload.get("sge_task_id", "-") or "-")
    acquired_at = str(payload.get("acquired_at_utc", "-") or "-")
    return RunLockError(
        "Run lock is held for this workspace. "
        f"lock={lock_path} run_id={owner_run_id} owner_pid={owner_pid} "
        f"owner_host={owner_host} job_id={owner_job_id} sge_task_id={owner_task_id} acquired_at_utc={acquired_at}."
    )


@contextmanager
def acquire_run_lock(*, run_root: Path, run_id: str) -> Iterator[Path]:
    meta_root = ensure_run_meta_dir(Path(run_root))
    lock_path = meta_root / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    lock_acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise _lock_held_error(lock_path=lock_path) from exc

        lock_acquired = True
        _write_lock_payload(handle, _lock_payload(run_id=str(run_id)))
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
