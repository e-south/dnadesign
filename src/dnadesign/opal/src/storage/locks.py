"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/locks.py

Provides filesystem locks for write operations. Detects stale locks and guides.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..core.utils import ExitCodes, OpalError, now_iso


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _read_lock_payload(lockfile: Path, *, subject: str) -> dict:
    try:
        payload = json.loads(lockfile.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OpalError(
            f"{subject} lock is unreadable at {lockfile}. Remove the lock file to proceed.",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    if not isinstance(payload, dict):
        raise OpalError(
            f"{subject} lock payload is invalid at {lockfile}. Remove the lock file to proceed.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return payload


def _acquire_lock_file(lockfile: Path, *, subject: str, payload_extra: dict | None = None) -> None:
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": int(os.getpid()), "ts": now_iso()}
    if payload_extra:
        payload.update(payload_extra)

    try:
        fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(
                fd,
                json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"),
            )
        finally:
            os.close(fd)
    except FileExistsError as exc:
        if not lockfile.exists():
            raise OpalError(
                f"{subject} lock disappeared during acquisition: {lockfile}",
                ExitCodes.CONTRACT_VIOLATION,
            ) from exc
        existing = _read_lock_payload(lockfile, subject=subject)
        pid = existing.get("pid")
        ts = existing.get("ts")
        if isinstance(pid, int) and not _pid_is_running(pid):
            raise OpalError(
                f"Detected stale lock for {subject} at {lockfile} (pid {pid}, ts {ts}). "
                "Remove the lock file to proceed.",
                ExitCodes.CONTRACT_VIOLATION,
            ) from exc
        raise OpalError(
            f"{subject} is locked by another process (pid {pid}, ts {ts}): {lockfile}",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc


class CampaignLock:
    """Very simple file lock to serialize write operations per campaign.
    Not distributed; good enough for single-host workflows.
    """

    def __init__(self, workdir: Path, *, payload_extra: dict | None = None):
        self.workdir = Path(workdir)
        self.lockfile = self.workdir / ".opal.lock"
        self.payload_extra = dict(payload_extra or {})
        self._acquired = False

    @property
    def acquired(self) -> bool:
        return self._acquired

    def acquire(self) -> CampaignLock:
        if self._acquired:
            raise OpalError(
                f"Campaign lock is already held by this instance: {self.lockfile}",
                ExitCodes.CONTRACT_VIOLATION,
            )
        _acquire_lock_file(self.lockfile, subject="Campaign", payload_extra=self.payload_extra)
        self._acquired = True
        return self

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self.lockfile.unlink(missing_ok=True)
        except Exception:
            pass
        finally:
            self._acquired = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def inspect_campaign_lock(workdir: Path) -> dict:
    lockfile = Path(workdir) / ".opal.lock"
    payload = None
    active = False
    stale = False
    unreadable = False
    if lockfile.exists():
        try:
            payload = _read_lock_payload(lockfile, subject="Campaign")
            pid = payload.get("pid")
            active = isinstance(pid, int) and _pid_is_running(pid)
            stale = isinstance(pid, int) and not active
        except OpalError:
            unreadable = True
    return {
        "schema_version": "opal.lock_state.v1",
        "scope": "local_host",
        "lockfile": str(lockfile),
        "exists": lockfile.exists(),
        "active": active,
        "stale": stale,
        "unreadable": unreadable,
        "payload": payload,
    }


class PathLock:
    """Serialize writes to a single shared file path on the local host."""

    def __init__(self, target: Path, *, lock_name: str = "Path"):
        self.target = Path(target)
        self.subject = str(lock_name)
        self.lockfile = self.target.with_name(f".{self.target.name}.lock")

    def __enter__(self):
        _acquire_lock_file(
            self.lockfile,
            subject=self.subject,
            payload_extra={"target": str(self.target)},
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.lockfile.unlink(missing_ok=True)
        except Exception:
            pass
        return False
