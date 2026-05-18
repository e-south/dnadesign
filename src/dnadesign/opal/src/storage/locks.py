"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/locks.py

Provides filesystem locks for write operations. Detects stale locks and guides
remediation to avoid silent failures.

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

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.lockfile = self.workdir / ".opal.lock"

    def __enter__(self):
        _acquire_lock_file(self.lockfile, subject="Campaign")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.lockfile.unlink(missing_ok=True)
        except Exception:
            pass
        return False


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
