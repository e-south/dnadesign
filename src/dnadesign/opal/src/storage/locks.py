# ABOUTME: Provides campaign-scoped filesystem locks for write operations.
# ABOUTME: Detects stale locks and guides remediation to avoid silent failures.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/locks.py

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


def _read_lock_payload(lockfile: Path) -> dict:
    try:
        payload = json.loads(lockfile.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OpalError(
            f"Campaign lock is unreadable at {lockfile}. Remove the lock file to proceed.",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    if not isinstance(payload, dict):
        raise OpalError(
            f"Campaign lock payload is invalid at {lockfile}. Remove the lock file to proceed.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return payload


class CampaignLock:
    """Very simple file lock to serialize write operations per campaign.
    Not distributed; good enough for single-host workflows.
    """

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.lockfile = self.workdir / ".opal.lock"

    def __enter__(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        # atomic create if not exists
        try:
            fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = {"pid": int(os.getpid()), "ts": now_iso()}
            os.write(fd, json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
            os.close(fd)
        except FileExistsError as e:
            if not self.lockfile.exists():
                raise OpalError(
                    f"Campaign lock disappeared during acquisition: {self.lockfile}",
                    ExitCodes.CONTRACT_VIOLATION,
                ) from e
            payload = _read_lock_payload(self.lockfile)
            pid = payload.get("pid")
            ts = payload.get("ts")
            if isinstance(pid, int) and not _pid_is_running(pid):
                raise OpalError(
                    f"Detected stale lock at {self.lockfile} (pid {pid}, ts {ts}). Remove the lock file to proceed.",
                    ExitCodes.CONTRACT_VIOLATION,
                ) from e
            raise OpalError(
                f"Campaign is locked by another process (pid {pid}, ts {ts}): {self.lockfile}",
                ExitCodes.CONTRACT_VIOLATION,
            ) from e
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.lockfile.unlink(missing_ok=True)
        except Exception:
            pass
        return False
