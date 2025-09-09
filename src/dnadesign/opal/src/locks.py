"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/locks.py

Provides a minimal file-based lock to prevent concurrent mutation of a campaign
workspace.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

from .utils import ExitCodes, OpalError


class CampaignLock:
    def __init__(self, workdir: Path):
        self.lock_path = workdir / ".campaign.lock"

    def acquire(self) -> None:
        try:
            # exclusive create; fail if exists
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError as e:
            raise OpalError(
                f"Lock acquisition failed: {self.lock_path}", ExitCodes.LOCK_FAILED
            ) from e

    def release(self) -> None:
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            # don't raise on cleanup
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
