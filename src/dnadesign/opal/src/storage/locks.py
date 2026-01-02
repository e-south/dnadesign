"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/locks.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core.utils import ExitCodes, OpalError


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
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
        except FileExistsError as e:
            raise OpalError(
                f"Campaign is locked by another process: {self.lockfile}",
                ExitCodes.CONTRACT_VIOLATION,
            ) from e
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.lockfile.unlink(missing_ok=True)
        except Exception:
            pass
        return False
