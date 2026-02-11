"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/src/locks.py

Filesystem-based write locks for USR dataset mutations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError as e:  # pragma: no cover
    raise RuntimeError("fcntl is required for dataset write locking") from e

LOCK_FILENAME = ".usr.lock"


@contextmanager
def dataset_write_lock(dataset_dir: Path):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    lock_path = dataset_dir / LOCK_FILENAME
    with lock_path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
