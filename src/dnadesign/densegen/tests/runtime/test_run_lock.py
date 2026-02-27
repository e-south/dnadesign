"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_run_lock.py

Concurrency guard tests for DenseGen workspace run locking.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest

from dnadesign.densegen.src.core.run_lock import RunLockError, acquire_run_lock


def _hold_run_lock(run_root: str, ready_queue: multiprocessing.Queue) -> None:
    path = Path(run_root)
    with acquire_run_lock(run_root=path, run_id="demo"):
        ready_queue.put("locked")
        time.sleep(1.0)


def test_acquire_run_lock_rejects_concurrent_holder(tmp_path: Path) -> None:
    run_root = tmp_path / "workspace"
    run_root.mkdir(parents=True, exist_ok=True)
    ready_queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_hold_run_lock, args=(str(run_root), ready_queue))
    proc.start()
    try:
        assert ready_queue.get(timeout=5) == "locked"
        with pytest.raises(RunLockError, match="Run lock is held for this workspace"):
            with acquire_run_lock(run_root=run_root, run_id="demo"):
                pass
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        assert proc.exitcode == 0

    with acquire_run_lock(run_root=run_root, run_id="demo"):
        pass


def test_failed_lock_attempt_does_not_drop_active_lock_path(tmp_path: Path) -> None:
    run_root = tmp_path / "workspace"
    run_root.mkdir(parents=True, exist_ok=True)
    ready_queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_hold_run_lock, args=(str(run_root), ready_queue))
    proc.start()
    try:
        assert ready_queue.get(timeout=5) == "locked"
        with pytest.raises(RunLockError, match="Run lock is held for this workspace"):
            with acquire_run_lock(run_root=run_root, run_id="demo"):
                pass
        with pytest.raises(RunLockError, match="Run lock is held for this workspace"):
            with acquire_run_lock(run_root=run_root, run_id="demo"):
                pass
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        assert proc.exitcode == 0
