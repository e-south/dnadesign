"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_run_lock.py

Concurrency guard tests for DenseGen workspace run locking.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from dnadesign.densegen.src.core.run_lock import RunLockError, acquire_run_lock


def _start_run_lock_holder(run_root: Path) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        "-c",
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            from dnadesign.densegen.src.core.run_lock import acquire_run_lock

            run_root = Path(sys.argv[1])
            with acquire_run_lock(run_root=run_root, run_id="demo"):
                print("locked", flush=True)
                sys.stdin.read()
            """
        ),
        str(run_root),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _assert_holder_ready(proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    ready = proc.stdout.readline().strip()
    if ready != "locked":
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        raise AssertionError(f"run-lock holder failed to start: stdout={ready!r} stderr={stderr!r}")


def _stop_run_lock_holder(proc: subprocess.Popen[str]) -> None:
    if proc.stdin is not None and not proc.stdin.closed:
        proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=5)
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    assert proc.returncode == 0, stderr


def test_acquire_run_lock_rejects_concurrent_holder(tmp_path: Path) -> None:
    run_root = tmp_path / "workspace"
    run_root.mkdir(parents=True, exist_ok=True)
    proc = _start_run_lock_holder(run_root)
    try:
        _assert_holder_ready(proc)
        with pytest.raises(RunLockError, match="Run lock is held for this workspace"):
            with acquire_run_lock(run_root=run_root, run_id="demo"):
                pass
    finally:
        _stop_run_lock_holder(proc)

    with acquire_run_lock(run_root=run_root, run_id="demo"):
        pass


def test_failed_lock_attempt_does_not_drop_active_lock_path(tmp_path: Path) -> None:
    run_root = tmp_path / "workspace"
    run_root.mkdir(parents=True, exist_ok=True)
    proc = _start_run_lock_holder(run_root)
    try:
        _assert_holder_ready(proc)
        with pytest.raises(RunLockError, match="Run lock is held for this workspace"):
            with acquire_run_lock(run_root=run_root, run_id="demo"):
                pass
        with pytest.raises(RunLockError, match="Run lock is held for this workspace"):
            with acquire_run_lock(run_root=run_root, run_id="demo"):
                pass
    finally:
        _stop_run_lock_holder(proc)
