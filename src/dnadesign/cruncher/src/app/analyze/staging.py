"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/staging.py

Stage analysis outputs safely while coordinating process-level analyze locks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from dnadesign.cruncher.analysis.layout import analysis_state_root, summary_path

ANALYZE_LOCK_META_FILE = ".analyze_lock.json"


def prepare_analysis_root(analysis_root_path: Path, *, analysis_id: str) -> Path | None:
    if not analysis_root_path.exists():
        return None
    managed_paths = analysis_managed_paths(analysis_root_path)
    if not managed_paths:
        return None
    prev_root = analysis_root_path / f".analysis_prev_{analysis_id}"
    prev_root.mkdir(parents=True, exist_ok=False)
    for path in managed_paths:
        shutil.move(str(path), prev_root / path.name)
    return prev_root


def finalize_analysis_root(
    analysis_root_path: Path,
    tmp_root: Path,
    *,
    archive: bool,
    prev_root: Path | None,
) -> None:
    for path in sorted(tmp_root.iterdir()):
        shutil.move(str(path), analysis_root_path / path.name)
    shutil.rmtree(tmp_root, ignore_errors=True)
    if prev_root is None:
        return
    if not archive:
        shutil.rmtree(prev_root, ignore_errors=True)
        return
    prev_id = None
    summary_file = summary_path(prev_root)
    if summary_file.exists():
        try:
            payload = json.loads(summary_file.read_text())
            if isinstance(payload, dict):
                prev_id = payload.get("analysis_id")
        except Exception:
            prev_id = None
    if not prev_id:
        prev_id = prev_root.name.replace(".analysis_prev_", "")
    archive_root = analysis_state_root(analysis_root_path) / "_archive" / str(prev_id)
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    if archive_root.exists():
        shutil.rmtree(archive_root)
    shutil.move(str(prev_root), archive_root)


def analysis_managed_paths(analysis_root_path: Path) -> list[Path]:
    managed = [
        analysis_root_path / "reports",
        analysis_root_path / "manifests",
        analysis_root_path / "tables",
        analysis_root_path / "plots",
        analysis_state_root(analysis_root_path) / "_archive",
        analysis_root_path / "notebook__run_overview.py",
    ]
    return [path for path in managed if path.exists()]


def delete_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def analyze_lock_meta_path(tmp_root: Path) -> Path:
    return tmp_root / ANALYZE_LOCK_META_FILE


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_analyze_lock_pid(tmp_root: Path) -> int | None:
    lock_meta_file = analyze_lock_meta_path(tmp_root)
    if not lock_meta_file.exists():
        return None
    try:
        payload = json.loads(lock_meta_file.read_text())
    except (OSError, ValueError, TypeError):
        return None
    pid = payload.get("pid") if isinstance(payload, dict) else None
    if isinstance(pid, int):
        return pid
    if isinstance(pid, str):
        stripped = pid.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def recoverable_analyze_lock_reason(tmp_root: Path) -> str | None:
    pid = _read_analyze_lock_pid(tmp_root)
    if pid is None:
        return "lock metadata missing"
    if pid == os.getpid():
        return f"lock owned by current process pid={pid}"
    if not _is_pid_alive(pid):
        return f"lock owned by non-running pid={pid}"
    return None
