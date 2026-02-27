"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_usr_sync_audit_drill_script.py

Contract tests for the deterministic USR sync audit drill script.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_usr_sync_audit_drill_script_runs_and_emits_audit_report(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script = repo_root / "src/dnadesign/usr/scripts/run_usr_sync_audit_drill.py"
    assert script.exists()

    report_path = tmp_path / "sync-audit-drill-report.json"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(script),
            "--work-dir",
            str(tmp_path / "work"),
            "--report-json",
            str(report_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["final_up_to_date"] is True

    diff_before_pull = report["audits"]["diff_before_pull"]["data"]
    pull = report["audits"]["pull"]["data"]
    diff_before_push = report["audits"]["diff_before_push"]["data"]
    push = report["audits"]["push"]["data"]
    diff_after_push = report["audits"]["diff_after_push"]["data"]

    assert diff_before_pull["action"] == "diff"
    assert diff_before_pull["transfer_state"] == "DIFF-ONLY"
    assert diff_before_pull["_derived"]["changed"] is True
    assert diff_before_pull["_auxiliary"]["changed"] is True

    assert pull["action"] == "pull"
    assert pull["transfer_state"] in {"TRANSFERRED", "NO-OP"}

    assert diff_before_push["action"] == "diff"
    assert diff_before_push["_derived"]["changed"] is True
    assert diff_before_push["_auxiliary"]["changed"] is True

    assert push["action"] == "push"
    assert push["transfer_state"] in {"TRANSFERRED", "NO-OP"}

    assert diff_after_push["action"] == "diff"
    assert diff_after_push["transfer_state"] == "DIFF-ONLY"
    assert diff_after_push["primary"]["changed"] is False
    assert diff_after_push["_derived"]["changed"] is False
    assert diff_after_push["_auxiliary"]["changed"] is False
