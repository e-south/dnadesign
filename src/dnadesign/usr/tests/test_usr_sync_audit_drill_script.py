"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_usr_sync_audit_drill_script.py

Contract tests for the deterministic USR sync audit drill entrypoint.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dnadesign.usr.ops.sync_audit_drill import _final_sync_state_is_current


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_usr_sync_audit_drill_script_runs_and_emits_audit_report(tmp_path: Path) -> None:
    repo_root = _repo_root()
    report_path = tmp_path / "sync-audit-drill-report.json"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "usr-sync-audit-drill",
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
    assert "_registry/operator-note.yaml" in diff_before_pull["_auxiliary"]["remote_only"]
    assert diff_before_pull["_auxiliary"]["local_only"] == []

    assert pull["action"] == "pull"
    assert pull["transfer_state"] in {"TRANSFERRED", "NO-OP"}
    assert "_registry/operator-note.yaml" in pull["_auxiliary"]["remote_only"]

    assert diff_before_push["action"] == "diff"
    assert diff_before_push["_derived"]["changed"] is True
    assert diff_before_push["_auxiliary"]["changed"] is True
    assert "_registry/local-note.yaml" in diff_before_push["_auxiliary"]["local_only"]
    assert any(
        path.startswith("infer/part-") and path.endswith(".parquet")
        for path in diff_before_push["_derived"]["local_only"]
    )

    assert push["action"] == "push"
    assert push["transfer_state"] in {"TRANSFERRED", "NO-OP"}
    assert "_registry/local-note.yaml" in push["_auxiliary"]["local_only"]
    assert any(path.startswith("infer/part-") and path.endswith(".parquet") for path in push["_derived"]["local_only"])

    assert diff_after_push["action"] == "diff"
    assert diff_after_push["transfer_state"] == "DIFF-ONLY"
    assert diff_after_push["primary"]["changed"] is False
    assert diff_after_push["meta"]["changed"] is False
    assert diff_after_push[".events.log"]["changed"] is False
    assert diff_after_push["_snapshots"]["changed"] is False
    assert diff_after_push["_derived"]["changed"] is False
    assert diff_after_push["_auxiliary"]["changed"] is False
    assert diff_after_push["_derived"]["local_only"] == []
    assert diff_after_push["_derived"]["remote_only"] == []
    assert diff_after_push["_auxiliary"]["local_only"] == []
    assert diff_after_push["_auxiliary"]["remote_only"] == []


@pytest.mark.parametrize(
    "section",
    ["primary", "meta", ".events.log", "_snapshots", "_derived", "_auxiliary"],
)
def test_final_sync_state_requires_every_reported_section_to_match(section: str) -> None:
    data = {
        name: {"changed": name == section}
        for name in ("primary", "meta", ".events.log", "_snapshots", "_derived", "_auxiliary")
    }

    assert _final_sync_state_is_current(data) is False
