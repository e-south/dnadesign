"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_usr_harness_script.py

Contract tests for deterministic USR harness cycle script behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _make_uv_stub(bin_dir: Path, *, fail_on_contains: str | None = None) -> Path:
    uv_stub = bin_dir / "uv"
    fail_line = f'if [[ "$*" == *"{fail_on_contains}"* ]]; then exit 7; fi\n' if fail_on_contains is not None else ""
    uv_stub.write_text(
        f'#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\n" "$*" >> "${{UV_LOG_PATH}}"\n{fail_line}exit 0\n',
        encoding="utf-8",
    )
    uv_stub.chmod(0o755)
    return uv_stub


def test_usr_harness_script_writes_success_report_with_stubbed_uv(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script = repo_root / "src/dnadesign/usr/scripts/run_usr_harness_cycle.sh"
    assert script.exists()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _make_uv_stub(bin_dir)

    log_path = tmp_path / "uv.log"
    report_path = tmp_path / "harness-report.json"

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["UV_LOG_PATH"] = str(log_path)
    env["USR_HARNESS_REPORT_PATH"] = str(report_path)

    completed = subprocess.run(["bash", str(script)], cwd=repo_root, env=env, capture_output=True, text=True)
    assert completed.returncode == 0
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["exit_code"] == 0
    assert report["failed_step"] is None
    step_names = [step["name"] for step in report["steps"]]
    assert step_names == [
        "preflight-cli-help",
        "preflight-sync-focused-tests",
        "run-full-usr-tests",
        "verify-ruff-check",
        "verify-ruff-format",
        "verify-docs-checks",
    ]


def test_usr_harness_script_reports_failure_step_with_stubbed_uv(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script = repo_root / "src/dnadesign/usr/scripts/run_usr_harness_cycle.sh"
    assert script.exists()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _make_uv_stub(bin_dir, fail_on_contains="ruff check")

    log_path = tmp_path / "uv.log"
    report_path = tmp_path / "harness-report-fail.json"

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["UV_LOG_PATH"] = str(log_path)
    env["USR_HARNESS_REPORT_PATH"] = str(report_path)

    completed = subprocess.run(["bash", str(script)], cwd=repo_root, env=env, capture_output=True, text=True)
    assert completed.returncode != 0
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failure"
    assert report["failed_step"] == "verify-ruff-check"
    assert report["exit_code"] != 0
