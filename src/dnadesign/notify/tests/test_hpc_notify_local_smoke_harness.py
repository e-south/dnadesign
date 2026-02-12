"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_hpc_notify_local_smoke_harness.py

Local smoke checks for qsub-like notify watcher flows without scheduler access.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("uv") is None, reason="bash and uv are required")
def test_local_notify_smoke_harness_env_mode(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/hpc/jobs/local_notify_watch_smoke.sh"
    workdir = tmp_path / "env-smoke"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--repo-root",
            str(repo_root),
            "--mode",
            "env",
            "--workdir",
            str(workdir),
            "--poll-interval-seconds",
            "0.05",
            "--idle-timeout-seconds",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    capture_path = workdir / "captures" / "requests.jsonl"
    assert capture_path.exists()
    payloads = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert payloads
    assert any("SUCCESS" in str(payload.get("text", "")) for payload in payloads)


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("uv") is None, reason="bash and uv are required")
def test_local_notify_smoke_harness_profile_mode(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/hpc/jobs/local_notify_watch_smoke.sh"
    workdir = tmp_path / "profile-smoke"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--repo-root",
            str(repo_root),
            "--mode",
            "profile",
            "--workdir",
            str(workdir),
            "--poll-interval-seconds",
            "0.05",
            "--idle-timeout-seconds",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    profile_path = workdir / "outputs" / "notify" / "densegen" / "profile.json"
    cursor_path = workdir / "outputs" / "notify" / "densegen" / "cursor"
    capture_path = workdir / "captures" / "requests.jsonl"
    assert profile_path.exists()
    assert cursor_path.exists()
    assert capture_path.exists()
