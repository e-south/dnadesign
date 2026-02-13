"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_hpc_notify_watch_qsub.py

Portable checks for notify watcher qsub script behavior without HPC dependencies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
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


def _write_fake_uv(bin_dir: Path) -> Path:
    capture_path = bin_dir / "uv.calls"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "echo \"$*\" >> \"$UV_CAPTURE\"",
                "if [[ \"$1\" == \"run\" && \"$2\" == \"notify\" && \"$3\" == \"setup\" &&",
                "      \"$4\" == \"resolve-events\" ]]; then",
                "  if [[ \" $* \" == *\" --print-policy \"* ]]; then",
                "    echo \"${FAKE_RESOLVED_POLICY:-densegen}\"",
                "    exit 0",
                "  fi",
                "  if [[ \" $* \" == *\" --json \"* ]]; then",
                (
                    "    echo \"{\\\"events\\\":\\\"$FAKE_EVENTS_PATH\\\","
                    "\\\"policy\\\":\\\"${FAKE_RESOLVED_POLICY:-densegen}\\\"}\""
                ),
                "    exit 0",
                "  fi",
                "  echo \"$FAKE_EVENTS_PATH\"",
                "fi",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    return capture_path


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_qsub_script_profile_mode_uses_profile_and_follow(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/notify-watch.qsub"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path = _write_fake_uv(bin_dir)

    profile_path = tmp_path / "notify.profile.json"
    profile_path.write_text(
        '{"profile_version":2,"provider":"slack","webhook":{"source":"env","ref":"NOTIFY_WEBHOOK"},"events":"/tmp/e"}\n',
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["UV_CAPTURE"] = str(capture_path)
    env["NOTIFY_PROFILE"] = str(profile_path)

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = capture_path.read_text(encoding="utf-8")
    assert "run notify usr-events watch --profile" in calls
    assert str(profile_path) in calls
    assert "--follow" in calls
    assert "--wait-for-events" in calls
    assert "--poll-interval-seconds 1.0" in calls
    assert "--stop-on-terminal-status" in calls
    assert "--idle-timeout" in calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_qsub_script_can_resolve_events_from_tool_config(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/notify-watch.qsub"

    events_path = tmp_path / "dataset" / ".events.log"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path = _write_fake_uv(bin_dir)

    config_path = tmp_path / "tool.config.yaml"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["UV_CAPTURE"] = str(capture_path)
    env["FAKE_EVENTS_PATH"] = str(events_path)
    env["FAKE_RESOLVED_POLICY"] = "densegen"
    env["NOTIFY_TOOL"] = "densegen"
    env["NOTIFY_CONFIG"] = str(config_path)
    env["DENSEGEN_WEBHOOK"] = "https://example.invalid/webhook"
    env["WEBHOOK_ENV"] = "DENSEGEN_WEBHOOK"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = capture_path.read_text(encoding="utf-8")
    assert "run notify setup resolve-events --tool densegen --config" in calls
    assert "run notify usr-events watch" in calls
    assert "--events" in calls
    assert str(events_path) in calls
    assert "--only-actions densegen_health,densegen_flush_failed,materialize" in calls
    assert "--only-tools densegen" in calls
    assert "--url-env DENSEGEN_WEBHOOK" in calls
    assert "--follow" in calls
    assert "--wait-for-events" in calls
    assert "--poll-interval-seconds 1.0" in calls
    assert "--stop-on-terminal-status" in calls
    assert "--idle-timeout" in calls
    assert f"--cursor {tmp_path / 'outputs' / 'notify' / 'densegen' / 'cursor'}" in calls
    assert f"--spool-dir {tmp_path / 'outputs' / 'notify' / 'densegen' / 'spool'}" in calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_qsub_script_accepts_missing_events_file_when_waiting(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/notify-watch.qsub"

    events_path = tmp_path / "dataset" / ".events.log"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path = _write_fake_uv(bin_dir)

    config_path = tmp_path / "tool.config.yaml"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["UV_CAPTURE"] = str(capture_path)
    env["FAKE_EVENTS_PATH"] = str(events_path)
    env["FAKE_RESOLVED_POLICY"] = "densegen"
    env["NOTIFY_TOOL"] = "densegen"
    env["NOTIFY_CONFIG"] = str(config_path)
    env["DENSEGEN_WEBHOOK"] = "https://example.invalid/webhook"
    env["WEBHOOK_ENV"] = "DENSEGEN_WEBHOOK"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = capture_path.read_text(encoding="utf-8")
    assert "run notify usr-events watch" in calls
    assert "--events" in calls
    assert str(events_path) in calls
    assert "--wait-for-events" in calls
    assert "--poll-interval-seconds 1.0" in calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_qsub_script_applies_infer_policy_filters(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/notify-watch.qsub"

    events_path = tmp_path / "dataset" / ".events.log"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path = _write_fake_uv(bin_dir)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["UV_CAPTURE"] = str(capture_path)
    env["EVENTS_PATH"] = str(events_path)
    env["DENSEGEN_WEBHOOK"] = "https://example.invalid/webhook"
    env["WEBHOOK_ENV"] = "DENSEGEN_WEBHOOK"
    env["NOTIFY_POLICY"] = "infer_evo2"
    env["NOTIFY_NAMESPACE"] = "infer_evo2"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = capture_path.read_text(encoding="utf-8")
    assert "--only-actions attach,materialize" in calls
    assert "--only-tools infer" in calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_qsub_script_requires_notify_policy_when_events_path_is_explicit(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/notify-watch.qsub"

    events_path = tmp_path / "dataset" / ".events.log"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path = _write_fake_uv(bin_dir)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["UV_CAPTURE"] = str(capture_path)
    env["EVENTS_PATH"] = str(events_path)
    env["NOTIFY_WEBHOOK"] = "https://example.invalid/webhook"
    env.pop("NOTIFY_POLICY", None)

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Missing NOTIFY_POLICY." in result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_qsub_script_requires_namespace_when_events_path_is_explicit_without_tool(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/notify-watch.qsub"

    events_path = tmp_path / "dataset" / ".events.log"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path = _write_fake_uv(bin_dir)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["UV_CAPTURE"] = str(capture_path)
    env["EVENTS_PATH"] = str(events_path)
    env["NOTIFY_WEBHOOK"] = "https://example.invalid/webhook"
    env["NOTIFY_POLICY"] = "densegen"
    env.pop("NOTIFY_NAMESPACE", None)
    env.pop("NOTIFY_TOOL", None)

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Missing NOTIFY_NAMESPACE." in result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_qsub_script_uses_notify_webhook_default_env_name(tmp_path: Path) -> None:
    repo_root = _repo_root()
    script_path = repo_root / "docs/bu-scc/jobs/notify-watch.qsub"

    events_path = tmp_path / "dataset" / ".events.log"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_path = _write_fake_uv(bin_dir)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["UV_CAPTURE"] = str(capture_path)
    env["EVENTS_PATH"] = str(events_path)
    env["NOTIFY_WEBHOOK"] = "https://example.invalid/webhook"
    env["NOTIFY_POLICY"] = "generic"
    env["NOTIFY_NAMESPACE"] = "generic"
    env.pop("WEBHOOK_ENV", None)

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = capture_path.read_text(encoding="utf-8")
    assert "--url-env NOTIFY_WEBHOOK" in calls
