"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_cli_failure_contract.py

Subprocess-facing failure contract tests for the installed OPS CLI entrypoint.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import dnadesign.ops.cli as ops_cli
import dnadesign.ops.cli.app as ops_cli_app
from dnadesign.ops.cli import app

_ANSI_ESCAPE_RE = re.compile("\x1b\\[[0-9;?]*[ -/]*[@-~]")


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _run_ops(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "ops", *args],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_console_entrypoint_surfaces_contract_error_to_stderr() -> None:
    argv = ["catalog", "show", "missing.registry"]
    runner = CliRunner().invoke(app, argv)
    completed = _run_ops(*argv)

    assert runner.exit_code == 2
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Catalog contract error: unknown registry id: missing.registry" in runner.output
    assert "Catalog contract error: unknown registry id: missing.registry" in completed.stderr
    assert not _ANSI_ESCAPE_RE.search(completed.stderr)


def test_console_entrypoint_surfaces_usage_errors_to_stderr() -> None:
    completed = _run_ops("bogus")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Usage: ops [OPTIONS] COMMAND [ARGS]..." in completed.stderr
    assert "No such command 'bogus'" in completed.stderr
    assert not _ANSI_ESCAPE_RE.search(completed.stderr)


def test_console_entrypoint_surfaces_runbook_contract_errors_to_stderr() -> None:
    argv = ["runbook", "plan", "--runbook", "missing.yaml"]
    runner = CliRunner().invoke(app, argv)
    completed = _run_ops(*argv)

    assert runner.exit_code == 2
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Runbook contract error:" in runner.output
    assert "Runbook contract error:" in completed.stderr
    assert "runbook path must not be at repository root" in completed.stderr


def test_console_entrypoint_surfaces_progress_input_errors_to_stderr() -> None:
    argv = ["progress", "show", "opal.downstream.usr-infer-x-active-learning"]
    runner = CliRunner().invoke(app, argv)
    completed = _run_ops(*argv)

    assert runner.exit_code == 2
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Progress contract error:" in runner.output
    assert "Progress contract error:" in completed.stderr
    assert "--opal-config" in completed.stderr
    assert "--opal-workdir" in completed.stderr


def test_console_entrypoint_surfaces_runbook_init_notify_warning_to_stderr(tmp_path: Path) -> None:
    runner_runbook = tmp_path / "runner" / "densegen-runbook.yaml"
    subprocess_runbook = tmp_path / "subprocess" / "densegen-runbook.yaml"
    runner_workspace = tmp_path / "runner-workspace"
    subprocess_workspace = tmp_path / "subprocess-workspace"
    runner_argv = [
        "runbook",
        "init",
        "--workflow",
        "densegen",
        "--runbook",
        str(runner_runbook),
        "--workspace-root",
        str(runner_workspace),
        "--project",
        "dunlop",
        "--id",
        "densegen_demo",
    ]
    subprocess_argv = [
        "runbook",
        "init",
        "--workflow",
        "densegen",
        "--runbook",
        str(subprocess_runbook),
        "--workspace-root",
        str(subprocess_workspace),
        "--project",
        "dunlop",
        "--id",
        "densegen_demo",
    ]

    runner = CliRunner().invoke(app, runner_argv)
    completed = _run_ops(*subprocess_argv)

    assert runner.exit_code == 0
    assert completed.returncode == 0
    assert runner.stdout.strip() == str(runner_runbook.resolve())
    assert completed.stdout.strip() == str(subprocess_runbook.resolve())
    assert "Notify contract required before planning." in runner.stderr
    assert "Notify contract required before planning." in completed.stderr
    assert "NOTIFY_WEBHOOK_FILE" in completed.stderr


def test_console_entrypoint_requires_explicit_project_or_preset_for_runbook_init(tmp_path: Path) -> None:
    runbook_path = tmp_path / "contracts" / "densegen-runbook.yaml"
    workspace_root = tmp_path / "workspace_densegen"
    argv = [
        "runbook",
        "init",
        "--workflow",
        "densegen",
        "--runbook",
        str(runbook_path),
        "--workspace-root",
        str(workspace_root),
    ]
    runner = CliRunner().invoke(app, argv)
    completed = _run_ops(*argv)

    assert runner.exit_code == 2
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Runbook contract error:" in runner.output
    assert "provide exactly one of --project or --preset" in runner.output
    assert "Runbook contract error:" in completed.stderr
    assert "provide exactly one of --project or --preset" in completed.stderr
    assert not _ANSI_ESCAPE_RE.search(completed.stderr)


def test_console_wrapper_duplicates_and_closes_stderr_fd_per_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_dup(fd: int) -> int:
        captured["dup_fd"] = fd
        return 91

    def _fake_close(fd: int) -> None:
        captured.setdefault("closed_fds", []).append(fd)

    def _fake_main(argv, *, stderr_fd: int) -> int:
        captured["argv"] = tuple(argv or ())
        captured["stderr_fd"] = stderr_fd
        return 2

    monkeypatch.setattr(ops_cli.os, "dup", _fake_dup)
    monkeypatch.setattr(ops_cli.os, "close", _fake_close)
    monkeypatch.setattr(ops_cli_app, "main", _fake_main)

    assert ops_cli.main(("bogus",)) == 2
    assert captured["dup_fd"] == 2
    assert captured["stderr_fd"] == 91
    assert captured["argv"] == ("bogus",)
    assert captured["closed_fds"] == [91]


def test_console_wrapper_closes_stderr_fd_when_downstream_main_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    closed_fds: list[int] = []

    monkeypatch.setattr(ops_cli.os, "dup", lambda fd: 92)
    monkeypatch.setattr(ops_cli.os, "close", lambda fd: closed_fds.append(fd))

    def _boom(argv, *, stderr_fd: int) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(ops_cli_app, "main", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        ops_cli.main(("bogus",))

    assert closed_fds == [92]
