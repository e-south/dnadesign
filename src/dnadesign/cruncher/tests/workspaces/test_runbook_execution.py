"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/workspaces/test_runbook_execution.py

Validate machine runbook loading and fail-fast execution contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dnadesign.cruncher.workspaces.runbook as runbook_module
from dnadesign.cruncher.workspaces.runbook import load_workspace_runbook, run_workspace_runbook


def _write_runbook(workspace: Path, payload: dict) -> Path:
    runbook_path = workspace / "configs" / "runbook.yaml"
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    runbook_path.write_text(yaml.safe_dump(payload))
    return runbook_path


def test_runbook_rejects_unknown_keys() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "name": "demo",
            "steps": [{"id": "lock", "run": ["lock"]}],
            "unexpected": True,
        }
    }
    with pytest.raises(ValueError):
        load_workspace_runbook(Path("runbook.yaml"), raw=payload)


def test_runbook_rejects_disallowed_cli_surface() -> None:
    payload = {
        "runbook": {
            "schema_version": 1,
            "name": "demo",
            "steps": [{"id": "danger", "run": ["rm", "-rf", "/"]}],
        }
    }
    with pytest.raises(ValueError, match="disallowed cruncher command"):
        load_workspace_runbook(Path("runbook.yaml"), raw=payload)


def test_runbook_executes_selected_steps_in_runbook_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [
                    {"id": "clean", "run": ["workspaces", "clean-transient", "--root", ".", "--confirm"]},
                    {"id": "lock", "run": ["lock", "-c", "configs/config.yaml"]},
                    {"id": "parse", "run": ["parse", "--force-overwrite", "-c", "configs/config.yaml"]},
                ],
            }
        },
    )

    calls: list[list[str]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        calls.append([str(item) for item in cmd])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)

    result = run_workspace_runbook(runbook_path, step_ids=["parse", "lock"])

    assert result.executed_step_ids == ["lock", "parse"]
    assert calls == [
        ["uv", "run", "cruncher", "lock", "-c", "configs/config.yaml"],
        ["uv", "run", "cruncher", "parse", "--force-overwrite", "-c", "configs/config.yaml"],
    ]


def test_runbook_streams_step_output_instead_of_capturing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [{"id": "lock", "run": ["lock", "-c", "configs/config.yaml"]}],
            }
        },
    )

    call_kwargs: list[dict[str, object]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        call_kwargs.append(dict(kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)

    run_workspace_runbook(runbook_path)

    assert len(call_kwargs) == 1
    kwargs = call_kwargs[0]
    assert kwargs.get("capture_output") is not True
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs


def test_runbook_routes_step_output_to_log_file_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [{"id": "lock", "run": ["lock", "-c", "configs/config.yaml"]}],
            }
        },
    )
    output_log = tmp_path / "runbook.log"
    call_kwargs: list[dict[str, object]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        _ = cmd
        call_kwargs.append(dict(kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)

    run_workspace_runbook(runbook_path, output_log_path=output_log)

    assert len(call_kwargs) == 1
    kwargs = call_kwargs[0]
    assert "stdout" in kwargs
    assert kwargs.get("stderr") == runbook_module.subprocess.STDOUT
    assert output_log.exists()


def test_runbook_fails_fast_on_step_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [
                    {"id": "lock", "run": ["lock", "-c", "configs/config.yaml"]},
                    {"id": "parse", "run": ["parse", "--force-overwrite", "-c", "configs/config.yaml"]},
                    {"id": "sample", "run": ["sample", "--force-overwrite", "-c", "configs/config.yaml"]},
                ],
            }
        },
    )

    calls: list[list[str]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        calls.append([str(item) for item in cmd])
        if cmd[3] == "parse":
            return SimpleNamespace(returncode=2, stdout="", stderr="parse failed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)

    with pytest.raises(RuntimeError, match="Runbook step failed"):
        run_workspace_runbook(runbook_path)

    assert calls == [
        ["uv", "run", "cruncher", "lock", "-c", "configs/config.yaml"],
        ["uv", "run", "cruncher", "parse", "--force-overwrite", "-c", "configs/config.yaml"],
    ]


def test_runbook_sets_writable_home_for_child_processes_when_home_is_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    runbook_path = _write_runbook(
        workspace,
        {
            "runbook": {
                "schema_version": 1,
                "name": "demo",
                "steps": [{"id": "sample", "run": ["sample", "-c", "configs/config.yaml"]}],
            }
        },
    )

    call_kwargs: list[dict[str, object]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        _ = cmd
        call_kwargs.append(dict(kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runbook_module.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(runbook_module.os, "environ", {"HOME": "/tmp/unwritable-home"})
    monkeypatch.setattr(runbook_module, "_is_writable_directory", lambda _: False)

    run_workspace_runbook(runbook_path)

    assert len(call_kwargs) == 1
    env = call_kwargs[0].get("env")
    assert isinstance(env, dict)
    expected_home = (workspace / ".cruncher" / ".runtime_home").resolve()
    assert Path(str(env["HOME"])).resolve() == expected_home
    assert expected_home.is_dir()
