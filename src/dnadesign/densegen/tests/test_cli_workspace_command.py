from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.densegen.src.cli import DEFAULT_CONFIG_FILENAME, _workspace_command


def test_workspace_command_uses_cd_when_config_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / DEFAULT_CONFIG_FILENAME).write_text("x")

    monkeypatch.chdir(tmp_path)
    cmd = _workspace_command("dense run", run_root=workspace)
    assert cmd == f"cd {workspace} && dense run"


def test_workspace_command_omits_cd_when_already_in_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / DEFAULT_CONFIG_FILENAME).write_text("x")

    monkeypatch.chdir(workspace)
    cmd = _workspace_command("dense run", run_root=workspace)
    assert cmd == "dense run"


def test_workspace_command_falls_back_to_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("x")
    other = tmp_path / "other"
    other.mkdir()

    monkeypatch.chdir(tmp_path)
    cmd = _workspace_command("dense run", cfg_path=cfg_path, run_root=other)
    assert cmd == f"dense run -c {cfg_path}"
