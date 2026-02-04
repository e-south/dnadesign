"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_cli_config_resolution.py

Config resolution behaviors for DenseGen CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from dnadesign.densegen.src import cli, cli_setup


class _Ctx:
    def __init__(self) -> None:
        self.obj = {}


def test_auto_config_path_single_workspace(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    ws = root / "src" / "dnadesign" / "densegen" / "workspaces" / "demo"
    ws.mkdir(parents=True)
    cfg = ws / "config.yaml"
    cfg.write_text("densegen:\n  schema_version: '2.9'\n")

    monkeypatch.chdir(root)
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(root))

    path, candidates = cli_setup._auto_config_path()
    assert candidates == []
    assert path == cfg


def test_auto_config_path_multiple_workspaces(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    for name in ("alpha", "beta"):
        ws = root / "src" / "dnadesign" / "densegen" / "workspaces" / name
        ws.mkdir(parents=True)
        (ws / "config.yaml").write_text("densegen:\n  schema_version: '2.9'\n")

    monkeypatch.chdir(root)
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(root))

    path, candidates = cli_setup._auto_config_path()
    assert path is None
    assert len(candidates) == 2


def test_resolve_config_path_prefers_env(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("densegen:\n  schema_version: '2.9'\n")

    monkeypatch.setenv("DENSEGEN_CONFIG_PATH", str(cfg))
    ctx = _Ctx()

    resolved, is_default = cli._resolve_config_path(ctx, None)
    assert resolved == cfg
    assert is_default is False


def test_resolve_config_path_prefers_parent_config(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    cfg = root / "config.yaml"
    cfg.write_text("densegen:\n  schema_version: '2.9'\n")
    child = root / "outputs"
    child.mkdir()
    monkeypatch.chdir(child)
    ctx = _Ctx()
    with pytest.raises(typer.Exit):
        cli._resolve_config_path(ctx, None)
