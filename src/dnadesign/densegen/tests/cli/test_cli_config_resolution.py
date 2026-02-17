"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_config_resolution.py

Config resolution behaviors for DenseGen CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from dnadesign.densegen.src.cli import main as cli


class _Ctx:
    def __init__(self) -> None:
        self.obj = {}


def test_resolve_config_path_prefers_env(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("densegen:\n  schema_version: '2.9'\n")

    monkeypatch.setenv("DENSEGEN_CONFIG_PATH", str(cfg))
    ctx = _Ctx()

    resolved, is_default = cli._resolve_config_path(ctx, None)
    assert resolved == cfg
    assert is_default is False


def test_resolve_config_path_prefers_cwd_config(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("densegen:\n  schema_version: '2.9'\n")
    monkeypatch.chdir(tmp_path)
    ctx = _Ctx()

    resolved, is_default = cli._resolve_config_path(ctx, None)
    assert resolved.resolve() == cfg.resolve()
    assert is_default is True


def test_resolve_config_path_does_not_use_parent_config(tmp_path: Path, monkeypatch) -> None:
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


def test_resolve_config_path_does_not_scan_workspace_candidates(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    for name in ("alpha", "beta"):
        ws = root / "src" / "dnadesign" / "densegen" / "workspaces" / name
        ws.mkdir(parents=True)
        (ws / "config.yaml").write_text("densegen:\n  schema_version: '2.9'\n")
    monkeypatch.chdir(root)
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(root))
    ctx = _Ctx()

    with pytest.raises(typer.Exit):
        cli._resolve_config_path(ctx, None)
