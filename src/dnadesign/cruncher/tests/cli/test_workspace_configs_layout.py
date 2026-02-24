"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_workspace_configs_layout.py

Contracts for the workspace configs/ layout and strict config discovery.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    discover_workspaces,
    resolve_config_path,
)


def _write_workspace_config(workspace: Path) -> Path:
    config_path = workspace / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("cruncher: {schema_version: 3, workspace: {out_dir: outputs, regulator_sets: [[lexA]]}}\n")
    return config_path


def test_discover_workspaces_requires_configs_config_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "workspaces"
    workspace = root / "demo"
    _write_workspace_config(workspace)

    monkeypatch.setenv("CRUNCHER_WORKSPACE_ROOTS", str(root))
    discovered = discover_workspaces(cwd=tmp_path)

    assert len(discovered) == 1
    assert discovered[0].name == "demo"
    assert discovered[0].config_path == (workspace / "configs" / "config.yaml").resolve()


def test_resolve_config_prefers_workspace_configs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "demo"
    config_path = _write_workspace_config(workspace)

    monkeypatch.setenv("CRUNCHER_NONINTERACTIVE", "1")
    resolved = resolve_config_path(None, cwd=workspace, log=False)

    assert resolved == config_path.resolve()


def test_resolve_config_from_workspace_configs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "demo"
    config_path = _write_workspace_config(workspace)
    configs_dir = workspace / "configs"

    monkeypatch.setenv("CRUNCHER_NONINTERACTIVE", "1")
    resolved = resolve_config_path(None, cwd=configs_dir, log=False)

    assert resolved == config_path.resolve()


def test_root_level_config_yaml_is_not_used_for_implicit_resolution(tmp_path: Path) -> None:
    workspace = tmp_path / "demo"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "config.yaml").write_text("cruncher: {}\n")

    with pytest.raises(ConfigResolutionError):
        resolve_config_path(None, cwd=workspace, log=False)
