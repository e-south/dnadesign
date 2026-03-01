"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_workspace_source.py

Workspace resolver tests for notify tool/config shorthand flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.profiles.workspace import (
    list_tool_workspaces,
    resolve_tool_workspace_config_path,
)


def test_resolve_tool_workspace_config_path_densegen_from_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = repo_root / "src" / "dnadesign" / "densegen" / "workspaces" / "demo_a" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    resolved = resolve_tool_workspace_config_path(
        tool="densegen",
        workspace="demo_a",
        search_start=repo_root,
    )

    assert resolved == config_path.resolve()


def test_resolve_tool_workspace_config_path_supports_infer_alias(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = repo_root / "src" / "dnadesign" / "infer" / "workspaces" / "demo_i" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("jobs: []\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    resolved = resolve_tool_workspace_config_path(
        tool="infer-evo2",
        workspace="demo_i",
        search_start=repo_root,
    )

    assert resolved == config_path.resolve()


def test_list_tool_workspaces_reports_available_workspace_names(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    workspaces_root = repo_root / "src" / "dnadesign" / "densegen" / "workspaces"
    (workspaces_root / "demo_a").mkdir(parents=True, exist_ok=True)
    (workspaces_root / "demo_a" / "config.yaml").write_text("densegen:\n  run:\n    id: a\n", encoding="utf-8")
    (workspaces_root / "demo_b").mkdir(parents=True, exist_ok=True)
    (workspaces_root / "demo_b" / "config.yaml").write_text("densegen:\n  run:\n    id: b\n", encoding="utf-8")
    (workspaces_root / "ignore_me").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    names = list_tool_workspaces(tool="densegen", search_start=repo_root)

    assert names == ["demo_a", "demo_b"]


def test_resolve_tool_workspace_config_path_rejects_path_like_workspace_name(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    with pytest.raises(NotifyConfigError, match="workspace must be a workspace name"):
        resolve_tool_workspace_config_path(
            tool="densegen",
            workspace="demo/path",
            search_start=repo_root,
        )


def test_resolve_tool_workspace_config_path_missing_workspace_lists_available(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = repo_root / "src" / "dnadesign" / "densegen" / "workspaces" / "demo_a" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    with pytest.raises(NotifyConfigError, match="Available workspaces: demo_a"):
        resolve_tool_workspace_config_path(
            tool="densegen",
            workspace="missing_demo",
            search_start=repo_root,
        )
