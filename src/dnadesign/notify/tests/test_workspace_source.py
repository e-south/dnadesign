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


def test_resolve_tool_workspace_config_path_supports_infer_tool(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = repo_root / "src" / "dnadesign" / "infer" / "workspaces" / "demo_i" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("jobs: []\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    resolved = resolve_tool_workspace_config_path(
        tool="infer",
        workspace="demo_i",
        search_start=repo_root,
    )

    assert resolved == config_path.resolve()


def test_resolve_tool_workspace_config_path_supports_local_infer_workspace_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = repo_root / "workspaces" / "demo_i" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("jobs: []\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    resolved = resolve_tool_workspace_config_path(
        tool="infer",
        workspace="demo_i",
        search_start=repo_root,
    )

    assert resolved == config_path.resolve()


def test_resolve_tool_workspace_config_path_supports_construct_tool_with_project_selector(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    workspace_dir = repo_root / "src" / "dnadesign" / "construct" / "workspaces" / "demo_c"
    config_path = workspace_dir / "config.slot_a.window.yaml"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("job:\n  id: slot_a_window\n  output:\n    dataset: demo_output\n", encoding="utf-8")
    (workspace_dir / "construct.workspace.yaml").write_text(
        "\n".join(
            [
                "workspace:",
                "  id: demo_c",
                "  profile: promoter-swap-demo",
                "  projects:",
                "    - id: slot_a_window",
                "      config: config.slot_a.window.yaml",
                "      flow: replace-anchor-in-template",
                "      input_dataset: anchors_demo",
                "      output_dataset: demo_output",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    resolved = resolve_tool_workspace_config_path(
        tool="construct",
        workspace="demo_c:slot_a_window",
        search_start=repo_root,
    )

    assert resolved == config_path.resolve()


def test_resolve_tool_workspace_config_path_supports_local_construct_workspace_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    workspace_dir = repo_root / "demo_c"
    config_path = workspace_dir / "config.slot_a.window.yaml"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("job:\n  id: slot_a_window\n  output:\n    dataset: demo_output\n", encoding="utf-8")
    (workspace_dir / "construct.workspace.yaml").write_text(
        "\n".join(
            [
                "workspace:",
                "  id: demo_c",
                "  profile: promoter-swap-demo",
                "  projects:",
                "    - id: slot_a_window",
                "      config: config.slot_a.window.yaml",
                "      flow: replace-anchor-in-template",
                "      input_dataset: anchors_demo",
                "      output_dataset: demo_output",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    resolved = resolve_tool_workspace_config_path(
        tool="construct",
        workspace="demo_c:slot_a_window",
        search_start=repo_root,
    )

    assert resolved == config_path.resolve()


def test_resolve_tool_workspace_config_path_construct_uses_single_project_by_default(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    workspace_dir = repo_root / "src" / "dnadesign" / "construct" / "workspaces" / "single_project"
    config_path = workspace_dir / "config.yaml"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("job:\n  id: slot_a_window\n  output:\n    dataset: demo_output\n", encoding="utf-8")
    (workspace_dir / "construct.workspace.yaml").write_text(
        "\n".join(
            [
                "workspace:",
                "  id: single_project",
                "  profile: blank",
                "  projects:",
                "    - id: slot_a_window",
                "      config: config.yaml",
                "      flow: replace-anchor-in-template",
                "      input_dataset: anchors_demo",
                "      output_dataset: demo_output",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    resolved = resolve_tool_workspace_config_path(
        tool="construct",
        workspace="single_project",
        search_start=repo_root,
    )

    assert resolved == config_path.resolve()


def test_resolve_tool_workspace_config_path_construct_rejects_ambiguous_workspace_without_project(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    workspace_dir = repo_root / "src" / "dnadesign" / "construct" / "workspaces" / "demo_c"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.slot_a.window.yaml", "config.slot_b.window.yaml"):
        (workspace_dir / name).write_text("job:\n  id: demo\n  output:\n    dataset: demo_output\n", encoding="utf-8")
    (workspace_dir / "construct.workspace.yaml").write_text(
        "\n".join(
            [
                "workspace:",
                "  id: demo_c",
                "  profile: promoter-swap-demo",
                "  projects:",
                "    - id: slot_a_window",
                "      config: config.slot_a.window.yaml",
                "      flow: replace-anchor-in-template",
                "      input_dataset: anchors_demo",
                "      output_dataset: demo_output_a",
                "    - id: slot_b_window",
                "      config: config.slot_b.window.yaml",
                "      flow: replace-anchor-in-template",
                "      input_dataset: anchors_demo",
                "      output_dataset: demo_output_b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    with pytest.raises(NotifyConfigError, match="Available project ids: slot_a_window, slot_b_window"):
        resolve_tool_workspace_config_path(
            tool="construct",
            workspace="demo_c",
            search_start=repo_root,
        )


def test_resolve_tool_workspace_config_path_rejects_legacy_infer_alias(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    with pytest.raises(NotifyConfigError, match="unsupported tool"):
        resolve_tool_workspace_config_path(
            tool="infer-evo2",
            workspace="demo_i",
            search_start=repo_root,
        )


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


def test_list_tool_workspaces_reports_construct_workspace_names(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    workspaces_root = repo_root / "src" / "dnadesign" / "construct" / "workspaces"
    (workspaces_root / "demo_a").mkdir(parents=True, exist_ok=True)
    (workspaces_root / "demo_a" / "construct.workspace.yaml").write_text(
        "workspace:\n  id: demo_a\n  profile: blank\n  projects: []\n",
        encoding="utf-8",
    )
    (workspaces_root / "demo_b").mkdir(parents=True, exist_ok=True)
    (workspaces_root / "demo_b" / "construct.workspace.yaml").write_text(
        "workspace:\n  id: demo_b\n  profile: blank\n  projects: []\n",
        encoding="utf-8",
    )
    (workspaces_root / "ignore_me").mkdir(parents=True, exist_ok=True)
    (workspaces_root / "ignore_me" / "config.yaml").write_text("job: {}\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    names = list_tool_workspaces(tool="construct", search_start=repo_root)

    assert names == ["demo_a", "demo_b"]


def test_list_tool_workspaces_merges_repo_and_local_infer_workspace_roots(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_scoped = repo_root / "src" / "dnadesign" / "infer" / "workspaces" / "repo_demo" / "config.yaml"
    local_scoped = repo_root / "workspaces" / "local_demo" / "config.yaml"
    repo_scoped.parent.mkdir(parents=True, exist_ok=True)
    local_scoped.parent.mkdir(parents=True, exist_ok=True)
    repo_scoped.write_text("jobs: []\n", encoding="utf-8")
    local_scoped.write_text("jobs: []\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")

    names = list_tool_workspaces(tool="infer", search_start=repo_root)

    assert names == ["local_demo", "repo_demo"]


def test_resolve_tool_workspace_config_path_construct_supports_external_workspace_root_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    external_root = tmp_path / "external_construct_workspaces"
    workspace_dir = external_root / "demo_external"
    config_path = workspace_dir / "config.yaml"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("job:\n  id: slot_a_window\n  output:\n    dataset: demo_output\n", encoding="utf-8")
    (workspace_dir / "construct.workspace.yaml").write_text(
        "\n".join(
            [
                "workspace:",
                "  id: demo_external",
                "  profile: blank",
                "  projects:",
                "    - id: slot_a_window",
                "      config: config.yaml",
                "      flow: replace-anchor-in-template",
                "      input_dataset: anchors_demo",
                "      output_dataset: demo_output",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text("[project]\nname='dnadesign'\n", encoding="utf-8")
    monkeypatch.setenv("CONSTRUCT_WORKSPACE_ROOT", str(external_root))

    resolved = resolve_tool_workspace_config_path(
        tool="construct",
        workspace="demo_external",
        search_start=repo_root,
    )

    assert resolved == config_path.resolve()


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
