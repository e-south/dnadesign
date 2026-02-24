"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_catalog_utils_densegen_workspace_root.py

Tests for DenseGen workspace-root discovery used by catalog export commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.cli import catalog_utils as catalog_utils_module


def test_densegen_workspace_root_prefers_explicit_env(monkeypatch, tmp_path: Path) -> None:
    fake_root = (tmp_path / "densegen_workspaces").resolve()
    fake_root.mkdir(parents=True)
    config_path = tmp_path / "workspace" / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("dummy: true\n", encoding="utf-8")

    monkeypatch.setenv("DNADESIGN_DENSEGEN_WORKSPACES_ROOT", str(fake_root))

    resolved = catalog_utils_module._densegen_workspaces_root(config_path)
    assert resolved == fake_root


def test_densegen_workspace_root_uses_installed_package_location(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "workspace" / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("dummy: true\n", encoding="utf-8")

    package_root = tmp_path / "pkg" / "dnadesign" / "densegen"
    workspaces_root = (package_root / "workspaces").resolve()
    workspaces_root.mkdir(parents=True)

    class _Spec:
        submodule_search_locations = [str(package_root)]

    monkeypatch.delenv("DNADESIGN_DENSEGEN_WORKSPACES_ROOT", raising=False)
    monkeypatch.setattr(catalog_utils_module.importlib_util, "find_spec", lambda name: _Spec())

    resolved = catalog_utils_module._densegen_workspaces_root(config_path)
    assert resolved == workspaces_root


def test_densegen_workspace_root_does_not_scan_monorepo_path_shape(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "workspace" / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("dummy: true\n", encoding="utf-8")

    monorepo_like = tmp_path / "src" / "dnadesign" / "densegen" / "workspaces"
    monorepo_like.mkdir(parents=True)

    monkeypatch.delenv("DNADESIGN_DENSEGEN_WORKSPACES_ROOT", raising=False)
    monkeypatch.setattr(catalog_utils_module.importlib_util, "find_spec", lambda name: None)

    resolved = catalog_utils_module._densegen_workspaces_root(config_path)
    assert resolved is None
