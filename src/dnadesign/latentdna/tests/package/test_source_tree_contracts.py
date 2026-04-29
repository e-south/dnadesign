"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/package/test_source_tree_contracts.py

Source-tree contracts for the latentdna package layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _latentdna_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "src" / "dnadesign" / "latentdna"
    raise RuntimeError("repo root not found")


def test_latentdna_root_keeps_progressive_disclosure_directories() -> None:
    latentdna_root = _latentdna_root()
    assert (latentdna_root / "README.md").is_file()
    assert (latentdna_root / "docs").is_dir()
    assert (latentdna_root / "ops").is_dir()
    assert (latentdna_root / "docs" / "reference").is_dir()
    assert (latentdna_root / "docs" / "dev").is_dir()
    assert (latentdna_root / "src").is_dir()
    assert (latentdna_root / "tests").is_dir()
    assert (latentdna_root / "tests" / "perf").is_dir()
    assert (latentdna_root / "workspaces").is_dir()


def test_latentdna_root_keeps_minimal_top_level_surface() -> None:
    latentdna_root = _latentdna_root()
    observed = {
        path.name
        for path in latentdna_root.iterdir()
        if path.name != "__pycache__" and not path.name.startswith(".") and path.name != "AGENTS.md"
    }
    assert observed == {
        "README.md",
        "docs",
        "ops",
        "src",
        "tests",
        "workspaces",
    }


def test_latentdna_internal_cli_is_nested_under_src() -> None:
    latentdna_src = _latentdna_root() / "src"
    cli_dir = latentdna_src / "cli"
    cluster_dir = latentdna_src / "clusters"
    notebook_dir = latentdna_src / "notebooks"
    workspaces_dir = latentdna_src / "workspaces"
    assert not (latentdna_src / "api.py").exists()
    assert cli_dir.is_dir()
    assert (cli_dir / "__init__.py").is_file()
    assert (cli_dir / "app.py").is_file()
    assert (cli_dir / "commands").is_dir()
    assert cluster_dir.is_dir()
    assert (cluster_dir / "__init__.py").is_file()
    assert (cluster_dir / "fit.py").is_file()
    assert notebook_dir.is_dir()
    assert (notebook_dir / "__init__.py").is_file()
    assert (notebook_dir / "browser_runtime.py").is_file()
    assert (notebook_dir / "browser_runtime_compare.py").is_file()
    assert (notebook_dir / "browser_runtime_projection.py").is_file()
    assert (notebook_dir / "browser_runtime_support.py").is_file()
    assert (notebook_dir / "scaffold.py").is_file()
    assert (notebook_dir / "scaffold_panels.py").is_file()
    assert (notebook_dir / "scaffold_pages.py").is_file()
    assert (notebook_dir / "scaffold_selectors.py").is_file()
    assert workspaces_dir.is_dir()
    assert (workspaces_dir / "__init__.py").is_file()
    assert (workspaces_dir / "loader.py").is_file()
    assert (workspaces_dir / "paths.py").is_file()
    assert (workspaces_dir / "scaffold.py").is_file()
    assert (workspaces_dir / "validation.py").is_file()


def test_latentdna_package_data_uses_workspace_shape_globs() -> None:
    repo_root = _latentdna_root().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"dnadesign.latentdna"' in pyproject
    assert "workspaces/templates/*/*.yaml" in pyproject
    assert "workspaces/templates/*/*.md" in pyproject


def test_latentdna_integration_tests_use_descriptive_module_names() -> None:
    integrations_dir = _latentdna_root() / "tests" / "integrations"
    legacy_modules = sorted(path.name for path in integrations_dir.glob("test_phase*.py"))
    assert legacy_modules == []
