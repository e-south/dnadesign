"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/package/test_source_tree_contracts.py

Source-tree contracts for the construct package layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _construct_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "src" / "dnadesign" / "construct"
    raise RuntimeError("repo root not found")


def test_construct_root_keeps_progressive_disclosure_directories() -> None:
    construct_root = _construct_root()
    assert (construct_root / "README.md").is_file()
    assert (construct_root / "docs").is_dir()
    assert (construct_root / "docs" / "reference").is_dir()
    assert (construct_root / "src").is_dir()
    assert (construct_root / "tests").is_dir()
    assert (construct_root / "workspaces").is_dir()


def test_construct_root_keeps_minimal_top_level_surface() -> None:
    construct_root = _construct_root()
    observed = {
        path.name for path in construct_root.iterdir() if path.name != "__pycache__" and not path.name.startswith(".")
    }
    assert observed == {
        "README.md",
        "__init__.py",
        "__main__.py",
        "cli.py",
        "docs",
        "src",
        "tests",
        "workspaces",
    }


def test_construct_internal_cli_is_nested_under_src() -> None:
    construct_src = _construct_root() / "src"
    cli_dir = construct_src / "cli"
    assert cli_dir.is_dir()
    assert (cli_dir / "__init__.py").is_file()
    assert (cli_dir / "app.py").is_file()
    assert (cli_dir / "commands").is_dir()
