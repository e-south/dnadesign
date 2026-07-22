"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/notebooks/test_notebooks_dir_layout.py

Tests OPAL notebook directory layout contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def test_notebooks_dir_has_no_checked_in_runtime_notebooks() -> None:
    nb_dir = Path("src/dnadesign/opal/notebooks")
    assert sorted(nb_dir.glob("*.py")) == []
    assert (nb_dir / "api" / "generated.py").is_file()
