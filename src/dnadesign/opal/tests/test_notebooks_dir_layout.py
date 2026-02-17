"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_notebooks_dir_layout.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def test_notebooks_dir_contains_only_marimo_notebooks() -> None:
    nb_dir = Path("src/dnadesign/opal/notebooks")
    py_files = sorted(nb_dir.glob("*.py"))
    assert py_files, "Expected at least one notebook in src/dnadesign/opal/notebooks."

    for path in py_files:
        txt = path.read_text()
        assert "marimo.App" in txt, f"Non-notebook helper found in notebooks dir: {path}"
