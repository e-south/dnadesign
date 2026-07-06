"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook.py

Marimo notebook writer for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

_TEMPLATE_RESOURCE = "notebook_template.py.txt"


def write_review_deliverables_notebook(path: Path) -> None:
    """Write a compact marimo notebook for Eco1 review deliverables."""

    template = Path(__file__).with_name(_TEMPLATE_RESOURCE).read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__MARIMO_VERSION__", version("marimo")), encoding="utf-8")
