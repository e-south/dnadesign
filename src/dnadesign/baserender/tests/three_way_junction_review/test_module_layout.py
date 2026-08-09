"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/test_module_layout.py

Architecture budgets for the Junction nucleotide-review renderer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

from dnadesign.baserender.src.render import three_way_junction_review

_ROOT = Path(three_way_junction_review.__file__).parent
_MODULE_BUDGETS = {
    "junction_annealed_review.py": 150,
    "junction_nucleotide_drawing.py": 240,
    "junction_pairing_layout.py": 130,
    "junction_review_sections.py": 230,
    "three_way_junction_review.py": 180,
}


def test_junction_review_modules_remain_bounded() -> None:
    for filename, line_budget in _MODULE_BUDGETS.items():
        line_count = len((_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"


def test_junction_review_renderer_does_not_import_study_code() -> None:
    violations: list[str] = []
    for filename in _MODULE_BUDGETS:
        path = _ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets = (node.module,)
            else:
                continue
            violations.extend(
                target for target in targets if target == "dnadesign.studies" or target.startswith("dnadesign.studies.")
            )
    assert violations == []
