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

from dnadesign.baserender.src.render import junction_annealed_fragments

_ROOT = Path(junction_annealed_fragments.__file__).parent
_MODULE_BUDGETS = {
    "junction_annealed_fragments.py": 100,
    "junction_three_way_assembly.py": 240,
    "junction_three_way_detail.py": 320,
}
_SUPPORT_BUDGETS = {
    "__init__.py": 20,
    "annealed_panel.py": 300,
    "assembly_geometry.py": 160,
    "assembly_panel.py": 130,
    "assembly_stages.py": 160,
    "detail_geometry.py": 100,
    "detail_primitives.py": 80,
    "foundation.py": 180,
    "fragment_geometry.py": 120,
    "input_panel.py": 120,
    "primitives.py": 230,
    "preligation_junction.py": 160,
    "preligation_panel.py": 230,
    "product_panel.py": 180,
}


def test_junction_review_modules_remain_bounded() -> None:
    for filename, line_budget in _MODULE_BUDGETS.items():
        line_count = len((_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"
    support_root = _ROOT / "junction_review"
    for filename, line_budget in _SUPPORT_BUDGETS.items():
        line_count = len((support_root / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, (
            f"junction_review/{filename} exceeds its {line_budget}-line architecture budget"
        )


def test_junction_review_renderer_does_not_import_study_code() -> None:
    violations: list[str] = []
    paths = tuple(_ROOT / filename for filename in _MODULE_BUDGETS) + tuple(
        _ROOT / "junction_review" / filename for filename in _SUPPORT_BUDGETS
    )
    for path in paths:
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
