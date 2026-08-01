"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/test_materialize_layout.py

Architecture contracts for the reporter-response materialization package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import materialize

_METASTUDY_ROOT = Path(__file__).parents[3] / "reporter_response" / "metastudy"
_MATERIALIZE_ROOT = _METASTUDY_ROOT / "materialize"
_EXPECTED_MODULES = {"__init__.py", "models.py", "profiles.py", "reference.py", "service.py", "temporal.py"}
_LINE_BUDGETS = {
    "__init__.py": 25,
    "models.py": 80,
    "profiles.py": 410,
    "reference.py": 70,
    "service.py": 380,
    "temporal.py": 280,
}


def test_materialization_package_has_one_semantic_module_per_owner() -> None:
    assert not (_METASTUDY_ROOT / "materialize.py").exists()
    assert {path.name for path in _MATERIALIZE_ROOT.glob("*.py")} == _EXPECTED_MODULES
    for filename, line_budget in _LINE_BUDGETS.items():
        line_count = len((_MATERIALIZE_ROOT / filename).read_text(encoding="utf-8").splitlines())
        assert line_count <= line_budget, f"{filename} exceeds its {line_budget}-line architecture budget"


def test_materialization_facade_exposes_only_supported_names() -> None:
    assert set(materialize.__all__) == {"MaterializationReadiness", "materialize_record_evidence"}


def test_materialization_production_modules_do_not_import_the_facade() -> None:
    for path in _MATERIALIZE_ROOT.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        facade_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
        ]
        assert not facade_imports, f"{path.name} must import the owning materialization leaf"
