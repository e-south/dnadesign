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
_EXPECTED_MODULES = {
    "__init__.py",
    "identities.py",
    "models.py",
    "profile_building.py",
    "reductions.py",
    "reference.py",
    "service.py",
    "temporal.py",
    "uncertainty.py",
}
_LINE_BUDGETS = {
    "__init__.py": 25,
    "identities.py": 80,
    "models.py": 80,
    "profile_building.py": 300,
    "reductions.py": 100,
    "reference.py": 70,
    "service.py": 380,
    "temporal.py": 280,
    "uncertainty.py": 80,
}
_ALLOWED_SIBLING_IMPORTS = {
    "identities.py": set(),
    "models.py": set(),
    "profile_building.py": {"reference", "temporal", "uncertainty"},
    "reductions.py": {"identities", "profile_building"},
    "reference.py": set(),
    "service.py": {"identities", "models", "reductions"},
    "temporal.py": set(),
    "uncertainty.py": {"temporal"},
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


def test_materialization_dependencies_follow_the_semantic_owner_graph() -> None:
    for filename, allowed in _ALLOWED_SIBLING_IMPORTS.items():
        path = _MATERIALIZE_ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        sibling_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is not None
        }
        assert sibling_imports <= allowed, (
            f"{filename} has reverse or undeclared dependencies: {sibling_imports - allowed}"
        )
