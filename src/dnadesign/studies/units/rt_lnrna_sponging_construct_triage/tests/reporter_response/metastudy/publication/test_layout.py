"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/publication/test_layout.py

Architecture guards for the study-owned publication package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    publication,
)


def test_publication_package_is_small_and_explicit() -> None:
    package_root = Path(publication.__file__).parent
    modules = {path.name for path in package_root.glob("*.py")}

    assert modules == {"__init__.py", "report.py", "service.py", "verification.py"}
    assert set(publication.__all__) == {"publish_metastudy", "verify_publication"}
    assert not (package_root.parent / "publication.py").exists()


def test_publication_modules_respect_cohesion_budgets() -> None:
    package_root = Path(publication.__file__).parent
    budgets = {
        "__init__.py": 10,
        "report.py": 50,
        "service.py": 180,
        "verification.py": 300,
    }

    for name, budget in budgets.items():
        lines = (package_root / name).read_text(encoding="utf-8").splitlines()
        assert len(lines) <= budget, f"{name} has {len(lines)} lines; expected <= {budget}"


def test_study_publication_uses_only_the_generic_artifact_facade() -> None:
    service_path = Path(publication.__file__).parent / "service.py"
    tree = ast.parse(service_path.read_text(encoding="utf-8"))
    artifact_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dnadesign.artifacts")
    }

    assert artifact_imports == {"dnadesign.artifacts"}


def test_publication_dependency_direction_is_one_way() -> None:
    package_root = Path(publication.__file__).parent

    def local_imports(name: str) -> set[str]:
        tree = ast.parse((package_root / name).read_text(encoding="utf-8"))
        return {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is not None
        }

    assert local_imports("report.py") == set()
    assert local_imports("verification.py") == {"report"}
    assert local_imports("service.py") == {"report", "verification"}
    assert local_imports("__init__.py") == {"service", "verification"}
