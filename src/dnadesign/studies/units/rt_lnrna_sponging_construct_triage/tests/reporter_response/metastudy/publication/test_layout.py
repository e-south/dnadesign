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


def _logical_source_line_count(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    physical_lines = len(source.splitlines())
    if not tree.body or not isinstance(tree.body[0], ast.Expr):
        return physical_lines
    value = tree.body[0].value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return physical_lines
    return physical_lines - (tree.body[0].end_lineno - tree.body[0].lineno)


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
        line_count = _logical_source_line_count(package_root / name)
        assert line_count <= budget, f"{name} has {line_count} logical lines; expected <= {budget}"


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
