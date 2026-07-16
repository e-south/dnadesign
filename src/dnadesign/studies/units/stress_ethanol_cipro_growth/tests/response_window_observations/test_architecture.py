"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/response_window_observations/test_architecture.py

Architecture guards for study-owned response-window observations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations")

MODULE_BODY_LIMITS = {
    "__init__.py": 60,
    "__main__.py": 20,
    "aggregation.py": 200,
    "artifact.py": 220,
    "artifact_contract.py": 80,
    "artifact_io.py": 100,
    "artifact_label_source_validation.py": 130,
    "artifact_manifest.py": 190,
    "artifact_recomputation.py": 130,
    "artifact_repeat_validation.py": 150,
    "artifact_uncertainty_validation.py": 80,
    "artifact_validation.py": 120,
    "censoring.py": 160,
    "cli.py": 140,
    "contracts.py": 100,
    "display_contract.py": 80,
    "label_sources.py": 190,
    "policy.py": 420,
    "policy_contract.py": 90,
    "reader_bundle.py": 230,
    "reader_bundle_validation.py": 130,
    "repeat_adjudication.py": 160,
    "repeat_diagnostics.py": 80,
    "repeat_evidence.py": 250,
    "sensitivity.py": 100,
    "source_integrity.py": 110,
    "sources.py": 250,
    "uncertainty.py": 100,
    "validation.py": 190,
}


def _module_body_line_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0] == '"""':
        closing_line = next(index for index, line in enumerate(lines[1:], start=1) if line == '"""')
        return len(lines) - closing_line - 1
    return len(lines)


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def test_observation_modules_have_reviewed_size_caps() -> None:
    modules = {path.name: path for path in PACKAGE_ROOT.glob("*.py")}
    assert modules.keys() == MODULE_BODY_LIMITS.keys()

    offenders = {
        name: _module_body_line_count(path)
        for name, path in modules.items()
        if _module_body_line_count(path) > MODULE_BODY_LIMITS[name]
    }
    assert offenders == {}


@pytest.mark.parametrize("module_path", sorted(PACKAGE_ROOT.glob("*.py")), ids=lambda path: path.name)
def test_observations_do_not_import_reader_or_opal(module_path: Path) -> None:
    offenders = [
        f"{module_path.name}:{line}:{module}"
        for line, module in _imported_modules(module_path)
        if module == "reader"
        or module.startswith("reader.")
        or module == "dnadesign.opal"
        or module.startswith("dnadesign.opal.")
    ]
    assert offenders == []
