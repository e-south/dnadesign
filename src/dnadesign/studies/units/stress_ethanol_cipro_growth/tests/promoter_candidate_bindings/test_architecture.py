"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/promoter_candidate_bindings/test_architecture.py

Architecture boundaries for the study-level candidate-binding package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings")


@pytest.mark.parametrize("module_path", sorted(PACKAGE_ROOT.glob("*.py")), ids=lambda path: path.name)
def test_binding_modules_stay_bounded(module_path: Path) -> None:
    assert len(module_path.read_text(encoding="utf-8").splitlines()) <= 300


def test_package_is_study_scoped_and_metric_neutral() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.glob("*.py"))

    for forbidden in (
        "decision.opal",
        "X_COLUMN_ID",
        "x_readiness",
        "exact_reader_design_alias",
    ):
        assert forbidden not in source


def test_source_adapters_do_not_infer_namespaces_from_alias_prefixes() -> None:
    source = (PACKAGE_ROOT / "source_adapters.py").read_text(encoding="utf-8")
    assert '.startswith("pDual-10")' not in source
    assert ".startswith('pDual-10')" not in source
