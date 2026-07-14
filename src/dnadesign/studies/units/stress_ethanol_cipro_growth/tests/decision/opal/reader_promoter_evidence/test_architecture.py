"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/reader_promoter_evidence/test_architecture.py

Enforce architecture boundaries for the study-owned Reader evidence adapter.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence")


@pytest.mark.parametrize("module_path", sorted(PACKAGE_ROOT.glob("*.py")), ids=lambda path: path.name)
def test_reader_evidence_modules_stay_bounded(module_path: Path) -> None:
    assert len(module_path.read_text(encoding="utf-8").splitlines()) <= 320


def test_study_adapter_does_not_import_reader_or_generic_opal_internals() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.glob("*.py"))

    assert "from reader" not in source
    assert "import reader" not in source
    assert "dnadesign.opal.src" not in source
