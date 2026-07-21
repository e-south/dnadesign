"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_report_structure.py

Tests for the response-metastudy report information architecture.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.reporting.report import (
    EVIDENCE_SECTION_ORDER,
    _validate_evidence_section_order,
)


def _headings(*sections: str) -> list[str]:
    return [f"## {section}" for section in sections]


def test_report_evidence_sections_have_one_objective_neutral_order() -> None:
    _validate_evidence_section_order(_headings(*EVIDENCE_SECTION_ORDER))


@pytest.mark.parametrize(
    "lines",
    [
        _headings(*reversed(EVIDENCE_SECTION_ORDER)),
        _headings(*EVIDENCE_SECTION_ORDER, EVIDENCE_SECTION_ORDER[-1]),
        _headings(*EVIDENCE_SECTION_ORDER[:-1]),
    ],
)
def test_report_evidence_section_validation_rejects_drift(lines: list[str]) -> None:
    with pytest.raises(ValueError, match="Report"):
        _validate_evidence_section_order(lines)
