"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/publication/_builders.py

Publishes canonical synthetic metastudy evidence for publication-owner tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    MetastudyDecision,
    ProfileEvidence,
    evaluate_sensitivity,
    publish_metastudy,
)

from .._builders import _evidence
from ..evidence._builders import _complete_sensitivity_evidence, _sensitivity_coverages


def _publish_selected(decision: MetastudyDecision, destination: Path, *, evidence=None) -> Path:
    primary = _evidence() if evidence is None else evidence
    return _publish_evaluated(decision, destination, evidence=primary)


def _publish_evaluated(
    decision: MetastudyDecision,
    destination: Path,
    *,
    evidence: tuple[ProfileEvidence, ...],
) -> Path:
    sensitivity = _complete_sensitivity_evidence(_evidence())
    return publish_metastudy(
        decision,
        destination,
        primary_evidence=evidence,
        sensitivity_evidence=sensitivity,
        sensitivity_evaluations=evaluate_sensitivity(sensitivity),
        sensitivity_coverages=_sensitivity_coverages(sensitivity, decision.materialization_attempts),
    )
