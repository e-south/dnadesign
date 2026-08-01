"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/contracts/test_candidate.py

Tests the canonical candidate gate and ranking policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts.candidate import (
    candidate_meets_selection_gates,
    candidate_quality_blockers,
    select_best_candidate,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts.protocol import (
    DEFAULT_PROTOCOL,
)

from .._builders import _evidence, _ready, evaluate_metastudy


def test_candidate_policy_reproduces_the_canonical_decision() -> None:
    decision = evaluate_metastudy(_evidence(), readiness=_ready())

    selected = select_best_candidate(decision.evaluations)

    assert selected is not None
    assert selected.reduction == decision.selected_reduction
    assert candidate_meets_selection_gates(selected, protocol=DEFAULT_PROTOCOL)


def test_candidate_quality_blockers_are_deduplicated_in_evaluation_order() -> None:
    decision = evaluate_metastudy(_evidence(), readiness=_ready())
    first, second, *remaining = decision.evaluations
    evaluations = (
        replace(first, eligible=False, blockers=("observation_clipping_detected",)),
        replace(
            second,
            eligible=False,
            blockers=("observation_clipping_detected", "required_observation_count_zero"),
        ),
        *remaining,
    )

    assert candidate_quality_blockers(evaluations) == (
        "observation_clipping_detected",
        "required_observation_count_zero",
    )
