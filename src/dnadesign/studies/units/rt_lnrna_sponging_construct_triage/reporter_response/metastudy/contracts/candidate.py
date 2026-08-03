"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/candidate.py

Candidate-evaluation contract and deterministic selection policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from ._values import MetastudyContractError, _unique_text
from .protocol import DEFAULT_PROTOCOL, MetastudyProtocol, Window

_QUALITY_BLOCKERS = frozenset(
    {
        "required_observation_count_zero",
        "observation_overflow_detected",
        "observation_clipping_detected",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Derived metrics for one selectable primary-cohort window."""

    reduction: Window
    eligible_experiment_count: int
    worst_experiment_control_separation: float | None
    repeated_anchor_drift: float
    within_acquisition_observation_range: float
    growth_phase_start: float
    growth_phase_end: float
    anchor_ordered_acquisition_count: int
    co_measured_anchor_acquisition_count: int
    loo_same_or_adjacent_fraction: float
    eligible: bool
    blockers: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.reduction not in DEFAULT_PROTOCOL.candidate_windows_h:
            raise MetastudyContractError("candidate evaluation reduction is undeclared")
        for name in (
            "eligible_experiment_count",
            "anchor_ordered_acquisition_count",
            "co_measured_anchor_acquisition_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MetastudyContractError(f"{name} must be a non-negative integer")
        for name in (
            "repeated_anchor_drift",
            "within_acquisition_observation_range",
            "growth_phase_start",
            "growth_phase_end",
            "loo_same_or_adjacent_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise MetastudyContractError(f"{name} must be finite")
        if self.worst_experiment_control_separation is not None:
            value = self.worst_experiment_control_separation
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise MetastudyContractError("worst_experiment_control_separation must be finite or null")
        if not 0.0 <= self.loo_same_or_adjacent_fraction <= 1.0:
            raise MetastudyContractError("loo_same_or_adjacent_fraction must be between zero and one")
        _unique_text(self.blockers, label="candidate blockers", allow_empty=self.eligible)
        if self.eligible and self.blockers:
            raise MetastudyContractError("eligible candidate cannot contain blockers")
        if not self.eligible and not self.blockers:
            raise MetastudyContractError("ineligible candidate requires blockers")
        _unique_text(self.limitations, label="candidate limitations", allow_empty=True)


def candidate_quality_blockers(evaluations: Iterable[CandidateEvaluation]) -> tuple[str, ...]:
    """Return canonical fail-closed data-quality blockers in evaluation order."""

    return tuple(
        dict.fromkeys(
            blocker for evaluation in evaluations for blocker in evaluation.blockers if blocker in _QUALITY_BLOCKERS
        )
    )


def candidate_selection_key(row: CandidateEvaluation) -> tuple[float, float, float, float, float]:
    """Return the predeclared lexicographic ordering for eligible candidates."""

    has_reference = row.worst_experiment_control_separation is not None
    return (
        0.0 if has_reference else 1.0,
        -row.worst_experiment_control_separation if has_reference else 0.0,
        (float("inf") if "repeated_reference_drift_not_estimable" in row.limitations else row.repeated_anchor_drift),
        row.within_acquisition_observation_range,
        row.reduction[1],
    )


def select_best_candidate(evaluations: Iterable[CandidateEvaluation]) -> CandidateEvaluation | None:
    """Select the canonical winner, or return ``None`` when no candidate is eligible."""

    eligible = tuple(row for row in evaluations if row.eligible)
    return min(eligible, key=candidate_selection_key) if eligible else None


def candidate_meets_selection_gates(
    candidate: CandidateEvaluation,
    *,
    protocol: MetastudyProtocol,
) -> bool:
    """Return whether one candidate satisfies the descriptive selection gates."""

    return (
        candidate.eligible
        and candidate.eligible_experiment_count >= protocol.minimum_kinetic_experiments
        and (
            candidate.worst_experiment_control_separation is None or candidate.worst_experiment_control_separation > 0.0
        )
        and candidate.growth_phase_start >= protocol.growth_phase_start_minimum
        and protocol.growth_phase_end_minimum <= candidate.growth_phase_end <= protocol.growth_phase_end_maximum
    )


__all__ = [
    "CandidateEvaluation",
    "candidate_meets_selection_gates",
    "candidate_quality_blockers",
    "candidate_selection_key",
    "select_best_candidate",
]
