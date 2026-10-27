"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/optimizer.py

Deterministic exhaustive optimizer for YIU v4 payload selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.yiu.candidate_generation import CandidatePlan
from dnadesign.cruncher.yiu.domain_models import (
    ChosenLigationKey,
    LigationMismatchRationale,
    LigationPolicyDecision,
    LigationSearchState,
    OptimizationTraceSample,
    build_trace_sample,
)
from dnadesign.cruncher.yiu.errors import NoFeasiblePlanError
from dnadesign.cruncher.yiu.ligation_scoring import (
    CandidateLigationFilterResult,
    build_candidate_ligation_score,
    evaluate_hard_ligation_filter,
)
from dnadesign.cruncher.yiu.scoring import CandidateScore, ScorableMotif, score_candidate


@dataclass(frozen=True)
class OptimizerResult:
    winner: CandidatePlan
    score: CandidateScore
    candidate_count: int
    ligation_state: LigationSearchState
    chosen_ligation_key: ChosenLigationKey | None
    ligation_rationale: tuple[LigationMismatchRationale, ...]
    ligation_policy: LigationPolicyDecision
    trace_sample: OptimizationTraceSample
    trace: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _CandidateEvaluation:
    candidate: CandidatePlan
    score: CandidateScore
    ligation_key: ChosenLigationKey | None
    ligation_sort_key: tuple[int, int, int, bool, int] | None
    ligation_rationale: tuple[LigationMismatchRationale, ...]
    filter_result: CandidateLigationFilterResult


def _legacy_terminal_positions_used(candidate: CandidatePlan) -> int:
    return sum(1 for item in candidate.mismatch_positions if item in {0, 3})


def _common_key(candidate: CandidatePlan) -> tuple[object, ...]:
    return (
        candidate.midpoint_distance,
        -candidate.default_strand_preference_count,
        candidate.lexical_key,
    )


def _legacy_ordering_key(
    *,
    candidate: CandidatePlan,
    score: CandidateScore,
    pwm_effective: bool,
) -> tuple[object, ...]:
    legacy_common = (
        candidate.midpoint_distance,
        _legacy_terminal_positions_used(candidate),
        -candidate.default_strand_preference_count,
        candidate.lexical_key,
    )
    if pwm_effective:
        return (score.worst_loss, score.total_loss, *legacy_common)
    return legacy_common


def _secondary_ordering_key(
    evaluation: _CandidateEvaluation,
    *,
    pwm_effective: bool,
) -> tuple[object, ...]:
    if evaluation.ligation_sort_key is None:
        return _legacy_ordering_key(
            candidate=evaluation.candidate,
            score=evaluation.score,
            pwm_effective=pwm_effective,
        )
    common = _common_key(evaluation.candidate)
    if pwm_effective:
        return (evaluation.score.worst_loss, evaluation.score.total_loss, *evaluation.ligation_sort_key, *common)
    return (*evaluation.ligation_sort_key, *common)


def _tolerance_ordering_key(
    evaluation: _CandidateEvaluation,
    *,
    pwm_effective: bool,
) -> tuple[object, ...]:
    if evaluation.ligation_sort_key is None:
        return _legacy_ordering_key(
            candidate=evaluation.candidate,
            score=evaluation.score,
            pwm_effective=pwm_effective,
        )
    common = _common_key(evaluation.candidate)
    if pwm_effective:
        return (*evaluation.ligation_sort_key, evaluation.score.worst_loss, evaluation.score.total_loss, *common)
    return (*evaluation.ligation_sort_key, *common)


def _evaluate_candidate(
    *,
    candidate: CandidatePlan,
    reference_payload_sequence: str,
    reference_complement_sequence: str,
    scorable_motifs: tuple[ScorableMotif, ...],
    pwm_effective: bool,
    ligation_state: LigationSearchState,
    bad_pattern_heuristics: bool,
) -> _CandidateEvaluation:
    score = (
        score_candidate(
            candidate=candidate,
            reference_payload_sequence=reference_payload_sequence,
            reference_complement_sequence=reference_complement_sequence,
            scorable_motifs=scorable_motifs,
        )
        if pwm_effective
        else CandidateScore(worst_loss=0.0, total_loss=0.0)
    )
    if not ligation_state.enabled:
        return _CandidateEvaluation(
            candidate=candidate,
            score=score,
            ligation_key=None,
            ligation_sort_key=None,
            ligation_rationale=(),
            filter_result=CandidateLigationFilterResult(admissible=True, failure_fields=()),
        )
    ligation_score = build_candidate_ligation_score(
        candidate=candidate,
        ligation_profile=ligation_state.profile,
        bad_pattern_heuristics=bad_pattern_heuristics,
        force_bad_pattern_penalty=(
            ligation_state.selection_mode == "hard_ligation_filter" and not ligation_state.allow_tnna_like_overhangs
        ),
        reference_payload_sequence=reference_payload_sequence,
        reference_complement_sequence=reference_complement_sequence,
    )
    filter_result = (
        evaluate_hard_ligation_filter(
            ligation_score=ligation_score,
            max_worst_mismatch_class_tier=ligation_state.max_worst_mismatch_class_tier,
            max_middle_mismatch_count=ligation_state.max_middle_mismatch_count,
            allow_double_middle=ligation_state.allow_double_middle,
            allow_tnna_like_overhangs=ligation_state.allow_tnna_like_overhangs,
        )
        if ligation_state.selection_mode == "hard_ligation_filter"
        else CandidateLigationFilterResult(admissible=True, failure_fields=())
    )
    return _CandidateEvaluation(
        candidate=candidate,
        score=score,
        ligation_key=ligation_score.chosen_key,
        ligation_sort_key=ligation_score.key,
        ligation_rationale=ligation_score.mismatch_rationales,
        filter_result=filter_result,
    )


def _best_pwm_losses(evaluations: tuple[_CandidateEvaluation, ...]) -> tuple[float, float]:
    best = min(evaluations, key=lambda item: (item.score.worst_loss, item.score.total_loss))
    return best.score.worst_loss, best.score.total_loss


def _build_hard_filter_failure_message(
    *,
    evaluations: tuple[_CandidateEvaluation, ...],
    ligation_state: LigationSearchState,
) -> str:
    ordered_fields = (
        "max_worst_mismatch_class_tier",
        "max_middle_mismatch_count",
        "allow_double_middle",
        "allow_tnna_like_overhangs",
    )
    reason_counts = {
        field: sum(field in item.filter_result.failure_fields for item in evaluations)
        for field in ordered_fields
        if any(field in item.filter_result.failure_fields for item in evaluations)
    }
    suggestions: list[str] = []
    for field_name, _count in sorted(
        reason_counts.items(),
        key=lambda item: (-item[1], ordered_fields.index(item[0])),
    ):
        relevant_evaluations = tuple(
            item
            for item in evaluations
            if field_name in item.filter_result.failure_fields and item.ligation_key is not None
        )
        if field_name == "max_worst_mismatch_class_tier":
            suggested_value = min(item.ligation_key.worst_mismatch_class_tier for item in relevant_evaluations)
            suggestions.append(f"optimization.mismatches.max_worst_mismatch_class_tier={suggested_value}")
        elif field_name == "max_middle_mismatch_count":
            suggested_value = min(item.ligation_key.middle_mismatch_count for item in relevant_evaluations)
            suggestions.append(f"optimization.mismatches.max_middle_mismatch_count={suggested_value}")
        elif field_name == "allow_double_middle":
            suggestions.append("optimization.mismatches.allow_double_middle=true")
        elif field_name == "allow_tnna_like_overhangs":
            suggestions.append("optimization.mismatches.allow_tnna_like_overhangs=true")
        if len(suggestions) == 3:
            break
    suggestion_text = " Relax one or more of: " + ", ".join(suggestions) + "." if suggestions else ""
    return (
        "ligation_selection_mode=hard_ligation_filter removed all "
        f"{len(evaluations)} candidates for ligation_profile={ligation_state.profile}." + suggestion_text
    )


def _resolve_policy(
    *,
    evaluations: tuple[_CandidateEvaluation, ...],
    pwm_effective: bool,
    ligation_state: LigationSearchState,
) -> tuple[tuple[_CandidateEvaluation, ...], LigationPolicyDecision, tuple[bool, ...]]:
    before_count = len(evaluations)
    if ligation_state.selection_mode == "hard_ligation_filter":
        selected = tuple(item for item in evaluations if item.filter_result.admissible)
        if not selected:
            raise NoFeasiblePlanError(
                _build_hard_filter_failure_message(evaluations=evaluations, ligation_state=ligation_state)
            )
        return (
            selected,
            LigationPolicyDecision(
                selection_mode=ligation_state.selection_mode,
                filter_applied=len(selected) != before_count,
                candidate_count_before_filter=before_count,
                candidate_count_after_filter=len(selected),
                filtered_candidate_count=before_count - len(selected),
            ),
            tuple(True for _ in evaluations),
        )
    if ligation_state.selection_mode == "pwm_tolerance_then_ligation" and pwm_effective:
        best_worst_loss, best_total_loss = _best_pwm_losses(evaluations)
        gate_passes = tuple(
            item.score.worst_loss <= best_worst_loss + ligation_state.pwm_worst_loss_tolerance
            and item.score.total_loss <= best_total_loss + ligation_state.pwm_total_loss_tolerance
            for item in evaluations
        )
        selected = tuple(item for item, gate_pass in zip(evaluations, gate_passes, strict=True) if gate_pass)
        return (
            selected,
            LigationPolicyDecision(
                selection_mode=ligation_state.selection_mode,
                filter_applied=len(selected) != before_count,
                candidate_count_before_filter=before_count,
                candidate_count_after_filter=len(selected),
                filtered_candidate_count=before_count - len(selected),
            ),
            gate_passes,
        )
    return (
        evaluations,
        LigationPolicyDecision(
            selection_mode=ligation_state.selection_mode,
            filter_applied=False,
            candidate_count_before_filter=before_count,
            candidate_count_after_filter=before_count,
            filtered_candidate_count=0,
        ),
        tuple(True for _ in evaluations),
    )


def _trace_row(
    *,
    evaluation: _CandidateEvaluation,
    ligation_state: LigationSearchState,
    pwm_gate_passed: bool,
) -> dict[str, object]:
    candidate = evaluation.candidate
    trace_fields: dict[str, object] = {
        "ligation_profile": ligation_state.profile,
        "ligation_awareness_mode": ligation_state.awareness_mode,
        "ligation_selection_mode": ligation_state.selection_mode,
        "ligation_enabled": ligation_state.enabled,
        "ligation_legacy_mode": ligation_state.legacy_mode,
        "ligation_edge_positions_available": ligation_state.edge_positions_available,
        "ligation_edge_comparison_available": ligation_state.edge_comparison_available,
        "ligation_state_note": ligation_state.state_note,
        "midpoint_distance": candidate.midpoint_distance,
        "default_strand_preference_count": candidate.default_strand_preference_count,
        "lexical_key": candidate.lexical_key,
    }
    if evaluation.ligation_key is not None:
        trace_fields.update(
            {
                "worst_mismatch_class_tier": evaluation.ligation_key.worst_mismatch_class_tier,
                "total_mismatch_class_tier": evaluation.ligation_key.total_mismatch_class_tier,
                "middle_mismatch_count": evaluation.ligation_key.middle_mismatch_count,
                "double_middle_flag": evaluation.ligation_key.double_middle_flag,
                "bad_pattern_penalty": evaluation.ligation_key.bad_pattern_penalty,
                "canonical_mismatch_classes": [item.canonical_mismatch_class for item in evaluation.ligation_rationale],
                "position_classes": [item.position_class for item in evaluation.ligation_rationale],
                "ligation_filter_passed": evaluation.filter_result.admissible,
                "ligation_filter_failures": list(evaluation.filter_result.failure_fields),
            }
        )
    if ligation_state.selection_mode == "pwm_tolerance_then_ligation":
        trace_fields["ligation_pwm_gate_passed"] = pwm_gate_passed
    return {
        "junction_start": candidate.junction_start,
        "mismatch_positions": list(candidate.mismatch_positions),
        "mutated_strands": [entry.mutated_strand for entry in candidate.mutations],
        "mutated_bases": [entry.mutated_base for entry in candidate.mutations],
        "worst_loss": evaluation.score.worst_loss,
        "total_loss": evaluation.score.total_loss,
        **trace_fields,
    }


def select_best_candidate(
    *,
    candidates: tuple[CandidatePlan, ...],
    reference_payload_sequence: str,
    reference_complement_sequence: str,
    scorable_motifs: tuple[ScorableMotif, ...],
    pwm_effective: bool,
    ligation_state: LigationSearchState,
    bad_pattern_heuristics: bool = False,
    trace_limit: int = 32,
) -> OptimizerResult:
    if not candidates:
        raise ValueError("select_best_candidate requires at least one candidate")
    if trace_limit < 0:
        raise ValueError("trace_limit must be non-negative")
    evaluations = tuple(
        _evaluate_candidate(
            candidate=candidate,
            reference_payload_sequence=reference_payload_sequence,
            reference_complement_sequence=reference_complement_sequence,
            scorable_motifs=scorable_motifs,
            pwm_effective=pwm_effective,
            ligation_state=ligation_state,
            bad_pattern_heuristics=bad_pattern_heuristics,
        )
        for candidate in candidates
    )
    selected_evaluations, ligation_policy, pwm_gate_passes = _resolve_policy(
        evaluations=evaluations,
        pwm_effective=pwm_effective,
        ligation_state=ligation_state,
    )
    ordering_fn = (
        _tolerance_ordering_key
        if ligation_state.selection_mode == "pwm_tolerance_then_ligation" and pwm_effective
        else _secondary_ordering_key
    )
    best_evaluation = min(selected_evaluations, key=lambda item: ordering_fn(item, pwm_effective=pwm_effective))
    trace: list[dict[str, object]] = []
    for evaluation, pwm_gate_passed in zip(evaluations, pwm_gate_passes, strict=True):
        if len(trace) < trace_limit:
            trace.append(
                _trace_row(
                    evaluation=evaluation,
                    ligation_state=ligation_state,
                    pwm_gate_passed=pwm_gate_passed,
                )
            )
    trace_sample = build_trace_sample(
        candidate_count=len(candidates),
        sample_limit=trace_limit,
        sampled_count=len(trace),
    )
    return OptimizerResult(
        winner=best_evaluation.candidate,
        score=best_evaluation.score,
        candidate_count=len(candidates),
        ligation_state=ligation_state,
        chosen_ligation_key=best_evaluation.ligation_key,
        ligation_rationale=best_evaluation.ligation_rationale,
        ligation_policy=ligation_policy,
        trace_sample=trace_sample,
        trace=tuple(trace),
    )
