"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/optimizer.py

Deterministic exhaustive optimizer for YIU v4 payload selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.yiu.candidate_generation import CandidatePlan
from dnadesign.cruncher.yiu.domain_models import ChosenLigationKey, LigationMismatchRationale
from dnadesign.cruncher.yiu.ligation_scoring import LigationProfile, build_candidate_ligation_score
from dnadesign.cruncher.yiu.scoring import CandidateScore, ScorableMotif, score_candidate


@dataclass(frozen=True)
class OptimizerResult:
    winner: CandidatePlan
    score: CandidateScore
    candidate_count: int
    chosen_ligation_key: ChosenLigationKey | None
    ligation_rationale: tuple[LigationMismatchRationale, ...]
    trace: tuple[dict[str, object], ...]


def _legacy_terminal_positions_used(candidate: CandidatePlan) -> int:
    return sum(1 for item in candidate.mismatch_positions if item in {0, 3})


def _ordering_key(
    *,
    candidate: CandidatePlan,
    score: CandidateScore,
    pwm_effective: bool,
    ligation_profile: LigationProfile,
    ligation_awareness_mode: str,
    bad_pattern_heuristics: bool,
    reference_payload_sequence: str,
    reference_complement_sequence: str,
) -> tuple[tuple[object, ...], ChosenLigationKey | None, tuple[LigationMismatchRationale, ...], dict[str, object]]:
    ligation_enabled = ligation_awareness_mode == "secondary" and ligation_profile != "none"
    common = (
        candidate.midpoint_distance,
        -candidate.default_strand_preference_count,
        candidate.lexical_key,
    )
    trace_fields: dict[str, object] = {
        "midpoint_distance": candidate.midpoint_distance,
        "default_strand_preference_count": candidate.default_strand_preference_count,
        "lexical_key": candidate.lexical_key,
    }
    if ligation_enabled:
        ligation_score = build_candidate_ligation_score(
            candidate=candidate,
            ligation_profile=ligation_profile,
            bad_pattern_heuristics=bad_pattern_heuristics,
            reference_payload_sequence=reference_payload_sequence,
            reference_complement_sequence=reference_complement_sequence,
        )
        trace_fields.update(
            {
                "worst_mismatch_class_tier": ligation_score.chosen_key.worst_mismatch_class_tier,
                "total_mismatch_class_tier": ligation_score.chosen_key.total_mismatch_class_tier,
                "middle_mismatch_count": ligation_score.chosen_key.middle_mismatch_count,
                "double_middle_flag": ligation_score.chosen_key.double_middle_flag,
                "bad_pattern_penalty": ligation_score.chosen_key.bad_pattern_penalty,
                "canonical_mismatch_classes": [
                    item.canonical_mismatch_class for item in ligation_score.mismatch_rationales
                ],
                "position_classes": [item.position_class for item in ligation_score.mismatch_rationales],
            }
        )
        if pwm_effective:
            return (
                (score.worst_loss, score.total_loss, *ligation_score.key, *common),
                ligation_score.chosen_key,
                ligation_score.mismatch_rationales,
                trace_fields,
            )
        return (
            (*ligation_score.key, *common),
            ligation_score.chosen_key,
            ligation_score.mismatch_rationales,
            trace_fields,
        )
    legacy_common = (
        candidate.midpoint_distance,
        _legacy_terminal_positions_used(candidate),
        -candidate.default_strand_preference_count,
        candidate.lexical_key,
    )
    if pwm_effective:
        return (score.worst_loss, score.total_loss, *legacy_common), None, (), trace_fields
    return legacy_common, None, (), trace_fields


def select_best_candidate(
    *,
    candidates: tuple[CandidatePlan, ...],
    reference_payload_sequence: str,
    reference_complement_sequence: str,
    scorable_motifs: tuple[ScorableMotif, ...],
    pwm_effective: bool,
    ligation_profile: LigationProfile = "none",
    ligation_awareness_mode: str = "disabled",
    bad_pattern_heuristics: bool = False,
    trace_limit: int = 32,
) -> OptimizerResult:
    if not candidates:
        raise ValueError("select_best_candidate requires at least one candidate")
    best_candidate: CandidatePlan | None = None
    best_score: CandidateScore | None = None
    best_ligation_key: ChosenLigationKey | None = None
    best_ligation_rationale: tuple[LigationMismatchRationale, ...] = ()
    best_key: tuple[object, ...] | None = None
    trace: list[dict[str, object]] = []
    for candidate in candidates:
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
        key, ligation_key, ligation_rationale, trace_fields = _ordering_key(
            candidate=candidate,
            score=score,
            pwm_effective=pwm_effective,
            ligation_profile=ligation_profile,
            ligation_awareness_mode=ligation_awareness_mode,
            bad_pattern_heuristics=bad_pattern_heuristics,
            reference_payload_sequence=reference_payload_sequence,
            reference_complement_sequence=reference_complement_sequence,
        )
        if len(trace) < trace_limit:
            trace.append(
                {
                    "junction_start": candidate.junction_start,
                    "mismatch_positions": list(candidate.mismatch_positions),
                    "mutated_strands": [entry.mutated_strand for entry in candidate.mutations],
                    "mutated_bases": [entry.mutated_base for entry in candidate.mutations],
                    "worst_loss": score.worst_loss,
                    "total_loss": score.total_loss,
                    **trace_fields,
                }
            )
        if best_key is None or key < best_key:
            best_candidate = candidate
            best_score = score
            best_ligation_key = ligation_key
            best_ligation_rationale = ligation_rationale
            best_key = key
    assert best_candidate is not None
    assert best_score is not None
    return OptimizerResult(
        winner=best_candidate,
        score=best_score,
        candidate_count=len(candidates),
        chosen_ligation_key=best_ligation_key,
        ligation_rationale=best_ligation_rationale,
        trace=tuple(trace),
    )
