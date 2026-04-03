"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/optimizer.py

Deterministic exhaustive optimizer for YIU v4 payload selection.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.yiu.candidate_generation import CandidatePlan
from dnadesign.cruncher.yiu.scoring import CandidateScore, ScorableMotif, score_candidate


@dataclass(frozen=True)
class OptimizerResult:
    winner: CandidatePlan
    score: CandidateScore
    candidate_count: int
    trace: tuple[dict[str, object], ...]


def _ordering_key(
    *,
    candidate: CandidatePlan,
    score: CandidateScore,
    pwm_effective: bool,
) -> tuple[object, ...]:
    common = (
        candidate.midpoint_distance,
        candidate.body_length_balance,
        candidate.terminal_positions_used,
        -candidate.default_strand_preference_count,
        candidate.lexical_key,
    )
    if pwm_effective:
        return (score.worst_loss, score.total_loss, *common)
    return common


def select_best_candidate(
    *,
    candidates: tuple[CandidatePlan, ...],
    reference_payload_sequence: str,
    reference_complement_sequence: str,
    scorable_motifs: tuple[ScorableMotif, ...],
    pwm_effective: bool,
    trace_limit: int = 32,
) -> OptimizerResult:
    if not candidates:
        raise ValueError("select_best_candidate requires at least one candidate")
    best_candidate: CandidatePlan | None = None
    best_score: CandidateScore | None = None
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
        key = _ordering_key(candidate=candidate, score=score, pwm_effective=pwm_effective)
        if len(trace) < trace_limit:
            trace.append(
                {
                    "junction_start": candidate.junction_start,
                    "mismatch_positions": list(candidate.mismatch_positions),
                    "mutated_strands": [entry.mutated_strand for entry in candidate.mutations],
                    "mutated_bases": [entry.mutated_base for entry in candidate.mutations],
                    "worst_loss": score.worst_loss,
                    "total_loss": score.total_loss,
                    "midpoint_distance": candidate.midpoint_distance,
                    "body_length_balance": candidate.body_length_balance,
                    "terminal_positions_used": candidate.terminal_positions_used,
                    "default_strand_preference_count": candidate.default_strand_preference_count,
                    "lexical_key": candidate.lexical_key,
                }
            )
        if best_key is None or key < best_key:
            best_candidate = candidate
            best_score = score
            best_key = key
    assert best_candidate is not None
    assert best_score is not None
    return OptimizerResult(
        winner=best_candidate,
        score=best_score,
        candidate_count=len(candidates),
        trace=tuple(trace),
    )
