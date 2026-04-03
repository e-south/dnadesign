"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/scoring.py

PWM-aware scoring primitives for YIU v4 optimization.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dnadesign.cruncher.yiu.candidate_generation import CandidatePlan
from dnadesign.cruncher.yiu.spec_models import YiuPwmMotifInstanceV1

_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
_LOG_FLOOR = 1e-9


@dataclass(frozen=True)
class ScorableMotif:
    motif: YiuPwmMotifInstanceV1
    native_score: float
    covered_payload_indices: frozenset[int]


@dataclass(frozen=True)
class CandidateScore:
    worst_loss: float
    total_loss: float


def apply_candidate_sequences(
    *,
    candidate: CandidatePlan,
    reference_payload_sequence: str,
    reference_complement_sequence: str,
) -> tuple[str, str]:
    payload_chars = list(reference_payload_sequence)
    complement_chars = list(reference_complement_sequence)
    for mutation in candidate.mutations:
        if mutation.mutated_strand == "payload":
            payload_chars[mutation.payload_index] = mutation.mutated_base
        else:
            complement_chars[mutation.payload_index] = mutation.mutated_base
    return "".join(payload_chars), "".join(complement_chars)


def extract_motif_sequence(
    *,
    payload_sequence: str,
    complement_sequence: str,
    motif: YiuPwmMotifInstanceV1,
) -> str:
    if motif.reference_strand == "+":
        return payload_sequence[motif.start : motif.end]
    return complement_sequence[motif.start : motif.end][::-1]


def pwm_log_likelihood(sequence: str, motif: YiuPwmMotifInstanceV1) -> float:
    score = 0.0
    for base, row in zip(sequence, motif.probabilities.rows, strict=True):
        score += math.log(max(float(row[_BASE_INDEX[base]]), _LOG_FLOOR))
    return score


def build_scorable_motifs(
    *,
    reference_payload_sequence: str,
    reference_complement_sequence: str,
    motifs: list[YiuPwmMotifInstanceV1],
) -> tuple[ScorableMotif, ...]:
    scored: list[ScorableMotif] = []
    for motif in motifs:
        native_seq = extract_motif_sequence(
            payload_sequence=reference_payload_sequence,
            complement_sequence=reference_complement_sequence,
            motif=motif,
        )
        scored.append(
            ScorableMotif(
                motif=motif,
                native_score=pwm_log_likelihood(native_seq, motif),
                covered_payload_indices=frozenset(range(motif.start, motif.end)),
            )
        )
    return tuple(scored)


def score_candidate(
    *,
    candidate: CandidatePlan,
    reference_payload_sequence: str,
    reference_complement_sequence: str,
    scorable_motifs: tuple[ScorableMotif, ...],
) -> CandidateScore:
    mutated_indices = {mutation.payload_index for mutation in candidate.mutations}
    if not mutated_indices or not scorable_motifs:
        return CandidateScore(worst_loss=0.0, total_loss=0.0)
    selected_payload, selected_complement = apply_candidate_sequences(
        candidate=candidate,
        reference_payload_sequence=reference_payload_sequence,
        reference_complement_sequence=reference_complement_sequence,
    )
    losses: list[float] = []
    for scored_motif in scorable_motifs:
        if mutated_indices.isdisjoint(scored_motif.covered_payload_indices):
            continue
        sequence = extract_motif_sequence(
            payload_sequence=selected_payload,
            complement_sequence=selected_complement,
            motif=scored_motif.motif,
        )
        candidate_score = pwm_log_likelihood(sequence, scored_motif.motif)
        losses.append(max(0.0, scored_motif.native_score - candidate_score))
    if not losses:
        return CandidateScore(worst_loss=0.0, total_loss=0.0)
    return CandidateScore(worst_loss=max(losses), total_loss=sum(losses))
