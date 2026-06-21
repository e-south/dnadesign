"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/qc/metrics.py

Target-relative pre-MSA sequence metrics for Eco1 roster rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from Bio.Align import PairwiseAligner


@dataclass(frozen=True)
class PairwiseTargetMetrics:
    """One target-vs-provider pairwise alignment summary."""

    query_coverage: float
    identity_to_target: float


def pairwise_target_metrics(*, sequence: str, target_sequence: str) -> PairwiseTargetMetrics:
    """Return target coverage and identity from one global protein alignment."""

    if not target_sequence:
        raise ValueError("target_sequence must be non-empty")
    if not sequence:
        return PairwiseTargetMetrics(query_coverage=0.0, identity_to_target=0.0)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(target_sequence, sequence)[0]
    target_covered = 0
    aligned_pairs = 0
    matches = 0
    for (target_start, target_end), (sequence_start, sequence_end) in zip(*alignment.aligned):
        target_covered += target_end - target_start
        target_slice = target_sequence[target_start:target_end]
        sequence_slice = sequence[sequence_start:sequence_end]
        for target_aa, sequence_aa in zip(target_slice, sequence_slice):
            aligned_pairs += 1
            if target_aa == sequence_aa:
                matches += 1
    return PairwiseTargetMetrics(
        query_coverage=float(target_covered) / float(len(target_sequence)),
        identity_to_target=float(matches) / float(aligned_pairs) if aligned_pairs else 0.0,
    )


def range_status(value: float, lower: float, upper: float) -> str:
    """Classify a numeric value against an inclusive two-sided range."""

    if value < lower:
        return "below_declared_range"
    if value > upper:
        return "above_declared_range"
    return "within_declared_range"
