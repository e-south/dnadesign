"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_motifs.py

Motif scoring for payload binding-site semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

from .models import MsdPayloadMotifAlignment, MsdRegionIngestError
from .payload_binding_models import MotifModel, PayloadBindingCatalog, PayloadMember
from .payload_binding_utils import reverse_complement


def motif_alignments(
    primary: str,
    *,
    member: PayloadMember | None,
    catalog: PayloadBindingCatalog | None,
) -> tuple[MsdPayloadMotifAlignment, ...]:
    if catalog is None:
        return ()
    motif_model_id = member.motif_model_id if member is not None else catalog.default_motif_model_id
    if motif_model_id is None:
        return ()
    motif = catalog.motif_models[motif_model_id]
    if member is not None and len(primary) < motif.width:
        span = member.retained_parent_span_0
        if span["end"] - span["start"] != len(primary):
            return ()
        return (score_fixed_motif_span(primary, motif=motif, motif_start=span["start"]),)
    if len(primary) < motif.width:
        return ()
    return (score_best_motif_window(primary, motif=motif),)


def score_best_motif_window(primary: str, *, motif: MotifModel) -> MsdPayloadMotifAlignment:
    best: tuple[float, int, str, str] | None = None
    for start in range(0, len(primary) - motif.width + 1):
        window = primary[start : start + motif.width]
        for strand, sequence in (("+", window), ("-", reverse_complement(window))):
            score = score_sequence(sequence, motif=motif, motif_start=0)
            if best is None or score > best[0]:
                best = (score, start, strand, sequence)
    if best is None:
        raise MsdRegionIngestError("Cannot score empty motif window.")
    score, start, strand, sequence = best
    return alignment_payload(
        motif=motif,
        motif_start=0,
        payload_start=start,
        sequence=sequence,
        strand=strand,
        score=score,
    )


def score_fixed_motif_span(primary: str, *, motif: MotifModel, motif_start: int) -> MsdPayloadMotifAlignment:
    reverse = reverse_complement(primary)
    candidates = (
        ("+", primary, score_sequence(primary, motif=motif, motif_start=motif_start)),
        ("-", reverse, score_sequence(reverse, motif=motif, motif_start=motif_start)),
    )
    strand, sequence, score = max(candidates, key=lambda item: item[2])
    return alignment_payload(
        motif=motif,
        motif_start=motif_start,
        payload_start=0,
        sequence=sequence,
        strand=strand,
        score=score,
    )


def alignment_payload(
    *,
    motif: MotifModel,
    motif_start: int,
    payload_start: int,
    sequence: str,
    strand: str,
    score: float,
) -> MsdPayloadMotifAlignment:
    motif_end = motif_start + len(sequence)
    consensus_sequence = motif.consensus[motif_start:motif_end]
    consensus_score = score_sequence(consensus_sequence, motif=motif, motif_start=motif_start)
    fraction = score / consensus_score if consensus_score else 0.0
    return MsdPayloadMotifAlignment(
        motif_model_id=motif.motif_model_id,
        motif_source_ref=motif.source_ref,
        motif_width_nt=motif.width,
        motif_span_0={"start": motif_start, "end": motif_end},
        payload_window_0={"start": payload_start, "end": payload_start + len(sequence)},
        strand=strand,
        sequence_5to3=sequence,
        consensus_sequence_5to3=consensus_sequence,
        score_bits=round(score, 6),
        consensus_score_bits=round(consensus_score, 6),
        consensus_score_fraction=round(fraction, 6),
    )


def score_sequence(sequence: str, *, motif: MotifModel, motif_start: int) -> float:
    if motif_start < 0 or motif_start + len(sequence) > motif.width:
        raise MsdRegionIngestError(
            f"Motif span {motif_start}:{motif_start + len(sequence)} exceeds width {motif.width}."
        )
    bases = "ACGT"
    score = 0.0
    for offset, base in enumerate(sequence):
        row = motif.matrix[motif_start + offset]
        probability = row[bases.index(base)]
        score += math.log2(max(probability, 1e-12) / 0.25)
    return score


__all__ = [
    "alignment_payload",
    "motif_alignments",
    "score_best_motif_window",
    "score_fixed_motif_span",
    "score_sequence",
]
