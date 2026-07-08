"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_sites.py

Derive payload binding-site semantics from MSD pairing segments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import (
    MsdPayloadBindingSite,
    MsdPayloadMotifAlignment,
    MsdPayloadReferenceComparison,
    MsdRegionPairingSegment,
)
from .payload_binding_models import PayloadBindingCatalog, PayloadMember
from .payload_binding_utils import identity_count, normalize_dna
from .payload_motifs import motif_alignments


def payload_binding_sites_for_segments(
    segments: Sequence[MsdRegionPairingSegment],
    *,
    catalog: PayloadBindingCatalog | None,
) -> tuple[MsdPayloadBindingSite, ...]:
    """Derive payload-binding-site semantics from pairing segments."""

    sites: list[MsdPayloadBindingSite] = []
    for segment in segments:
        if segment.segment != "payload_stem":
            continue
        sites.append(_payload_binding_site(segment, catalog=catalog))
    return tuple(sites)


def _payload_binding_site(
    segment: MsdRegionPairingSegment,
    *,
    catalog: PayloadBindingCatalog | None,
) -> MsdPayloadBindingSite:
    primary = normalize_dna(segment.left_sequence_5to3)
    complement = normalize_dna(segment.right_sequence_5to3)
    member = catalog.member_for_primary_sequence(primary) if catalog is not None else None
    alignments = motif_alignments(primary, member=member, catalog=catalog)
    return MsdPayloadBindingSite(
        segment=segment.segment,
        primary_sequence_5to3=primary,
        complement_sequence_5to3=complement,
        payload_length_nt=len(primary),
        payload_family_id=member.family_id if member is not None else None,
        parent_payload_id=member.parent_payload_id if member is not None else None,
        payload_member_id=member.member_id if member is not None else None,
        payload_class=_payload_class(member=member, motif_alignments=alignments),
        retained_parent_span_0=member.retained_parent_span_0 if member is not None else None,
        motif_alignments=alignments,
        reference_comparisons=_reference_comparisons(
            primary,
            member=member,
            catalog=catalog,
            motif_alignments=alignments,
        ),
    )


def _reference_comparisons(
    primary: str,
    *,
    member: PayloadMember | None,
    catalog: PayloadBindingCatalog | None,
    motif_alignments: Sequence[MsdPayloadMotifAlignment],
) -> tuple[MsdPayloadReferenceComparison, ...]:
    if catalog is None:
        return ()
    rows: list[MsdPayloadReferenceComparison] = []
    query_alignment = motif_alignments[0] if motif_alignments else None
    for reference in catalog.reference_members():
        query_sequence, query_span, reference_start, reference_sequence = _reference_sequence_for_query(
            primary,
            query_member=member,
            query_alignment=query_alignment,
            reference=reference,
            catalog=catalog,
        )
        rows.append(_reference_comparison(reference, query_sequence, query_span, reference_start, reference_sequence))
    return tuple(rows)


def _reference_comparison(
    reference: PayloadMember,
    query_sequence: str,
    query_span: dict[str, int],
    reference_start: int,
    reference_sequence: str,
) -> MsdPayloadReferenceComparison:
    compared = min(len(query_sequence), len(reference_sequence))
    mismatch_count = sum(
        query_base != reference_base
        for query_base, reference_base in zip(query_sequence[:compared], reference_sequence[:compared])
    ) + abs(len(query_sequence) - len(reference_sequence))
    compared_nt = max(len(query_sequence), len(reference_sequence))
    identity = 1.0 - (mismatch_count / compared_nt) if compared_nt else 0.0
    return MsdPayloadReferenceComparison(
        reference_payload_id=reference.member_id,
        reference_payload_family_id=reference.family_id,
        reference_span_0={"start": reference_start, "end": reference_start + len(reference_sequence)},
        query_span_0=query_span,
        query_sequence_5to3=query_sequence,
        reference_sequence_5to3=reference_sequence,
        compared_nt=compared_nt,
        mismatch_count=mismatch_count,
        identity_fraction=round(identity, 6),
        comparison_class=_comparison_class(mismatch_count=mismatch_count, compared_nt=compared_nt),
    )


def _reference_sequence_for_query(
    primary: str,
    *,
    query_member: PayloadMember | None,
    query_alignment: MsdPayloadMotifAlignment | None,
    reference: PayloadMember,
    catalog: PayloadBindingCatalog,
) -> tuple[str, dict[str, int], int, str]:
    if query_alignment is not None and reference.motif_model_id == query_alignment.motif_model_id:
        resolved = _motif_aligned_reference(
            primary,
            query_alignment=query_alignment,
            reference=reference,
            catalog=catalog,
        )
        if resolved is not None:
            return resolved
    reference_parent = reference.parent_primary_sequence_5to3
    if query_member is not None:
        span = query_member.retained_parent_span_0
        if 0 <= span["start"] < span["end"] <= len(reference_parent) and span["end"] - span["start"] == len(primary):
            return primary, dict(span), span["start"], reference_parent[span["start"] : span["end"]]
    if len(primary) == len(reference_parent):
        span = {"start": 0, "end": len(primary)}
        return primary, span, 0, reference_parent
    if len(primary) < len(reference_parent):
        best_start = max(
            range(0, len(reference_parent) - len(primary) + 1),
            key=lambda start: identity_count(primary, reference_parent[start : start + len(primary)]),
        )
        span = {"start": 0, "end": len(primary)}
        return primary, span, best_start, reference_parent[best_start : best_start + len(primary)]
    span = {"start": 0, "end": len(primary)}
    return primary, span, 0, reference_parent


def _motif_aligned_reference(
    primary: str,
    *,
    query_alignment: MsdPayloadMotifAlignment,
    reference: PayloadMember,
    catalog: PayloadBindingCatalog,
) -> tuple[str, dict[str, int], int, str] | None:
    reference_alignments = motif_alignments(reference.primary_sequence_5to3, member=reference, catalog=catalog)
    if not reference_alignments:
        return None
    reference_alignment = reference_alignments[0]
    query_span = query_alignment.motif_span_0
    reference_span = reference_alignment.motif_span_0
    offset = query_span["start"] - reference_span["start"]
    if not (0 <= offset and offset + len(query_alignment.sequence_5to3) <= len(reference_alignment.sequence_5to3)):
        return None
    reference_sequence = reference_alignment.sequence_5to3[offset : offset + len(query_alignment.sequence_5to3)]
    return query_alignment.sequence_5to3, dict(query_span), query_span["start"], reference_sequence


def _payload_class(
    *,
    member: PayloadMember | None,
    motif_alignments: Sequence[MsdPayloadMotifAlignment],
) -> str:
    if member is not None and member.is_parent:
        return "catalog_parent_payload"
    if member is not None:
        return "catalog_trim_payload"
    if motif_alignments:
        alignment = motif_alignments[0]
        if alignment.consensus_score_fraction >= 0.65:
            return "motif_congruent_uncataloged_payload"
    return "uncataloged_payload"


def _comparison_class(*, mismatch_count: int, compared_nt: int) -> str:
    if compared_nt == 0:
        return "unscored"
    if mismatch_count == 0:
        return "identical"
    if mismatch_count == 1:
        return "single_edit"
    identity = 1.0 - (mismatch_count / compared_nt)
    if identity >= 0.65:
        return "moderate_difference"
    return "large_difference"


__all__ = ["payload_binding_sites_for_segments"]
