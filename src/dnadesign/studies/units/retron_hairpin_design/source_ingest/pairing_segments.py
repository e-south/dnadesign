"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/pairing_segments.py

Derived pairing-segment facts for decomposed MSD-region records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .genbank_utils import reverse_complement
from .models import MsdRegionPairingSegment, NormalizedMsdFeature


def pairing_segments_for_features(
    *,
    features: Sequence[NormalizedMsdFeature],
    display_sequence: str,
) -> tuple[MsdRegionPairingSegment, ...]:
    by_role = {feature.role: feature for feature in features if feature.role is not None}
    left = by_role.get("stem_base_left")
    right = by_role.get("stem_base_right")
    payload = by_role.get("payload_primary")
    complement = by_role.get("payload_complement")
    if left is None or right is None or payload is None or complement is None:
        return ()
    segments: list[MsdRegionPairingSegment] = []
    stem_extension_left = display_sequence[left.display_end_0 : payload.display_start_0].upper()
    stem_extension_right = display_sequence[complement.display_end_0 : right.display_start_0].upper()
    if stem_extension_left or stem_extension_right:
        segments.append(
            _pairing_segment(
                segment="stem_extension",
                left_role="stem_extension_left",
                right_role="stem_extension_right",
                left_sequence=stem_extension_left,
                right_sequence=stem_extension_right,
                inferred_intent=_intent_for_segment(
                    segment="stem_extension",
                    features=features,
                    left_sequence=stem_extension_left,
                    right_sequence=stem_extension_right,
                ),
                note="Non-payload paired stem context between the 4 nt stem bases and payload annotation.",
            )
        )
    segments.append(
        _pairing_segment(
            segment="payload_stem",
            left_role="payload_primary",
            right_role="payload_complement",
            left_sequence=payload.sequence_5to3,
            right_sequence=complement.sequence_5to3,
            inferred_intent=_intent_for_segment(
                segment="payload_stem",
                features=features,
                left_sequence=payload.sequence_5to3,
                right_sequence=complement.sequence_5to3,
            ),
            note="Payload-bearing stem segment, usually tetO or a tetO-derived payload.",
        )
    )
    cap = by_role.get("snapback_cap")
    if (
        cap is not None
        and payload.display_end_0 <= cap.display_start_0 <= cap.display_end_0 <= complement.display_start_0
    ):
        foldback_stem = display_sequence[payload.display_end_0 : cap.display_start_0].upper()
        foldback_return = display_sequence[cap.display_end_0 : complement.display_start_0].upper()
    else:
        foldback_stem = _feature_sequence_from_features(by_role, "snapback_retained_stem")
        foldback_return = _feature_sequence_from_features(by_role, "snapback_foldback_return")
    if foldback_stem or foldback_return:
        segments.append(
            _pairing_segment(
                segment="foldback_stem",
                left_role="snapback_retained_stem",
                right_role="snapback_foldback_return",
                left_sequence=foldback_stem,
                right_sequence=foldback_return,
                inferred_intent=_intent_for_segment(
                    segment="foldback_stem",
                    features=features,
                    left_sequence=foldback_stem,
                    right_sequence=foldback_return,
                ),
                note="Foldback stem/return segment between payload stem and snapback cap.",
            )
        )
    return tuple(segments)


def _pairing_segment(
    *,
    segment: str,
    left_role: str,
    right_role: str,
    left_sequence: str,
    right_sequence: str,
    inferred_intent: str,
    note: str,
) -> MsdRegionPairingSegment:
    left = left_sequence.upper()
    right = right_sequence.upper()
    compared = min(len(left), len(right))
    watson_crick = 0
    wobble = 0
    mismatch = 0
    for left_base, right_base in zip(left[:compared], reversed(right[-compared:] if compared else "")):
        pair = (left_base, right_base)
        if pair in {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}:
            watson_crick += 1
        elif pair in {("G", "T"), ("T", "G")}:
            wobble += 1
        else:
            mismatch += 1
    unpaired = abs(len(left) - len(right))
    if not left or not right:
        status = "unknown"
    elif unpaired:
        status = "review_required"
    elif mismatch == 0 and wobble == 0:
        status = "canonical_wc"
    elif inferred_intent in {"declared_design", "inferred_from_source"}:
        status = "intentional_mismatch"
    else:
        status = "review_required"
    return MsdRegionPairingSegment(
        segment=segment,
        left_role=left_role,
        right_role=right_role,
        left_sequence_5to3=left,
        right_sequence_5to3=right,
        length_bp=compared,
        watson_crick_bp=watson_crick,
        wobble_bp=wobble,
        mismatch_bp=mismatch,
        unpaired_nt=unpaired,
        pairing_status=status,
        intent=inferred_intent,
        note=note,
    )


def _intent_for_segment(
    *,
    segment: str,
    features: Sequence[NormalizedMsdFeature],
    left_sequence: str,
    right_sequence: str,
) -> str:
    if not left_sequence or not right_sequence:
        return "unresolved"
    if right_sequence.upper() == reverse_complement(left_sequence.upper()):
        return "declared_design"
    labels = " ".join(feature.label.lower() for feature in features)
    if "mismatch" in labels or "wobble" in labels:
        return "declared_design"
    if segment == "payload_stem" and any(feature.role == "payload_complement" for feature in features):
        return "inferred_from_source"
    return "unresolved"


def _feature_sequence_from_features(by_role: Mapping[str | None, NormalizedMsdFeature], role: str) -> str:
    feature = by_role.get(role)
    return feature.sequence_5to3.upper() if feature is not None else ""


__all__ = ["pairing_segments_for_features"]
