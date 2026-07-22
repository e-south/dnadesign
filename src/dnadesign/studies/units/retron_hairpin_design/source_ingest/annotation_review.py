"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/annotation_review.py

Annotation review notes for normalized MSD-region features.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

from .genbank_utils import reverse_complement
from .models import (
    MsdRegionAnnotationNote,
    MsdRegionAnnotationWarning,
    NormalizedMsdFeature,
)


def annotation_review_findings(
    features: Sequence[NormalizedMsdFeature],
    *,
    display_sequence: str,
) -> tuple[tuple[MsdRegionAnnotationWarning, ...], tuple[MsdRegionAnnotationNote, ...]]:
    warnings: list[MsdRegionAnnotationWarning] = []
    notes: list[MsdRegionAnnotationNote] = []
    by_role = {feature.role: feature for feature in features if feature.role is not None}
    for annotated_role, primitive_role in (
        ("stem_base_left_annotated_span", "stem_base_left"),
        ("stem_base_right_annotated_span", "stem_base_right"),
    ):
        annotated = by_role.get(annotated_role)
        primitive = by_role.get(primitive_role)
        if annotated is not None and primitive is not None:
            notes.append(_stem_boundary_note(primitive_role=primitive_role, annotated=annotated, primitive=primitive))
    _add_foldback_boundary_note(notes, by_role=by_role, display_sequence=display_sequence)
    _add_payload_boundary_notes(notes, by_role=by_role, display_sequence=display_sequence)
    return tuple(warnings), tuple(notes)


def _stem_boundary_note(
    *,
    primitive_role: str,
    annotated: NormalizedMsdFeature,
    primitive: NormalizedMsdFeature,
) -> MsdRegionAnnotationNote:
    return MsdRegionAnnotationNote(
        kind="stem_base_boundary_derived_from_extended_annotation",
        role=primitive_role,
        label=annotated.label,
        source_span_0=(annotated.source_start_0, annotated.source_end_0),
        display_span_0=(annotated.display_start_0, annotated.display_end_0),
        annotated_sequence_5to3=annotated.sequence_5to3,
        compiler_sequence_5to3=primitive.sequence_5to3,
        severity="info",
        note=(
            "Source annotation spans a larger stem context; the compiler-facing stem base remains the "
            "4 nt boundary sequence and the extra sequence is represented by stem-extension pairing facts."
        ),
    )


def _add_foldback_boundary_note(
    notes: list[MsdRegionAnnotationNote],
    *,
    by_role: dict[str | None, NormalizedMsdFeature],
    display_sequence: str,
) -> None:
    foldback = by_role.get("snapback_foldback_geometry")
    payload = by_role.get("payload_primary")
    complement = by_role.get("payload_complement")
    if foldback is None or payload is None or complement is None or payload.display_end_0 > complement.display_start_0:
        return
    compiler_cap = display_sequence[payload.display_end_0 : complement.display_start_0].upper()
    if compiler_cap and compiler_cap != foldback.sequence_5to3:
        notes.append(
            _boundary_note(
                kind="foldback_feature_boundary_granularity",
                role="snapback_foldback_geometry",
                feature=foldback,
                compiler_sequence=compiler_cap,
                text=(
                    "Source foldback annotation is a narrower subfeature; compiler-facing snapback geometry "
                    "uses the full interval between payload_primary and payload_complement."
                ),
            )
        )


def _add_payload_boundary_notes(
    notes: list[MsdRegionAnnotationNote],
    *,
    by_role: dict[str | None, NormalizedMsdFeature],
    display_sequence: str,
) -> None:
    left = by_role.get("stem_base_left")
    right = by_role.get("stem_base_right")
    payload = by_role.get("payload_primary")
    complement = by_role.get("payload_complement")
    if left is None or right is None or payload is None or complement is None:
        return
    compiler_payload = display_sequence[left.display_end_0 : payload.display_end_0].upper()
    compiler_complement = display_sequence[complement.display_start_0 : right.display_start_0].upper()
    if compiler_payload and compiler_payload != payload.sequence_5to3:
        notes.append(
            _boundary_note(
                kind="payload_primary_boundary_extends_to_stem_context",
                role="payload_primary",
                feature=payload,
                compiler_sequence=compiler_payload,
                text=(
                    "Source payload annotation marks the payload proper; compiler-facing primary arm uses the "
                    "full paired interval from the left stem base to the cap."
                ),
            )
        )
    if compiler_complement and compiler_complement != complement.sequence_5to3:
        notes.append(
            _boundary_note(
                kind="payload_complement_boundary_extends_to_stem_context",
                role="payload_complement",
                feature=complement,
                compiler_sequence=compiler_complement,
                text=(
                    "Source payload-complement annotation marks the payload proper; compiler-facing complement "
                    "arm uses the full paired interval from the cap to the right stem base."
                ),
            )
        )
    if compiler_payload and compiler_complement and compiler_complement != reverse_complement(compiler_payload):
        notes.append(
            _boundary_note(
                kind="payload_complement_sequence_is_source_explicit",
                role="payload_complement",
                feature=complement,
                compiler_sequence=compiler_complement,
                text=(
                    "Source complement arm is not the reverse complement of the source primary arm; this is "
                    "represented as an explicit source arm and classified through pairing_segments."
                ),
            )
        )


def _boundary_note(
    *,
    kind: str,
    role: str,
    feature: NormalizedMsdFeature,
    compiler_sequence: str,
    text: str,
) -> MsdRegionAnnotationNote:
    return MsdRegionAnnotationNote(
        kind=kind,
        role=role,
        label=feature.label,
        source_span_0=(feature.source_start_0, feature.source_end_0),
        display_span_0=(feature.display_start_0, feature.display_end_0),
        annotated_sequence_5to3=feature.sequence_5to3,
        compiler_sequence_5to3=compiler_sequence,
        severity="info",
        note=text,
    )


__all__ = ["annotation_review_findings"]
