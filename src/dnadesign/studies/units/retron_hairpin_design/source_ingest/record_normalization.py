"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/record_normalization.py

Normalize one GenBank MSD-region record into compiler-facing primitives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from Bio.SeqRecord import SeqRecord

from .annotation_review import annotation_review_findings
from .feature_roles import feature_label, normalized_role_for_feature
from .genbank_utils import (
    qualifier_values,
    sha256_text,
    simple_span,
    variant_number,
)
from .models import NormalizedMsdFeature, NormalizedMsdRegionRecord
from .pairing_segments import pairing_segments_for_features
from .payload_binding import PayloadBindingCatalog, payload_binding_sites_for_segments

PRIMITIVE_ROLES = (
    "stem_base_left",
    "payload_primary",
    "snapback_foldback_geometry",
    "payload_complement",
    "stem_base_right",
)


def normalize_msd_region_record(
    record: SeqRecord,
    *,
    variant_id: str,
    payload_catalog: PayloadBindingCatalog | None,
) -> NormalizedMsdRegionRecord:
    source_sequence = str(record.seq).upper()
    display_sequence = str(record.seq.reverse_complement()).upper()
    features = tuple(
        _normalize_feature(feature, display_sequence=display_sequence, source_length=len(record.seq))
        for feature in record.features
    )
    features = tuple(feature for feature in features if feature.label or feature.role)
    features = _with_derived_stem_bases(features, display_sequence=display_sequence, source_length=len(record.seq))
    features = _deduplicate_equivalent_features(features)
    annotation_warnings, annotation_notes = annotation_review_findings(features, display_sequence=display_sequence)
    pairing_segments = pairing_segments_for_features(features=features, display_sequence=display_sequence)
    payload_binding_sites = payload_binding_sites_for_segments(pairing_segments, catalog=payload_catalog)
    direct_roles = any(feature.source_role for feature in features)
    inferred_roles = any(feature.role is not None and feature.source_role is None for feature in features)
    if direct_roles and inferred_roles:
        annotation_status = "mixed_typed_and_label_normalized"
    elif direct_roles:
        annotation_status = "typed_dnadesign_roles"
    else:
        annotation_status = "label_only_normalized"
    number = variant_number(variant_id)
    return NormalizedMsdRegionRecord(
        variant_id=variant_id,
        display_id=f"pES-retron-{number}",
        file_stem=f"pes-retron-{number}-msd-region",
        source_record_id=record.id,
        source_description=record.description,
        source_sequence_sha256=sha256_text(source_sequence),
        msd_sequence_sha256=sha256_text(display_sequence),
        sequence_length_nt=len(display_sequence),
        msd_sequence_5to3=display_sequence,
        rna_sequence_5to3=display_sequence.replace("T", "U"),
        annotation_status=annotation_status,
        annotation_warnings=annotation_warnings,
        annotation_notes=annotation_notes,
        pairing_segments=pairing_segments,
        payload_binding_sites=payload_binding_sites,
        features=features,
    )


def _with_derived_stem_bases(
    features: tuple[NormalizedMsdFeature, ...],
    *,
    display_sequence: str,
    source_length: int,
) -> tuple[NormalizedMsdFeature, ...]:
    roles = {feature.role for feature in features}
    derived: list[NormalizedMsdFeature] = []
    if "stem_base_left" not in roles:
        derived.extend(_derive_left_stem_base(features, display_sequence=display_sequence, source_length=source_length))
    if "stem_base_right" not in roles:
        derived.extend(
            _derive_right_stem_base(features, display_sequence=display_sequence, source_length=source_length)
        )
    return (*features, *derived)


def _derive_left_stem_base(
    features: Sequence[NormalizedMsdFeature],
    *,
    display_sequence: str,
    source_length: int,
) -> tuple[NormalizedMsdFeature, ...]:
    flank_5p = _single_feature_or_none(features, "flank_5p")
    if flank_5p is not None and flank_5p.display_end_0 - flank_5p.display_start_0 >= 4:
        return (
            _derived_feature(
                role="stem_base_left",
                label="Left Base",
                display_start_0=flank_5p.display_end_0 - 4,
                display_end_0=flank_5p.display_end_0,
                display_sequence=display_sequence,
                source_length=source_length,
            ),
        )
    annotated = _single_feature_or_none(features, "stem_base_left_annotated_span")
    if annotated is not None and annotated.display_end_0 - annotated.display_start_0 >= 4:
        return (
            _derived_feature(
                role="stem_base_left",
                label="Left Base",
                display_start_0=annotated.display_start_0,
                display_end_0=annotated.display_start_0 + 4,
                display_sequence=display_sequence,
                source_length=source_length,
            ),
        )
    return ()


def _derive_right_stem_base(
    features: Sequence[NormalizedMsdFeature],
    *,
    display_sequence: str,
    source_length: int,
) -> tuple[NormalizedMsdFeature, ...]:
    flank_3p = _single_feature_or_none(features, "flank_3p")
    if flank_3p is not None and flank_3p.display_end_0 - flank_3p.display_start_0 >= 4:
        return (
            _derived_feature(
                role="stem_base_right",
                label="Right Base",
                display_start_0=flank_3p.display_start_0,
                display_end_0=flank_3p.display_start_0 + 4,
                display_sequence=display_sequence,
                source_length=source_length,
            ),
        )
    annotated = _single_feature_or_none(features, "stem_base_right_annotated_span")
    if annotated is not None and annotated.display_end_0 - annotated.display_start_0 >= 4:
        return (
            _derived_feature(
                role="stem_base_right",
                label="Right Base",
                display_start_0=annotated.display_end_0 - 4,
                display_end_0=annotated.display_end_0,
                display_sequence=display_sequence,
                source_length=source_length,
            ),
        )
    return ()


def _deduplicate_equivalent_features(
    features: tuple[NormalizedMsdFeature, ...],
) -> tuple[NormalizedMsdFeature, ...]:
    by_key: dict[tuple[str | None, int, int, str], NormalizedMsdFeature] = {}
    ordered: list[tuple[str | None, int, int, str]] = []
    for feature in features:
        key = (feature.role, feature.display_start_0, feature.display_end_0, feature.sequence_5to3)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = feature
            ordered.append(key)
            continue
        if previous.source_role is None and feature.source_role is not None:
            by_key[key] = feature
    return tuple(by_key[key] for key in ordered)


def _single_feature_or_none(features: Sequence[NormalizedMsdFeature], role: str) -> NormalizedMsdFeature | None:
    matches = [feature for feature in features if feature.role == role]
    if len(matches) == 1:
        return matches[0]
    return None


def _derived_feature(
    *,
    role: str,
    label: str,
    display_start_0: int,
    display_end_0: int,
    display_sequence: str,
    source_length: int,
) -> NormalizedMsdFeature:
    source_start_0 = source_length - display_end_0
    source_end_0 = source_length - display_start_0
    return NormalizedMsdFeature(
        role=role,
        source_role=None,
        label=label,
        feature_type="derived_feature",
        source_start_0=source_start_0,
        source_end_0=source_end_0,
        source_strand=None,
        display_start_0=display_start_0,
        display_end_0=display_end_0,
        display_strand=1,
        sequence_5to3=display_sequence[display_start_0:display_end_0].upper(),
    )


def _normalize_feature(feature: Any, *, display_sequence: str, source_length: int) -> NormalizedMsdFeature:
    start, end = simple_span(feature)
    display_start = source_length - end
    display_end = source_length - start
    labels = qualifier_values(feature, "label") + qualifier_values(feature, "note")
    source_roles = qualifier_values(feature, "dnadesign_role")
    role, source_role = normalized_role_for_feature(
        labels=labels,
        source_roles=source_roles,
        source_start_0=start,
        source_end_0=end,
        source_length=source_length,
        source_strand=feature.location.strand,
    )
    source_strand = feature.location.strand
    display_strand = -source_strand if source_strand in {-1, 1} else source_strand
    return NormalizedMsdFeature(
        role=role,
        source_role=source_role,
        label=feature_label(labels),
        feature_type=str(feature.type),
        source_start_0=start,
        source_end_0=end,
        source_strand=source_strand,
        display_start_0=display_start,
        display_end_0=display_end,
        display_strand=display_strand,
        sequence_5to3=display_sequence[display_start:display_end].upper(),
    )


__all__ = ["PRIMITIVE_ROLES", "normalize_msd_region_record"]
