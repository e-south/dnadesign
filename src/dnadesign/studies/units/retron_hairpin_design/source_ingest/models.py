"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/models.py

Typed records for MSD region source ingest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class MsdRegionIngestError(ValueError):
    """Raised when MSD region source records cannot be normalized safely."""


@dataclass(frozen=True, slots=True)
class SkippedMsdSourceRecord:
    record_id: str
    reason: str
    sequence_length_nt: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NormalizedMsdFeature:
    role: str | None
    source_role: str | None
    label: str
    feature_type: str
    source_start_0: int
    source_end_0: int
    source_strand: int | None
    display_start_0: int
    display_end_0: int
    display_strand: int | None
    sequence_5to3: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MsdRegionAnnotationWarning:
    kind: str
    role: str
    label: str
    source_span_0: tuple[int, int]
    display_span_0: tuple[int, int]
    annotated_sequence_5to3: str
    compiler_sequence_5to3: str
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MsdRegionAnnotationNote:
    kind: str
    role: str
    label: str
    source_span_0: tuple[int, int]
    display_span_0: tuple[int, int]
    annotated_sequence_5to3: str
    compiler_sequence_5to3: str
    severity: str
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MsdRegionPairingSegment:
    segment: str
    left_role: str
    right_role: str
    left_sequence_5to3: str
    right_sequence_5to3: str
    length_bp: int
    watson_crick_bp: int
    wobble_bp: int
    mismatch_bp: int
    unpaired_nt: int
    pairing_status: str
    intent: str
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MsdPayloadMotifAlignment:
    motif_model_id: str
    motif_source_ref: str
    motif_width_nt: int
    motif_span_0: dict[str, int]
    payload_window_0: dict[str, int]
    strand: str
    sequence_5to3: str
    consensus_sequence_5to3: str
    score_bits: float
    consensus_score_bits: float
    consensus_score_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MsdPayloadReferenceComparison:
    reference_payload_id: str
    reference_payload_family_id: str
    reference_span_0: dict[str, int]
    query_span_0: dict[str, int]
    query_sequence_5to3: str
    reference_sequence_5to3: str
    compared_nt: int
    mismatch_count: int
    identity_fraction: float
    comparison_class: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MsdPayloadBindingSite:
    segment: str
    primary_sequence_5to3: str
    complement_sequence_5to3: str
    payload_length_nt: int
    payload_family_id: str | None
    parent_payload_id: str | None
    payload_member_id: str | None
    payload_class: str
    retained_parent_span_0: dict[str, int] | None
    motif_alignments: tuple[MsdPayloadMotifAlignment, ...]
    reference_comparisons: tuple[MsdPayloadReferenceComparison, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NormalizedMsdRegionRecord:
    variant_id: str
    display_id: str
    file_stem: str
    source_record_id: str
    source_description: str
    source_sequence_sha256: str
    msd_sequence_sha256: str
    sequence_length_nt: int
    msd_sequence_5to3: str
    rna_sequence_5to3: str
    annotation_status: str
    annotation_warnings: tuple[MsdRegionAnnotationWarning, ...]
    annotation_notes: tuple[MsdRegionAnnotationNote, ...]
    pairing_segments: tuple[MsdRegionPairingSegment, ...]
    payload_binding_sites: tuple[MsdPayloadBindingSite, ...]
    features: tuple[NormalizedMsdFeature, ...]

    def primitive(self, role: str) -> NormalizedMsdFeature:
        matches = [feature for feature in self.features if feature.role == role]
        if len(matches) != 1:
            raise MsdRegionIngestError(
                f"{self.variant_id}: expected exactly one normalized feature with role {role!r}, found {len(matches)}."
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["contract"] = "retron_msd_region_record_v1"
        payload["schema_version"] = 1
        return payload


@dataclass(frozen=True, slots=True)
class MsdRegionSourceBundle:
    source_path: str
    source_sha256: str
    source_record_count: int
    records: tuple[NormalizedMsdRegionRecord, ...]
    skipped_records: tuple[SkippedMsdSourceRecord, ...]
    replacement_sources: tuple[dict[str, object], ...] = ()
    source_kind: str = "bulk_migration_genbank"
    source_inputs: tuple[dict[str, object], ...] = ()
    retired_sources: tuple[dict[str, object], ...] = ()

    @property
    def included_record_count(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class MsdRegionDiscrepancy:
    kind: str
    variant_id: str
    compared_path: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MsdRegionComparisonReport:
    comparison_count: int
    discrepancies: tuple[MsdRegionDiscrepancy, ...]

    @property
    def discrepancy_count(self) -> int:
        return len(self.discrepancies)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": "retron_msd_region_discrepancy_report_v1",
            "schema_version": 1,
            "comparison_count": self.comparison_count,
            "discrepancy_count": self.discrepancy_count,
            "discrepancies": [item.to_dict() for item in self.discrepancies],
        }


@dataclass(frozen=True, slots=True)
class MsdRegionBundleWriteResult:
    output_dir: str
    manifest_path: str
    compiler_spec_path: str
    discrepancy_report_path: str | None
    variant_record_paths: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def path_text(path: str | Path) -> str:
    return Path(path).as_posix()


__all__ = [
    "MsdRegionBundleWriteResult",
    "MsdRegionAnnotationNote",
    "MsdRegionAnnotationWarning",
    "MsdRegionComparisonReport",
    "MsdRegionDiscrepancy",
    "MsdRegionIngestError",
    "MsdRegionPairingSegment",
    "MsdPayloadBindingSite",
    "MsdPayloadMotifAlignment",
    "MsdPayloadReferenceComparison",
    "MsdRegionSourceBundle",
    "NormalizedMsdFeature",
    "NormalizedMsdRegionRecord",
    "SkippedMsdSourceRecord",
    "path_text",
]
