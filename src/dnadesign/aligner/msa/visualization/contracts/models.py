"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/contracts/models.py

Data models for generic MSA visualization sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PROFILE_ID_STEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class MsaVisualizationRequest:
    """Inputs for generic MSA QC and visualization sidecars."""

    alignment_root: Path
    output_root: Path
    profile_ids: tuple[str, ...]
    target_row_id: str
    target_sequence_hash: str | None = None
    annotation_tracks_yaml: Path | None = None
    exemplar_rows_yaml: Path | None = None
    panel_spec_yaml: Path | None = None
    allow_missing_profiles: bool = False
    created_at: str = "unknown"

    def __post_init__(self) -> None:
        if not self.profile_ids:
            raise ValueError("profile_ids must be non-empty")
        if any(not profile_id.strip() for profile_id in self.profile_ids):
            raise ValueError("profile_ids must contain only non-empty values")
        for profile_id in self.profile_ids:
            _validate_profile_id_stem(profile_id)
        if len(set(self.profile_ids)) != len(self.profile_ids):
            raise ValueError("profile_ids must be unique")
        if not self.target_row_id.strip():
            raise ValueError("target_row_id must be non-empty")


@dataclass(frozen=True)
class MsaVisualizationResult:
    """Materialized visualization sidecar paths."""

    profile_ids: tuple[str, ...]
    missing_profile_ids: tuple[str, ...]
    index_manifest_path: Path
    position_qc_csv_path: Path
    html_report_path: Path
    profile_qc_paths: dict[str, Path]
    profile_svg_paths: dict[str, Path]
    profile_exemplar_svg_paths: dict[str, Path]
    profile_alignment_overview_svg_paths: dict[str, Path]
    profile_consensus_histogram_svg_paths: dict[str, Path]


@dataclass(frozen=True)
class ProfileQc:
    """Per-profile MSA visualization metadata."""

    profile_id: str
    aligned_fasta_path: Path
    record_count: int
    alignment_length: int
    canonical_position_count: int
    inserted_column_count: int
    target_ungapped_sha256: str
    profile_qc_path: Path
    profile_svg_path: Path
    profile_exemplar_svg_path: Path
    profile_alignment_overview_svg_path: Path
    profile_consensus_histogram_svg_path: Path


@dataclass(frozen=True)
class AnnotationFeature:
    """Target-position feature annotation."""

    id: str
    label: str
    start: int
    end: int
    color: str
    fill_opacity: float
    stroke_color: str
    stroke_width: float
    text_color: str | None
    label_position: str


@dataclass(frozen=True)
class AnnotationTrack:
    """Grouped target-position feature annotations."""

    id: str
    label: str
    color: str
    features: tuple[AnnotationFeature, ...]


@dataclass(frozen=True)
class FeatureWindow:
    """Display window around one target-position annotation feature."""

    feature: AnnotationFeature
    start: int
    end: int


@dataclass(frozen=True)
class ExemplarRow:
    """Explicit row selected for display-only alignment views."""

    record_id: str
    label: str
    group: str


@dataclass(frozen=True)
class ExemplarRowsSpec:
    """Profile-aware exemplar-row selections."""

    default_rows: tuple[ExemplarRow, ...]
    profile_rows: dict[str, tuple[ExemplarRow, ...]]

    def rows_for_profile(self, profile_id: str) -> tuple[ExemplarRow, ...]:
        return self.profile_rows.get(profile_id, self.default_rows)

    @property
    def has_rows(self) -> bool:
        return bool(self.default_rows or self.profile_rows)


@dataclass(frozen=True)
class PositionQc:
    """Target-position MSA QC values."""

    profile_id: str
    canonical_position: int
    alignment_column: int
    target_aa: str
    non_gap_count: int
    gap_count: int
    gap_fraction: float
    plurality_aa: str
    plurality_count: int
    plurality_frequency: float


def _validate_profile_id_stem(profile_id: str) -> None:
    """Reject profile ids that cannot be safely interpolated into filenames."""

    if (
        profile_id != profile_id.strip()
        or profile_id in {".", ".."}
        or ".." in profile_id
        or "/" in profile_id
        or "\\" in profile_id
        or not _PROFILE_ID_STEM_RE.fullmatch(profile_id)
    ):
        raise ValueError(
            "profile_ids must be file-safe stems containing only letters, digits, dots, hyphens, or underscores"
        )
