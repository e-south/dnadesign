"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_explicit_evaluation.py

Explicit released-product Snapback evaluation and report shaping.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.nickases.models import NickaseCatalog, NickaseCatalogEntry
from dnadesign.cruncher.nickases.scanning import enumerate_site_instances as enumerate_nickase_site_instances
from dnadesign.cruncher.nickases.selection import matching_nickase_warning_codes
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog
from dnadesign.cruncher.snapback.models import SnapbackIssue
from dnadesign.cruncher.snapback.released_models import (
    ReleasedSnapbackEvaluationReport,
    ReleasedSnapbackReportMetadata,
    SingleNickReleasedSnapbackSpec,
    build_release_catalog_info,
    build_released_nickase_catalog_info,
)
from dnadesign.cruncher.snapback.released_projection import evaluate_released_precursor


class AmbiguousPrecursorOriginError(ValueError):
    def __init__(self, *, offsets: list[int], target_boundary: int, variant_id: str) -> None:
        self.offsets = offsets
        self.target_boundary = target_boundary
        self.variant_id = variant_id
        super().__init__(
            "The released-product precursor maps to multiple possible origin offsets for the requested nick boundary."
        )


def infer_precursor_coordinate_offset(
    spec: SingleNickReleasedSnapbackSpec,
    *,
    nick_entry: NickaseCatalogEntry,
) -> int:
    matches = enumerate_nickase_site_instances(
        spec.input.precursor_top_strand,
        coordinate_offset=0,
        entry=nick_entry,
    )
    matches = [match for match in matches if match.nick.strand == "primary"]
    if spec.nick_stage.intended_site_sequence is not None:
        matches = [
            match for match in matches if match.site.matched_span_sequence == spec.nick_stage.intended_site_sequence
        ]
    offsets = {
        match.nick.boundary_context - spec.final_target.nick_boundary_from_left
        for match in matches
        if match.nick.boundary_context >= spec.final_target.nick_boundary_from_left
    }
    if not offsets:
        return 0
    ordered_offsets = sorted(offsets)
    if len(ordered_offsets) > 1:
        raise AmbiguousPrecursorOriginError(
            offsets=ordered_offsets,
            target_boundary=spec.final_target.nick_boundary_from_left,
            variant_id=nick_entry.id,
        )
    return ordered_offsets[0]


def build_invalid_catalog_report(
    spec: SingleNickReleasedSnapbackSpec,
    *,
    spec_path: Path,
    workspace_root: Path,
    nick_catalog_source: str,
    release_catalog_source: str,
    disallowed_nickase_warning_codes: list[str],
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> ReleasedSnapbackEvaluationReport:
    return ReleasedSnapbackEvaluationReport(
        status="invalid_catalog",
        spec_name=spec.name,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        metadata=ReleasedSnapbackReportMetadata(
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=list(disallowed_nickase_warning_codes),
            final_target=spec.final_target,
        ),
        issues=[SnapbackIssue(code=code, message=message, details=details or {})],
    )


def build_invalid_precursor_report(
    spec: SingleNickReleasedSnapbackSpec,
    *,
    spec_path: Path,
    workspace_root: Path,
    nick_catalog_source: str,
    release_catalog_source: str,
    disallowed_nickase_warning_codes: list[str],
    code: str,
    message: str,
    details: dict[str, object] | None = None,
    nick_entry: NickaseCatalogEntry | None = None,
    release_entry=None,
) -> ReleasedSnapbackEvaluationReport:
    return ReleasedSnapbackEvaluationReport(
        status="invalid_precursor",
        spec_name=spec.name,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        metadata=ReleasedSnapbackReportMetadata(
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=list(disallowed_nickase_warning_codes),
            final_target=spec.final_target,
            nickase_catalog_variants=(
                [build_released_nickase_catalog_info(nick_entry)] if nick_entry is not None else []
            ),
            release_catalog_variants=([build_release_catalog_info(release_entry)] if release_entry is not None else []),
        ),
        issues=[SnapbackIssue(code=code, message=message, details=details or {})],
    )


def build_released_explicit_report(
    spec: SingleNickReleasedSnapbackSpec,
    *,
    spec_path: Path,
    workspace_root: Path,
    nick_catalog: NickaseCatalog,
    release_catalog: ReleaseEnzymeCatalog,
    nick_catalog_source: str,
    release_catalog_source: str,
) -> ReleasedSnapbackEvaluationReport:
    nick_catalog_by_id = nick_catalog.by_id()
    if spec.nick_stage.nickase_variant_id not in nick_catalog_by_id:
        return build_invalid_catalog_report(
            spec,
            spec_path=spec_path,
            workspace_root=workspace_root,
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=spec.constraints.disallowed_nickase_warning_codes,
            code="UNKNOWN_NICKASE_VARIANT_ID",
            message="nick_stage.nickase_variant_id was not found in the resolved nickase catalog.",
            details={"variant_id": spec.nick_stage.nickase_variant_id},
        )
    release_catalog_by_id = release_catalog.by_id()
    if spec.release_stage.release_variant_id not in release_catalog_by_id:
        return build_invalid_catalog_report(
            spec,
            spec_path=spec_path,
            workspace_root=workspace_root,
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=spec.constraints.disallowed_nickase_warning_codes,
            code="UNKNOWN_RELEASE_VARIANT_ID",
            message="release_stage.release_variant_id was not found in the resolved release-enzyme catalog.",
            details={"variant_id": spec.release_stage.release_variant_id},
        )
    nick_entry = nick_catalog_by_id[spec.nick_stage.nickase_variant_id]
    release_entry = release_catalog_by_id[spec.release_stage.release_variant_id]
    disallowed_warning_codes = matching_nickase_warning_codes(
        nick_entry,
        warning_codes=spec.constraints.disallowed_nickase_warning_codes,
    )
    if disallowed_warning_codes:
        return build_invalid_catalog_report(
            spec,
            spec_path=spec_path,
            workspace_root=workspace_root,
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=spec.constraints.disallowed_nickase_warning_codes,
            code="DISALLOWED_NICKASE_WARNING_CODE",
            message=(
                "nick_stage.nickase_variant_id is disallowed by released-product operational policy "
                "because it carries a blocked warning code."
            ),
            details={
                "variant_id": spec.nick_stage.nickase_variant_id,
                "matching_warning_codes": disallowed_warning_codes,
            },
        )
    try:
        precursor_coordinate_offset = infer_precursor_coordinate_offset(spec, nick_entry=nick_entry)
    except AmbiguousPrecursorOriginError as exc:
        return build_invalid_precursor_report(
            spec,
            spec_path=spec_path,
            workspace_root=workspace_root,
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=spec.constraints.disallowed_nickase_warning_codes,
            code="PRECURSOR_ORIGIN_AMBIGUOUS",
            message=(
                "The precursor contains multiple pre-nick anchors for the requested released-product origin. "
                "Choose one anchor explicitly before validating or materializing released-product artifacts."
            ),
            details={
                "variant_id": exc.variant_id,
                "target_boundary": exc.target_boundary,
                "candidate_coordinate_offsets": exc.offsets,
            },
            nick_entry=nick_entry,
            release_entry=release_entry,
        )
    evaluation = evaluate_released_precursor(
        precursor_top_strand=spec.input.precursor_top_strand,
        nick_entry=nick_entry,
        release_entry=release_entry,
        target=spec.final_target,
        constraints=spec.constraints,
        nick_intended_site_sequence=spec.nick_stage.intended_site_sequence,
        release_intended_site_sequence=spec.release_stage.intended_site_sequence,
        precursor_coordinate_offset=precursor_coordinate_offset,
    )
    return ReleasedSnapbackEvaluationReport(
        status=evaluation.status,  # type: ignore[arg-type]
        spec_name=spec.name,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        metadata=ReleasedSnapbackReportMetadata(
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=list(spec.constraints.disallowed_nickase_warning_codes),
            final_target=spec.final_target,
            nickase_catalog_variants=[build_released_nickase_catalog_info(nick_entry)],
            release_catalog_variants=[build_release_catalog_info(release_entry)],
        ),
        issues=evaluation.issues,
        pre_nick_site=evaluation.pre_nick_match.site if evaluation.pre_nick_match is not None else None,
        pre_nick_event=evaluation.pre_nick_match.nick if evaluation.pre_nick_match is not None else None,
        release_site=evaluation.release_match.site if evaluation.release_match is not None else None,
        release_event=evaluation.release_match.cut if evaluation.release_match is not None else None,
        projection=evaluation.projection,
        candidate=evaluation.candidate,
    )


__all__ = [
    "AmbiguousPrecursorOriginError",
    "build_invalid_catalog_report",
    "build_invalid_precursor_report",
    "build_released_explicit_report",
    "infer_precursor_coordinate_offset",
]
