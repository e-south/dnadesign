"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_released_workflow.py

Application orchestration for released-product snapback explicit workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.nickases.catalog import dump_nickase_catalog_yaml, load_merged_nickase_catalog
from dnadesign.cruncher.nickases.errors import NickaseCatalogError
from dnadesign.cruncher.nickases.scanning import enumerate_site_instances as enumerate_nickase_site_instances
from dnadesign.cruncher.nickases.selection import matching_nickase_warning_codes
from dnadesign.cruncher.release_enzymes.catalog import (
    dump_release_enzyme_catalog_yaml,
    load_merged_release_enzyme_catalog,
)
from dnadesign.cruncher.release_enzymes.errors import ReleaseEnzymeCatalogError
from dnadesign.cruncher.snapback.catalog_sources import catalog_source_label
from dnadesign.cruncher.snapback.load import load_released_snapback_spec
from dnadesign.cruncher.snapback.released_artifacts import (
    build_released_manifest,
    build_released_run_dir,
    ensure_released_run_dirs,
    released_design_id,
    snapshot_released_inputs,
    write_released_manifest,
    write_released_report,
    write_released_status,
    write_released_summary_table,
)
from dnadesign.cruncher.snapback.released_models import (
    ReleasedSnapbackEvaluationReport,
    ReleasedSnapbackReportMetadata,
    build_release_catalog_info,
    build_released_nickase_catalog_info,
)
from dnadesign.cruncher.snapback.released_projection import evaluate_released_precursor


def _infer_precursor_coordinate_offset(
    spec,
    *,
    nick_entry,
) -> int:
    matches = enumerate_nickase_site_instances(
        spec.input.precursor_top_strand,
        coordinate_offset=0,
        entry=nick_entry,
    )
    if spec.nick_stage.normalized_to_top_strand_nick:
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
    if len(offsets) != 1:
        return 0
    return next(iter(offsets))


def _invalid_catalog_report(
    spec,
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
    from dnadesign.cruncher.snapback.models import SnapbackIssue

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


def _build_report(
    spec,
    *,
    spec_path: Path,
    workspace_root: Path,
    nick_catalog,
    release_catalog,
    nick_catalog_source: str,
    release_catalog_source: str,
) -> ReleasedSnapbackEvaluationReport:
    nick_catalog_by_id = nick_catalog.by_id()
    if spec.nick_stage.nickase_variant_id not in nick_catalog_by_id:
        return _invalid_catalog_report(
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
        return _invalid_catalog_report(
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
        return _invalid_catalog_report(
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
    evaluation = evaluate_released_precursor(
        precursor_top_strand=spec.input.precursor_top_strand,
        nick_entry=nick_entry,
        release_entry=release_entry,
        target=spec.final_target,
        constraints=spec.constraints,
        nick_intended_site_sequence=spec.nick_stage.intended_site_sequence,
        release_intended_site_sequence=spec.release_stage.intended_site_sequence,
        precursor_coordinate_offset=_infer_precursor_coordinate_offset(spec, nick_entry=nick_entry),
        normalize_to_top_strand_nick=spec.nick_stage.normalized_to_top_strand_nick,
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


def validate_released_snapback_spec(path: str | Path) -> ReleasedSnapbackEvaluationReport:
    spec, spec_path, workspace_root = load_released_snapback_spec(path)
    nick_catalog_source = catalog_source_label(
        preset_ids=spec.nick_stage.catalog.resolved_preset_ids(),
        resolved_paths=spec.nick_stage.catalog.additional_paths,
    )
    release_catalog_source = catalog_source_label(
        preset_ids=spec.release_stage.catalog.resolved_preset_ids(),
        resolved_paths=spec.release_stage.catalog.additional_paths,
    )
    try:
        nick_catalog, nick_resolved_paths = load_merged_nickase_catalog(
            preset_id=spec.nick_stage.catalog.preset,
            additional_preset_ids=spec.nick_stage.catalog.additional_presets,
            additional_paths=spec.nick_stage.catalog.additional_paths,
            workspace_root=workspace_root,
        )
        release_catalog, release_resolved_paths = load_merged_release_enzyme_catalog(
            preset_id=spec.release_stage.catalog.preset,
            additional_preset_ids=spec.release_stage.catalog.additional_presets,
            additional_paths=spec.release_stage.catalog.additional_paths,
            workspace_root=workspace_root,
        )
    except (FileNotFoundError, NickaseCatalogError, ReleaseEnzymeCatalogError) as exc:
        return _invalid_catalog_report(
            spec,
            spec_path=spec_path,
            workspace_root=workspace_root,
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=spec.constraints.disallowed_nickase_warning_codes,
            code="CATALOG_LOAD_FAILED",
            message=str(exc),
        )
    return _build_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        nick_catalog=nick_catalog,
        release_catalog=release_catalog,
        nick_catalog_source=catalog_source_label(
            preset_ids=spec.nick_stage.catalog.resolved_preset_ids(),
            resolved_paths=nick_resolved_paths,
        ),
        release_catalog_source=catalog_source_label(
            preset_ids=spec.release_stage.catalog.resolved_preset_ids(),
            resolved_paths=release_resolved_paths,
        ),
    )


def run_released_snapback_design(path: str | Path, *, force_overwrite: bool = False):
    spec, spec_path, workspace_root = load_released_snapback_spec(path)
    nick_catalog, nick_resolved_paths = load_merged_nickase_catalog(
        preset_id=spec.nick_stage.catalog.preset,
        additional_preset_ids=spec.nick_stage.catalog.additional_presets,
        additional_paths=spec.nick_stage.catalog.additional_paths,
        workspace_root=workspace_root,
    )
    release_catalog, release_resolved_paths = load_merged_release_enzyme_catalog(
        preset_id=spec.release_stage.catalog.preset,
        additional_preset_ids=spec.release_stage.catalog.additional_presets,
        additional_paths=spec.release_stage.catalog.additional_paths,
        workspace_root=workspace_root,
    )
    nick_catalog_yaml = dump_nickase_catalog_yaml(nick_catalog)
    release_catalog_yaml = dump_release_enzyme_catalog_yaml(release_catalog)
    report = _build_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        nick_catalog=nick_catalog,
        release_catalog=release_catalog,
        nick_catalog_source=catalog_source_label(
            preset_ids=spec.nick_stage.catalog.resolved_preset_ids(),
            resolved_paths=nick_resolved_paths,
        ),
        release_catalog_source=catalog_source_label(
            preset_ids=spec.release_stage.catalog.resolved_preset_ids(),
            resolved_paths=release_resolved_paths,
        ),
    )
    run_id = released_design_id(
        spec_bytes=spec_path.read_bytes(),
        nick_catalog_bytes=nick_catalog_yaml.encode("utf-8"),
        release_catalog_bytes=release_catalog_yaml.encode("utf-8"),
    )
    run_dir = build_released_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
        released_design_run_id=run_id,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(
                "Released-product snapback run directory already exists: "
                f"{run_dir}. Use --force-overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    ensure_released_run_dirs(run_dir)
    snapshot_released_inputs(
        run_dir,
        spec_path=spec_path,
        nick_catalog_yaml=nick_catalog_yaml,
        release_catalog_yaml=release_catalog_yaml,
    )
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_released_report(run_dir, report)
    write_released_summary_table(run_dir, report)
    write_released_manifest(
        run_dir,
        build_released_manifest(
            run_dir=run_dir,
            workspace_root=workspace_root,
            spec_path=spec_path,
            report=report,
        ),
    )
    write_released_status(run_dir, report=report)
    return run_dir, report


__all__ = ["run_released_snapback_design", "validate_released_snapback_spec"]
