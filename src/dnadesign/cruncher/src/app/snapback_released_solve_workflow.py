"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_released_solve_workflow.py

Application orchestration for released-product snapback solve/materialization.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.nickases.catalog import dump_nickase_catalog_yaml, load_merged_nickase_catalog
from dnadesign.cruncher.release_enzymes.catalog import (
    dump_release_enzyme_catalog_yaml,
    load_merged_release_enzyme_catalog,
)
from dnadesign.cruncher.snapback.catalog_sources import catalog_source_label
from dnadesign.cruncher.snapback.released_artifacts import (
    build_released_run_dir,
    build_released_solve_manifest,
    ensure_released_solve_run_dirs,
    released_pre_nick_site_json_path,
    released_projection_json_path,
    released_release_site_json_path,
    released_solve_hit_json_path,
    released_solve_hit_plot_context_path,
    released_solve_hit_plot_path,
    released_solve_hit_run_dir,
    snapshot_released_solve_inputs,
    write_released_solve_manifest,
    write_released_solve_report,
    write_released_solve_status,
    write_released_solve_summary_table,
)
from dnadesign.cruncher.snapback.released_hit_plot import build_released_hit_plot_context, render_released_hit_plot
from dnadesign.cruncher.snapback.released_models import (
    ReleasedSolveHit,
    ReleasedSolveOutputConfig,
    ReleasedSolveReport,
    ReleasedSolveReportMetadata,
    ReleasedTargetSearchHit,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.snapback.released_target_search import search_released_target_hits
from dnadesign.cruncher.viz.mpl import ensure_workspace_mpl_cache


def _relative_to_workspace(path: Path, *, workspace_root: Path) -> str:
    return str(path.resolve().relative_to(workspace_root.resolve()))


def _request_snapshot_payload(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    output: ReleasedSolveOutputConfig,
) -> dict[str, object]:
    return {
        "released_solve": {
            "schema_version": 1,
            "kind": "single_nick_released_solve_v1",
        },
        "target": request.target.model_dump(mode="json"),
        "nick_sources": {
            "preset": request.nick_sources.preset,
            "additional_presets": list(request.nick_sources.additional_presets),
            "additional_paths": [str(path) for path in request.nick_sources.additional_paths],
        },
        "release_sources": {
            "preset": request.release_sources.preset,
            "additional_presets": list(request.release_sources.additional_presets),
            "additional_paths": [str(path) for path in request.release_sources.additional_paths],
        },
        "search": request.search.model_dump(mode="json"),
        "output": output.model_dump(mode="json"),
    }


def _ensure_materialized_hit_dirs(hit_run_dir: Path) -> None:
    hit_run_dir.mkdir(parents=True, exist_ok=True)
    (hit_run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (hit_run_dir / "plots").mkdir(parents=True, exist_ok=True)


def _materialize_hit_bundle(
    *,
    hit: ReleasedTargetSearchHit,
    hit_run_dir: Path,
    workspace_root: Path,
    output: ReleasedSolveOutputConfig,
) -> tuple[str, str | None, str | None]:
    _ensure_materialized_hit_dirs(hit_run_dir)
    atomic_write_json(released_solve_hit_json_path(hit_run_dir), hit.model_dump(mode="json"))
    atomic_write_json(released_projection_json_path(hit_run_dir), hit.projection.model_dump(mode="json"))
    atomic_write_json(
        released_pre_nick_site_json_path(hit_run_dir),
        {
            "site": hit.pre_nick_site.model_dump(mode="json"),
            "event": hit.pre_nick_event.model_dump(mode="json"),
        },
    )
    atomic_write_json(
        released_release_site_json_path(hit_run_dir),
        {
            "site": hit.release_site.model_dump(mode="json"),
            "event": hit.release_event.model_dump(mode="json"),
        },
    )

    rendered_plot_path: Path | None = None
    if output.emit_renders:
        ensure_workspace_mpl_cache(workspace_root)
        rendered_plot_path = released_solve_hit_plot_path(hit_run_dir, fmt=output.render_format)
        plot_context = render_released_hit_plot(hit, rendered_plot_path)
        if not rendered_plot_path.exists():
            raise FileNotFoundError(f"Released solve render missing expected plot: {rendered_plot_path}")
    else:
        plot_context = build_released_hit_plot_context(hit)
    atomic_write_json(released_solve_hit_plot_context_path(hit_run_dir), plot_context)

    return (
        _relative_to_workspace(hit_run_dir, workspace_root=workspace_root),
        None,
        _relative_to_workspace(rendered_plot_path, workspace_root=workspace_root) if rendered_plot_path else None,
    )


def run_released_snapback_solve(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    output: ReleasedSolveOutputConfig,
    workspace_root: Path,
    force_overwrite: bool = False,
) -> tuple[Path, ReleasedSolveReport]:
    nick_catalog, nick_resolved_paths = load_merged_nickase_catalog(
        preset_id=request.nick_sources.preset,
        additional_preset_ids=request.nick_sources.additional_presets,
        additional_paths=request.nick_sources.additional_paths,
        workspace_root=workspace_root,
    )
    release_catalog, release_resolved_paths = load_merged_release_enzyme_catalog(
        preset_id=request.release_sources.preset,
        additional_preset_ids=request.release_sources.additional_presets,
        additional_paths=request.release_sources.additional_paths,
        workspace_root=workspace_root,
    )
    nick_catalog_source = catalog_source_label(
        preset_ids=request.nick_sources.resolved_preset_ids(),
        resolved_paths=nick_resolved_paths,
    )
    release_catalog_source = catalog_source_label(
        preset_ids=request.release_sources.resolved_preset_ids(),
        resolved_paths=release_resolved_paths,
    )
    search_report = search_released_target_hits(
        request=request,
        nick_catalog=nick_catalog,
        release_catalog=release_catalog,
        workspace_root=workspace_root,
        nick_catalog_source=nick_catalog_source,
        release_catalog_source=release_catalog_source,
    )

    nick_catalog_yaml = dump_nickase_catalog_yaml(nick_catalog)
    release_catalog_yaml = dump_release_enzyme_catalog_yaml(release_catalog)
    request_yaml = yaml.safe_dump(
        _request_snapshot_payload(request=request, output=output),
        sort_keys=False,
    )
    run_dir = build_released_run_dir(
        workspace_root=workspace_root,
        run_root=output.run_dir,
        released_design_run_id="released_solve",
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(
                f"Released-product snapback solve run directory already exists: {run_dir}. "
                "Use --force-overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    ensure_released_solve_run_dirs(run_dir)
    snapshot_released_solve_inputs(
        run_dir,
        request_yaml=request_yaml,
        nick_catalog_yaml=nick_catalog_yaml,
        release_catalog_yaml=release_catalog_yaml,
    )

    selected_hits = search_report.exact_hits if search_report.exact_hits else search_report.near_hits
    selected_hit_kind = "exact" if search_report.exact_hits else "nearest" if search_report.near_hits else None
    materialized_hits: list[ReleasedSolveHit] = []
    issues = list(search_report.issues)
    for materialize_rank, hit in enumerate(selected_hits[: output.materialize_top_k], start=1):
        hit_run_dir = released_solve_hit_run_dir(run_dir, rank=materialize_rank)
        materialized_run_dir, render_job_path, rendered_plot_path = _materialize_hit_bundle(
            hit=hit,
            hit_run_dir=hit_run_dir,
            workspace_root=workspace_root,
            output=output,
        )
        materialized_hits.append(
            ReleasedSolveHit(
                rank=materialize_rank,
                hit_kind=hit.hit_kind,
                nickase_variant_id=hit.nickase_variant_id,
                release_variant_id=hit.release_variant_id,
                materialized_run_dir=materialized_run_dir,
                render_job_path=render_job_path,
                rendered_plot_path=rendered_plot_path,
                target_search_hit=hit,
            )
        )

    status = (
        "exact_hits_materialized"
        if selected_hit_kind == "exact"
        else "near_hits_materialized"
        if selected_hit_kind == "nearest"
        else "no_hits"
    )
    report = ReleasedSolveReport(
        status=status,
        workspace_root=str(workspace_root.resolve()),
        run_dir=str(run_dir.resolve()),
        metadata=ReleasedSolveReportMetadata(
            target=request.target,
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=list(request.search.disallowed_nickase_warning_codes),
            evaluated_pair_count=search_report.metadata.evaluated_pair_count,
            available_exact_hit_count=search_report.metadata.pre_truncation_exact_hit_count,
            available_near_hit_count=search_report.metadata.pre_truncation_near_hit_count,
            selected_hit_kind=selected_hit_kind,
            materialized_hit_count=len(materialized_hits),
            requested_materialize_top_k=output.materialize_top_k,
            render_format=output.render_format,
            emit_renders=output.emit_renders,
            blocker_counts=dict(search_report.metadata.blocker_counts),
        ),
        issues=issues,
        search_report=search_report,
        hits=materialized_hits,
    )
    write_released_solve_report(run_dir, report)
    write_released_solve_summary_table(run_dir, report)
    write_released_solve_manifest(
        run_dir,
        build_released_solve_manifest(run_dir=run_dir, workspace_root=workspace_root, report=report),
    )
    write_released_solve_status(run_dir, report=report)
    return run_dir, report


__all__ = ["run_released_snapback_solve"]
