"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_solve_materialize.py

Per-hit materialization helpers for preserved-site Snapback solve runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.snapback_publish import (
    build_publication_bundle,
    write_publication_bundle,
)
from dnadesign.cruncher.app.snapback_solve_snapshot import (
    build_snapback_explicit_spec_payload_for_hit,
    dump_snapback_explicit_spec_yaml_for_hit,
)
from dnadesign.cruncher.snapback.artifacts import (
    build_manifest,
    design_id,
    ensure_run_dirs,
    solve_hit_run_dir,
    spec_snapshot_path,
    write_candidate_table,
    write_manifest,
    write_materialized_explicit_inputs,
    write_report,
    write_status,
)
from dnadesign.cruncher.snapback.models import SnapbackEvaluationReport, SnapbackReportMetadata
from dnadesign.cruncher.snapback.planner import render_markdown_report
from dnadesign.cruncher.snapback.solve_models import SingleNickSnapbackSolveSpec, SnapbackSolveHit


def _relative_to_workspace(path: Path, *, workspace_root: Path) -> str:
    return str(path.resolve().relative_to(workspace_root.resolve()))


def materialize_snapback_solve_hit(
    *,
    spec: SingleNickSnapbackSolveSpec,
    hit: SnapbackSolveHit,
    run_dir: Path,
    workspace_root: Path,
    catalog_yaml: str,
    catalog_source: str,
) -> SnapbackSolveHit:
    hit_run_dir = solve_hit_run_dir(run_dir, rank=hit.rank)
    spec_payload = build_snapback_explicit_spec_payload_for_hit(
        spec,
        hit=hit,
        workspace_root=workspace_root,
        hit_run_dir=hit_run_dir,
    )
    spec_text = dump_snapback_explicit_spec_yaml_for_hit(
        spec,
        hit=hit,
        workspace_root=workspace_root,
        hit_run_dir=hit_run_dir,
    )
    explicit_design_id = design_id(
        spec_bytes=spec_text.encode("utf-8"),
        catalog_bytes=catalog_yaml.encode("utf-8"),
    )
    ensure_run_dirs(
        hit_run_dir,
        include_visual_contracts=spec.output.emit_visual_contracts,
        include_baserender_jobs=spec.output.emit_baserender_jobs,
    )
    write_materialized_explicit_inputs(hit_run_dir, spec_payload=spec_payload, catalog_yaml=catalog_yaml)
    report = SnapbackEvaluationReport(
        status="satisfied",
        spec_name=str(spec_payload["snapback"]["name"]),
        run_dir=str(hit_run_dir.resolve()),
        workspace_root=str(workspace_root),
        spec_path=str(spec_snapshot_path(hit_run_dir)),
        catalog_source=catalog_source,
        metadata=SnapbackReportMetadata(
            input_length_nt=len(hit.explicit_report.input_sequence),
            added_nt=hit.explicit_report.added_nt,
            designed_length_nt=len(hit.explicit_report.designed_sequence),
            catalog_source=catalog_source,
            catalog_presets=[
                *([spec.catalog.preset] if spec.catalog.preset else []),
                *list(spec.catalog.additional_presets),
            ],
            catalog_variants=[hit.nickase],
        ),
        candidate=hit.explicit_report,
    )
    write_report(hit_run_dir, report, markdown=render_markdown_report(report))
    write_candidate_table(hit_run_dir, report)
    if spec.output.emit_visual_contracts:
        write_publication_bundle(
            hit_run_dir,
            bundle=build_publication_bundle(
                report=report,
                solution_id=explicit_design_id,
                include_jobs=spec.output.emit_baserender_jobs,
                render_format=spec.output.render_format,
            ),
        )
    write_manifest(
        hit_run_dir,
        build_manifest(
            run_dir=hit_run_dir,
            workspace_root=workspace_root,
            spec_path=spec_snapshot_path(hit_run_dir),
            report=report,
        ),
    )
    write_status(hit_run_dir, report=report)
    return hit.model_copy(
        update={"materialized_run_dir": _relative_to_workspace(hit_run_dir, workspace_root=workspace_root)}
    )


def materialize_snapback_solve_hits(
    *,
    spec: SingleNickSnapbackSolveSpec,
    hits: list[SnapbackSolveHit],
    run_dir: Path,
    workspace_root: Path,
    catalog_yaml: str,
    catalog_source: str,
) -> list[SnapbackSolveHit]:
    return [
        materialize_snapback_solve_hit(
            spec=spec,
            hit=hit,
            run_dir=run_dir,
            workspace_root=workspace_root,
            catalog_yaml=catalog_yaml,
            catalog_source=catalog_source,
        )
        for hit in hits[: spec.search.materialize_top_k]
    ]


__all__ = [
    "materialize_snapback_solve_hit",
    "materialize_snapback_solve_hits",
]
