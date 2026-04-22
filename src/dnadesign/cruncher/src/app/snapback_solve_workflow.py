"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_solve_workflow.py

Application orchestration for v3 co-design snapback solve workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from dnadesign.cruncher.app.snapback_publish import (
    build_publication_bundle,
    write_publication_bundle,
)
from dnadesign.cruncher.nickases.catalog import dump_nickase_catalog_yaml, load_merged_nickase_catalog
from dnadesign.cruncher.snapback.artifacts import (
    build_manifest,
    build_solve_manifest,
    build_solve_run_dir,
    design_id,
    ensure_run_dirs,
    ensure_solve_run_dirs,
    solve_hit_run_dir,
    solve_id,
    spec_snapshot_path,
    write_candidate_table,
    write_manifest,
    write_materialized_explicit_inputs,
    write_report,
    write_solve_frontier_table,
    write_solve_hits_table,
    write_solve_inputs,
    write_solve_manifest,
    write_solve_report,
    write_solve_status,
    write_status,
)
from dnadesign.cruncher.snapback.catalog_sources import catalog_source_label
from dnadesign.cruncher.snapback.load import load_snapback_solve_spec
from dnadesign.cruncher.snapback.models import SnapbackEvaluationReport, SnapbackReportMetadata
from dnadesign.cruncher.snapback.planner import render_markdown_report
from dnadesign.cruncher.snapback.solver import render_solve_markdown_report, solve_snapback_search


def _resolve_catalog(spec, *, workspace_root: Path):
    catalog, resolved_paths = load_merged_nickase_catalog(
        preset_id=spec.catalog.preset,
        additional_preset_ids=spec.catalog.additional_presets,
        additional_paths=spec.catalog.additional_paths,
        workspace_root=workspace_root,
    )
    return (
        catalog,
        resolved_paths,
        catalog_source_label(preset_ids=spec.catalog.resolved_preset_ids(), resolved_paths=resolved_paths),
    )


def _explicit_spec_payload_for_hit(spec, *, hit, workspace_root: Path, hit_run_dir: Path) -> dict[str, object]:
    candidate = hit.explicit_report
    resolved_terminal_ligatable_duplex_bp = spec.resolved_terminal_ligatable_duplex_bp()
    resolved_max_uninterrupted_duplex_bp = spec.resolved_max_uninterrupted_duplex_bp()
    materialized_run_dir = hit_run_dir.resolve().relative_to(workspace_root.resolve())
    payload: dict[str, object] = {
        "snapback": {
            "schema_version": 2,
            "contract": "single_nick_snapback_v2",
            "name": f"{spec.name}__hit_{hit.rank:02d}",
        },
        "input": {
            "canonical_top_strand": {
                "sequence": candidate.input_sequence,
                "protected_region": candidate.protected_region.model_dump(mode="json"),
                "pre_nick_duplex_window": candidate.pre_nick_duplex_window.model_dump(mode="json"),
            },
        },
        "design": {
            "nickase": {
                "variant_id": hit.variant_id,
                "catalog": {
                    "preset": spec.catalog.preset,
                    "additional_presets": list(spec.catalog.additional_presets),
                    "additional_paths": [str(path) for path in spec.catalog.additional_paths],
                },
            },
            "orientation_policy": {
                "normalize_to_top_strand_nick": spec.orientation_policy.normalize_to_top_strand_nick,
                "release_direction": "left_to_right_from_nick",
            },
            "single_nick_goal": {
                "nick_boundary_window": {
                    "min": candidate.nick_boundary,
                    "max": candidate.nick_boundary,
                }
            },
            "topology": {
                "retained_homology_window": candidate.retained_homology_window.model_dump(mode="json"),
                "cap_sequence": candidate.cap_sequence,
                "foldback_arm": candidate.foldback_arm,
                "homology_policy": {
                    "max_mismatches": spec.search.max_mismatches,
                    "min_paired_bp": candidate.paired_bp,
                    "max_paired_bp": candidate.paired_bp,
                },
            },
            "constraints": {
                "terminal_ligatable_duplex_bp": resolved_terminal_ligatable_duplex_bp.model_dump(mode="json"),
                "max_uninterrupted_duplex_bp": resolved_max_uninterrupted_duplex_bp,
                "max_added_nt": spec.search.max_added_nt,
                "forbid_additional_target_strand_nicks": spec.constraints.forbid_additional_target_strand_nicks,
                "forbid_any_additional_nicks": spec.constraints.forbid_any_additional_nicks,
            },
            "sequence_quality": spec.sequence_quality.model_dump(mode="json"),
        },
        "output": {
            "run_dir": str(materialized_run_dir),
            "emit_visual_contracts": spec.output.emit_visual_contracts,
            "render_format": spec.output.render_format,
        },
    }
    payload["output"]["emit_baserender_jobs"] = spec.output.emit_baserender_jobs
    return payload


def _write_materialized_hit_bundle(
    *,
    hit_run_dir: Path,
    solution_id: str,
    spec_payload: dict[str, object],
    catalog_yaml: str,
    workspace_root: Path,
    candidate,
    nickase_info,
    catalog_source: str,
) -> None:
    ensure_run_dirs(
        hit_run_dir,
        include_visual_contracts=bool(spec_payload["output"].get("emit_visual_contracts")),
        include_baserender_jobs=bool(spec_payload["output"].get("emit_baserender_jobs")),
    )
    write_materialized_explicit_inputs(hit_run_dir, spec_payload=spec_payload, catalog_yaml=catalog_yaml)
    report = SnapbackEvaluationReport(
        status="satisfied",
        spec_name=spec_payload["snapback"]["name"],
        run_dir=str(hit_run_dir.resolve()),
        workspace_root=str(workspace_root),
        spec_path=str(spec_snapshot_path(hit_run_dir)),
        catalog_source=catalog_source,
        metadata=SnapbackReportMetadata(
            input_length_nt=len(candidate.input_sequence),
            added_nt=candidate.added_nt,
            designed_length_nt=len(candidate.designed_sequence),
            catalog_source=catalog_source,
            catalog_presets=[
                *(
                    [spec_payload["design"]["nickase"]["catalog"]["preset"]]
                    if spec_payload["design"]["nickase"]["catalog"].get("preset")
                    else []
                ),
                *list(spec_payload["design"]["nickase"]["catalog"].get("additional_presets", [])),
            ],
            catalog_variants=[nickase_info],
        ),
        candidate=candidate,
    )
    write_report(hit_run_dir, report, markdown=render_markdown_report(report))
    write_candidate_table(hit_run_dir, report)
    if spec_payload["output"].get("emit_visual_contracts"):
        write_publication_bundle(
            hit_run_dir,
            bundle=build_publication_bundle(
                report=report,
                solution_id=solution_id,
                include_jobs=bool(spec_payload["output"].get("emit_baserender_jobs")),
                render_format=str(spec_payload["output"].get("render_format", "png")),
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


def run_snapback_solve(path: str | Path, *, force_overwrite: bool = False):
    spec, spec_path, workspace_root = load_snapback_solve_spec(path)
    catalog, resolved_paths, catalog_source = _resolve_catalog(spec, workspace_root=workspace_root)
    report = solve_snapback_search(spec, spec_path=spec_path, workspace_root=workspace_root, catalog=catalog)
    catalog_yaml = dump_nickase_catalog_yaml(catalog)
    snapback_solve_id = solve_id(
        spec_bytes=spec_path.read_bytes(),
        catalog_bytes=catalog_yaml.encode("utf-8"),
    )
    run_dir = build_solve_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
        snapback_solve_id=snapback_solve_id,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(
                f"Snapback solve run directory already exists: {run_dir}. Use --force-overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    ensure_solve_run_dirs(run_dir)
    write_solve_inputs(run_dir, spec_path=spec_path, catalog_yaml=catalog_yaml)

    materialized_hits = []
    for hit in report.hits[: spec.search.materialize_top_k]:
        hit_dir = solve_hit_run_dir(run_dir, rank=hit.rank)
        spec_payload = _explicit_spec_payload_for_hit(
            spec,
            hit=hit,
            workspace_root=workspace_root,
            hit_run_dir=hit_dir,
        )
        spec_text = yaml.safe_dump(spec_payload, sort_keys=False)
        explicit_design_id = design_id(
            spec_bytes=spec_text.encode("utf-8"),
            catalog_bytes=catalog_yaml.encode("utf-8"),
        )
        _write_materialized_hit_bundle(
            hit_run_dir=hit_dir,
            solution_id=explicit_design_id,
            spec_payload=spec_payload,
            catalog_yaml=catalog_yaml,
            workspace_root=workspace_root,
            candidate=hit.explicit_report,
            nickase_info=hit.nickase,
            catalog_source=catalog_source,
        )
        materialized_hits.append(
            hit.model_copy(
                update={"materialized_run_dir": str(hit_dir.resolve().relative_to(workspace_root.resolve()))}
            )
        )

    remaining_hits = report.hits[spec.search.materialize_top_k :]
    report = report.model_copy(
        update={
            "solve_id": snapback_solve_id,
            "run_dir": str(run_dir.resolve()),
            "metadata": report.metadata.model_copy(update={"materialized_hit_count": len(materialized_hits)}),
            "hits": [*materialized_hits, *remaining_hits],
        }
    )
    write_solve_report(run_dir, report, markdown=render_solve_markdown_report(report))
    write_solve_hits_table(run_dir, report)
    write_solve_frontier_table(run_dir, report)
    write_solve_manifest(
        run_dir,
        build_solve_manifest(run_dir=run_dir, workspace_root=workspace_root, spec_path=spec_path, report=report),
    )
    write_solve_status(
        run_dir,
        report=report,
        status_message=(
            f"snapback solve {report.status} (hits={len(report.hits)}, materialized={len(materialized_hits)})"
        ),
    )
    return run_dir, report
