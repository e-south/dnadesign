"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_solve_workflow.py

Application orchestration for v2 snapback solve workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from dnadesign.cruncher.nickases.catalog import dump_nickase_catalog_yaml, load_merged_nickase_catalog
from dnadesign.cruncher.snapback.artifacts import (
    build_manifest,
    build_solve_manifest,
    build_solve_run_dir,
    catalog_snapshot_path,
    design_id,
    ensure_run_dirs,
    ensure_solve_run_dirs,
    post_nick_exposed_job_path,
    post_nick_exposed_visual_contract_path,
    post_nick_foldback_job_path,
    post_nick_foldback_visual_contract_path,
    pre_nick_duplex_job_path,
    pre_nick_duplex_visual_contract_path,
    solve_hit_run_dir,
    solve_id,
    spec_snapshot_path,
    write_baserender_job,
    write_candidate_table,
    write_manifest,
    write_report,
    write_solve_hits_table,
    write_solve_inputs,
    write_solve_manifest,
    write_solve_report,
    write_solve_status,
    write_status,
    write_view_bundle,
)
from dnadesign.cruncher.snapback.load import load_snapback_solve_spec
from dnadesign.cruncher.snapback.models import SnapbackEvaluationReport, SnapbackReportMetadata
from dnadesign.cruncher.snapback.planner import render_markdown_report
from dnadesign.cruncher.snapback.solver import render_solve_markdown_report, solve_snapback_search
from dnadesign.cruncher.snapback.view_contracts import (
    build_post_nick_exposed_snapback_visual,
    build_post_nick_exposed_view,
    build_post_nick_foldback_snapback_visual,
    build_post_nick_foldback_view,
    build_pre_nick_duplex_view,
    build_pre_nick_snapback_visual,
    build_single_view_job,
    build_views_manifest,
)


def _catalog_source_label(*, preset: str | None, resolved_paths: list[Path]) -> str:
    labels: list[str] = []
    if preset is not None:
        labels.append(f"preset:{preset}")
    labels.extend(str(path) for path in resolved_paths)
    return ", ".join(labels) if labels else "resolved_catalog"


def _resolve_catalog(spec, *, workspace_root: Path):
    catalog, resolved_paths = load_merged_nickase_catalog(
        preset_id=spec.catalog.preset,
        additional_paths=spec.catalog.additional_paths,
        workspace_root=workspace_root,
    )
    return catalog, resolved_paths, _catalog_source_label(preset=spec.catalog.preset, resolved_paths=resolved_paths)


def _view_title(*, spec_name: str, solution_id: str, state_label: str) -> str:
    _ = (spec_name, solution_id)
    return state_label


def _explicit_spec_payload_for_hit(spec, *, hit) -> dict[str, object]:
    candidate = hit.explicit_report
    payload: dict[str, object] = {
        "snapback": {
            "schema_version": 2,
            "contract": "single_nick_snapback_v2",
            "name": f"{spec.name}__hit_{hit.rank:02d}",
        },
        "input": {
            "canonical_top_strand": spec.input.canonical_top_strand.model_dump(mode="json"),
        },
        "design": {
            "nickase": {
                "variant_id": hit.variant_id,
                "catalog": {
                    "preset": spec.catalog.preset,
                    "additional_paths": [str(path) for path in spec.catalog.additional_paths],
                },
            },
            "orientation_policy": {
                "normalize_to_top_strand_nick": spec.nickase_policy.normalize_to_top_strand_nick,
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
                "terminal_ligatable_duplex_bp": spec.constraints.terminal_ligatable_duplex_bp.model_dump(mode="json"),
                "max_uninterrupted_duplex_bp": spec.constraints.max_uninterrupted_duplex_bp,
                "max_added_nt": spec.search.max_added_nt,
                "forbid_additional_target_strand_nicks": spec.constraints.forbid_additional_target_strand_nicks,
                "forbid_any_additional_nicks": spec.constraints.forbid_any_additional_nicks,
            },
            "sequence_quality": spec.sequence_quality.model_dump(mode="json"),
        },
        "output": {"run_dir": "outputs/snapback", "emit_visual_contracts": spec.output.emit_visual_contracts},
    }
    payload["output"]["emit_baserender_jobs"] = spec.output.emit_baserender_jobs
    return payload


def _write_materialized_hit_bundle(
    *,
    hit_run_dir: Path,
    spec_payload: dict[str, object],
    catalog_yaml: str,
    workspace_root: Path,
    candidate,
    catalog_source: str,
) -> None:
    ensure_run_dirs(hit_run_dir)
    spec_snapshot_path(hit_run_dir).write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")
    catalog_snapshot_path(hit_run_dir).write_text(catalog_yaml, encoding="utf-8")
    report = SnapbackEvaluationReport(
        status="satisfied",
        spec_name=spec_payload["snapback"]["name"],
        workspace_root=str(workspace_root),
        spec_path=str(spec_snapshot_path(hit_run_dir)),
        catalog_source=catalog_source,
        metadata=SnapbackReportMetadata(
            input_length_nt=len(candidate.input_sequence),
            added_nt=candidate.added_nt,
            designed_length_nt=len(candidate.designed_sequence),
            catalog_source=catalog_source,
        ),
        candidate=candidate,
    )
    write_report(hit_run_dir, report, markdown=render_markdown_report(report))
    write_candidate_table(hit_run_dir, report)
    if spec_payload["output"].get("emit_visual_contracts"):
        pre_nick_duplex = build_pre_nick_duplex_view(
            report=report,
            solution_id=hit_run_dir.name,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=hit_run_dir.name,
                state_label="pre-nick duplex",
            ),
        )
        post_nick_exposed = build_post_nick_exposed_view(
            report=report,
            solution_id=hit_run_dir.name,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=hit_run_dir.name,
                state_label="post-nick exposed",
            ),
        )
        post_nick_foldback = build_post_nick_foldback_view(
            report=report,
            solution_id=hit_run_dir.name,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=hit_run_dir.name,
                state_label="post-nick foldback",
            ),
        )
        pre_nick_duplex_visual_contract = build_pre_nick_snapback_visual(
            report=report,
            solution_id=hit_run_dir.name,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=hit_run_dir.name,
                state_label="pre-nick duplex",
            ),
        )
        post_nick_exposed_visual_contract = build_post_nick_exposed_snapback_visual(
            report=report,
            solution_id=hit_run_dir.name,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=hit_run_dir.name,
                state_label="post-nick exposed",
            ),
        )
        post_nick_foldback_visual_contract = build_post_nick_foldback_snapback_visual(
            report=report,
            solution_id=hit_run_dir.name,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=hit_run_dir.name,
                state_label="post-nick foldback",
            ),
        )
        write_view_bundle(
            hit_run_dir,
            pre_nick_duplex=pre_nick_duplex,
            post_nick_exposed=post_nick_exposed,
            post_nick_foldback=post_nick_foldback,
            pre_nick_duplex_visual_contract=pre_nick_duplex_visual_contract,
            post_nick_exposed_visual_contract=post_nick_exposed_visual_contract,
            post_nick_foldback_visual_contract=post_nick_foldback_visual_contract,
            manifest=build_views_manifest(
                solution_id=hit_run_dir.name,
                include_jobs=bool(spec_payload["output"].get("emit_baserender_jobs")),
            ),
        )
        if spec_payload["output"].get("emit_baserender_jobs"):
            write_baserender_job(
                pre_nick_duplex_job_path(hit_run_dir),
                build_single_view_job(
                    input_filename=pre_nick_duplex_visual_contract_path(hit_run_dir).name,
                    output_filename="pre_nick_duplex.png",
                ),
            )
            write_baserender_job(
                post_nick_exposed_job_path(hit_run_dir),
                build_single_view_job(
                    input_filename=post_nick_exposed_visual_contract_path(hit_run_dir).name,
                    output_filename="post_nick_exposed.png",
                ),
            )
            write_baserender_job(
                post_nick_foldback_job_path(hit_run_dir),
                build_single_view_job(
                    input_filename=post_nick_foldback_visual_contract_path(hit_run_dir).name,
                    output_filename="post_nick_foldback.png",
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
        spec_payload = _explicit_spec_payload_for_hit(spec, hit=hit)
        spec_text = yaml.safe_dump(spec_payload, sort_keys=False)
        explicit_design_id = design_id(
            spec_bytes=spec_text.encode("utf-8"),
            catalog_bytes=catalog_yaml.encode("utf-8"),
        )
        hit_dir = solve_hit_run_dir(run_dir, rank=hit.rank, explicit_design_id=explicit_design_id)
        _write_materialized_hit_bundle(
            hit_run_dir=hit_dir,
            spec_payload=spec_payload,
            catalog_yaml=catalog_yaml,
            workspace_root=workspace_root,
            candidate=hit.explicit_report,
            catalog_source=catalog_source,
        )
        materialized_hits.append(hit.model_copy(update={"materialized_run_dir": str(hit_dir.resolve())}))

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
