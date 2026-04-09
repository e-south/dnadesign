"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/cassette_solve_workflow.py

Application orchestration for cassette solve/search workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import yaml

from dnadesign.cruncher.cassette.artifacts import (
    build_solve_manifest,
    build_solve_run_dir,
    design_id,
    ensure_solve_run_dirs,
    hairpin_job_path,
    hairpin_view_path,
    linear_duplex_job_path,
    linear_duplex_view_path,
    solve_hit_dir,
    solve_id,
    solve_resolved_catalog_path,
    top_hits_duplex_job_path,
    top_hits_hairpin_job_path,
    top_hits_hairpin_jsonl_path,
    top_hits_linear_duplex_jsonl_path,
    views_manifest_path,
    write_baserender_job,
    write_jsonl_records,
    write_solve_hit_bundle,
    write_solve_hits_table,
    write_solve_inputs,
    write_solve_manifest,
    write_solve_report,
    write_solve_status,
    write_view_bundle,
)
from dnadesign.cruncher.cassette.catalog import dump_nickase_catalog_yaml, load_merged_nickase_catalog
from dnadesign.cruncher.cassette.errors import CassetteSpecError, NickaseCatalogError
from dnadesign.cruncher.cassette.load import load_cassette_solve_spec, resolve_workspace_root_for_solve_spec
from dnadesign.cruncher.cassette.planner import render_markdown_report
from dnadesign.cruncher.cassette.solve_models import SolveReport, SolveReportMetadata
from dnadesign.cruncher.cassette.solver import build_solve_report, solve_cassette_search
from dnadesign.cruncher.cassette.view_contracts import (
    build_hairpin_topology_view,
    build_linear_duplex_view,
    build_single_view_job,
    build_top_hits_job,
    build_views_manifest,
)


def _issue_from_exception(*, code: str, message: str, details: dict[str, object] | None = None) -> dict[str, object]:
    return {"code": code, "message": message, "details": details or {}}


@dataclass(frozen=True)
class _HitPublicationBundle:
    explicit_design_id: str
    canonical_spec_payload: dict[str, object]
    linear_view: dict[str, object] | None
    hairpin_view: dict[str, object] | None


def _build_invalid_report(
    *,
    status: str,
    spec_path: Path,
    workspace_root: Path | None,
    code: str,
    message: str,
) -> SolveReport:
    return SolveReport.model_validate(
        {
            "status": status,
            "workspace_root": str(workspace_root) if workspace_root is not None else None,
            "spec_path": str(spec_path),
            "metadata": SolveReportMetadata().model_dump(mode="json"),
            "issues": [_issue_from_exception(code=code, message=message)],
            "hits": [],
        }
    )


def _mapping_member(payload: object, key: str) -> object | None:
    if not isinstance(payload, dict):
        return None
    return payload.get(key)


def _best_effort_preflight_run_root(spec_path: Path) -> tuple[Path, list[str]]:
    default_run_root = Path("outputs/cassette_solves")
    unresolved_note = (
        "Preflight solve artifacts used the default output.run_dir because the "
        "solve spec output path could not be resolved."
    )
    unsafe_note = (
        "Preflight solve artifacts used the default output.run_dir because the solve spec output path was unsafe."
    )
    try:
        payload = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return default_run_root, [unresolved_note]
    if not isinstance(payload, dict):
        return default_run_root, [unresolved_note]
    cassette_solve = _mapping_member(payload, "cassette_solve")
    output = _mapping_member(cassette_solve, "output")
    raw_run_dir = _mapping_member(output, "run_dir") if isinstance(output, dict) else None
    if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
        return default_run_root, [unresolved_note]
    run_root = Path(raw_run_dir)
    if run_root.is_absolute() or any(part == ".." for part in run_root.parts):
        return default_run_root, [unsafe_note]
    return run_root, []


def _preflight_solve_id(*, report: SolveReport, spec_path: Path) -> str:
    try:
        spec_bytes = spec_path.read_bytes()
    except FileNotFoundError:
        spec_bytes = str(spec_path).encode("utf-8")
    fingerprint = "\n".join(
        [
            f"status={report.status}",
            f"preset={report.metadata.catalog_preset or ''}",
            f"overlays={','.join(report.metadata.catalog_additional_paths)}",
        ]
    ).encode("utf-8")
    return solve_id(spec_bytes=spec_bytes, catalog_bytes=fingerprint)


def _persist_preflight_failure(
    *,
    report: SolveReport,
    spec_path: Path,
    workspace_root: Path | None,
    run_root: Path,
    force_overwrite: bool,
) -> tuple[Path | None, SolveReport]:
    if workspace_root is None:
        return None, report
    cassette_solve_id = _preflight_solve_id(report=report, spec_path=spec_path)
    run_dir = build_solve_run_dir(
        workspace_root=workspace_root,
        run_root=run_root,
        cassette_solve_id=cassette_solve_id,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(
                f"Cassette solve run directory already exists: {run_dir}. Use --force-overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    ensure_solve_run_dirs(run_dir)
    write_solve_inputs(run_dir, spec_path=spec_path if spec_path.exists() else None)
    report = report.model_copy(
        update={
            "solve_id": cassette_solve_id,
            "run_dir": str(run_dir.resolve()),
            "workspace_root": str(workspace_root.resolve()),
        }
    )
    _write_solve_bundle(
        run_dir=run_dir,
        report=report,
        workspace_root=workspace_root,
        spec_path=spec_path,
        status_message=f"cassette solve {report.status} (preflight; hits={len(report.hits)})",
    )
    return run_dir, report


def _write_solve_bundle(
    *,
    run_dir: Path,
    report: SolveReport,
    workspace_root: Path,
    spec_path: Path,
    status_message: str,
) -> None:
    write_solve_report(run_dir, report, markdown=render_solve_markdown_report(report))
    write_solve_hits_table(run_dir, report)
    write_solve_manifest(
        run_dir,
        build_solve_manifest(
            run_dir=run_dir,
            workspace_root=workspace_root,
            spec_path=spec_path,
            report=report,
        ),
    )
    write_solve_status(
        run_dir,
        report=report,
        status_message=status_message,
    )


def _relative_to_run(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.resolve().relative_to(run_dir.resolve()))


def _relative_to_run_if_exists(run_dir: Path, path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return _relative_to_run(run_dir, path)


def _canonical_hit_spec_payload(hit_record: object) -> dict[str, object]:
    explicit_spec = getattr(hit_record, "explicit_spec")
    return {"cassette": explicit_spec.model_dump(mode="json")}


def _materialized_hit_spec_payload(*, canonical_spec_payload: dict[str, object], run_dir: Path) -> dict[str, object]:
    payload = deepcopy(canonical_spec_payload)
    payload["cassette"]["catalog"]["path"] = str(solve_resolved_catalog_path(run_dir).resolve())
    return payload


def _explicit_design_id_for_payload(*, spec_payload: dict[str, object], resolved_catalog_yaml: str) -> str:
    return design_id(
        spec_bytes=yaml.safe_dump(spec_payload, sort_keys=False).encode("utf-8"),
        catalog_bytes=resolved_catalog_yaml.encode("utf-8"),
    )


def _hit_view_title(*, rank: int, solution_id: str, view_label: str) -> str:
    return f"Hit {rank} [{solution_id}] - {view_label}"


def _build_hit_publication_bundle(
    *,
    hit: object,
    hit_record: object,
    cassette_solve_id: str,
    resolved_catalog_yaml: str,
    emit_visual_contracts: bool,
) -> _HitPublicationBundle:
    canonical_spec_payload = _canonical_hit_spec_payload(hit_record)
    explicit_design_id = _explicit_design_id_for_payload(
        spec_payload=canonical_spec_payload,
        resolved_catalog_yaml=resolved_catalog_yaml,
    )
    if not emit_visual_contracts:
        return _HitPublicationBundle(
            explicit_design_id=explicit_design_id,
            canonical_spec_payload=canonical_spec_payload,
            linear_view=None,
            hairpin_view=None,
        )
    linear_view = build_linear_duplex_view(
        report=hit_record.report,
        solution_id=hit.solution_id,
        title=_hit_view_title(rank=hit.rank, solution_id=hit.solution_id, view_label="Linear duplex"),
        rank=hit.rank,
        source_solve_id=cassette_solve_id,
        explicit_design_id=explicit_design_id,
    )
    hairpin_view = build_hairpin_topology_view(
        report=hit_record.report,
        solution_id=hit.solution_id,
        title=_hit_view_title(rank=hit.rank, solution_id=hit.solution_id, view_label="ssDNA hairpin"),
        rank=hit.rank,
        source_solve_id=cassette_solve_id,
        explicit_design_id=explicit_design_id,
    )
    return _HitPublicationBundle(
        explicit_design_id=explicit_design_id,
        canonical_spec_payload=canonical_spec_payload,
        linear_view=linear_view.model_dump(mode="json"),
        hairpin_view=hairpin_view.model_dump(mode="json"),
    )


def render_solve_markdown_report(report: SolveReport) -> str:
    lines = [
        "# Cassette Solve Report",
        "",
        f"- status: {report.status}",
        f"- spec_path: {report.spec_path}",
    ]
    if report.solve_id:
        lines.append(f"- solve_id: {report.solve_id}")
    if report.run_dir:
        lines.append(f"- run_dir: {report.run_dir}")
    if report.metadata.catalog_preset:
        lines.append(f"- catalog_preset: {report.metadata.catalog_preset}")
    for path in report.metadata.catalog_additional_paths:
        lines.append(f"- catalog_overlay: {path}")
    for code, warning in zip_longest(report.metadata.warning_codes, report.metadata.warnings, fillvalue=None):
        if code and warning:
            lines.append(f"- warning[{code}]: {warning}")
        elif warning:
            lines.append(f"- warning: {warning}")
        elif code:
            lines.append(f"- warning_code: {code}")
    lines.extend(
        [
            f"- enumerated_candidate_count: {report.metadata.enumerated_candidate_count}",
            f"- accepted_candidate_count: {report.metadata.accepted_candidate_count}",
            f"- considered_variant_pair_count: {report.metadata.considered_variant_pair_count}",
            f"- materialized_hit_count: {report.metadata.materialized_hit_count}",
        ]
    )
    if report.selection_summary is not None:
        lines.extend(
            [
                f"- selection_policy: {report.selection_summary.policy}",
                f"- selection_pool_size: {report.selection_summary.pool_size}",
                f"- accepted_pool_size: {report.selection_summary.accepted_pool_size}",
                f"- accepted_pool_truncated: {report.selection_summary.accepted_pool_truncated}",
                f"- selection_defaulted: {report.selection_summary.selection_policy_defaulted}",
                f"- selection_non_exhaustive_reason: {report.selection_summary.selection_pool_non_exhaustive_reason}",
                f"- selection_policy_underfilled: {report.selection_summary.policy_underfilled}",
                f"- selection_policy_limited_hit_count: {report.selection_summary.policy_limited_hit_count}",
            ]
        )
    if report.issues:
        lines.extend(["", "## Issues"])
        for issue in report.issues:
            lines.append(f"- {issue.code}: {issue.message}")
    if report.hits:
        lines.extend(["", "## Hits"])
        for hit in report.hits:
            lines.append(
                f"- rank {hit.rank}: {hit.hit_id} score={hit.score} "
                f"{hit.left_variant_id}@{hit.left_nick_boundary} -> {hit.right_variant_id}@{hit.right_nick_boundary} "
                f"gc={hit.gc_fraction:.3f} extra_sites={hit.extra_site_count}"
            )
    return "\n".join(lines) + "\n"


def solve_cassette_spec(path: str | Path) -> SolveReport:
    spec_path = Path(path).expanduser().resolve()
    report, _solve_spec, _resolved_spec_path, _workspace_root, _catalog, _search_result = _solve_loaded_spec(spec_path)
    return report


def _solve_loaded_spec(
    spec_path: Path,
) -> tuple[
    SolveReport,
    object | None,
    Path | None,
    Path | None,
    object | None,
    object | None,
]:
    try:
        solve_spec, resolved_spec_path, workspace_root = load_cassette_solve_spec(spec_path)
    except (CassetteSpecError, FileNotFoundError) as exc:
        return (
            _build_invalid_report(
                status="invalid_spec",
                spec_path=spec_path,
                workspace_root=None,
                code="INVALID_SOLVE_SPEC",
                message=str(exc),
            ),
            None,
            None,
            None,
            None,
            None,
        )

    try:
        catalog, resolved_overlay_paths = load_merged_nickase_catalog(
            preset_id=solve_spec.catalog.preset,
            additional_paths=solve_spec.catalog.additional_paths,
            workspace_root=workspace_root,
        )
    except (NickaseCatalogError, FileNotFoundError) as exc:
        report = _build_invalid_report(
            status="invalid_catalog",
            spec_path=resolved_spec_path,
            workspace_root=workspace_root,
            code="INVALID_SOLVE_CATALOG",
            message=str(exc),
        )
        report.metadata = report.metadata.model_copy(
            update={
                "catalog_preset": solve_spec.catalog.preset,
                "catalog_additional_paths": [str(path) for path in solve_spec.catalog.additional_paths],
            }
        )
        return report, solve_spec, resolved_spec_path, workspace_root, None, None

    placeholder_catalog_path = (
        resolved_overlay_paths[0]
        if len(resolved_overlay_paths) == 1 and solve_spec.catalog.preset is None
        else Path(f"merged::{solve_spec.catalog.preset or 'overlay_catalog'}")
    )
    search_result = solve_cassette_search(
        solve_spec=solve_spec,
        spec_path=resolved_spec_path,
        workspace_root=workspace_root,
        catalog=catalog,
        catalog_path=placeholder_catalog_path,
    )
    report = build_solve_report(
        solve_spec=solve_spec,
        spec_path=resolved_spec_path,
        workspace_root=workspace_root,
        catalog=catalog,
        search_result=search_result,
    )
    report.metadata = report.metadata.model_copy(
        update={
            "catalog_preset": solve_spec.catalog.preset,
            "catalog_additional_paths": [str(path) for path in resolved_overlay_paths],
        }
    )
    return report, solve_spec, resolved_spec_path, workspace_root, catalog, search_result


def run_cassette_solve(path: str | Path, *, force_overwrite: bool = False) -> tuple[Path | None, SolveReport]:
    spec_path = Path(path).expanduser().resolve()
    report, solve_spec, resolved_spec_path, workspace_root, catalog, search_result = _solve_loaded_spec(spec_path)
    if report.status in {"invalid_spec", "invalid_catalog"}:
        if workspace_root is None:
            try:
                workspace_root = resolve_workspace_root_for_solve_spec(spec_path)
            except (CassetteSpecError, FileNotFoundError):
                return None, report
        if solve_spec is None:
            run_root, notes = _best_effort_preflight_run_root(spec_path)
            if notes:
                report = report.model_copy(update={"notes": [*report.notes, *notes]})
        else:
            run_root = solve_spec.output.run_dir
        return _persist_preflight_failure(
            report=report,
            spec_path=spec_path,
            workspace_root=workspace_root,
            run_root=run_root,
            force_overwrite=force_overwrite,
        )

    assert solve_spec is not None
    assert resolved_spec_path is not None
    assert workspace_root is not None
    assert catalog is not None
    assert search_result is not None
    resolved_catalog_yaml = dump_nickase_catalog_yaml(catalog)
    cassette_solve_id = solve_id(
        spec_bytes=resolved_spec_path.read_bytes(),
        catalog_bytes=resolved_catalog_yaml.encode("utf-8"),
    )
    run_dir = build_solve_run_dir(
        workspace_root=workspace_root,
        run_root=solve_spec.output.run_dir,
        cassette_solve_id=cassette_solve_id,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(
                f"Cassette solve run directory already exists: {run_dir}. Use --force-overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    ensure_solve_run_dirs(run_dir)
    write_solve_inputs(run_dir, spec_path=resolved_spec_path, resolved_catalog_yaml=resolved_catalog_yaml)

    materialized_hit_runs: list[str] = []
    ranked_records = search_result.hits
    hit_records_by_id = {selected_hit.record.hit_id: selected_hit.record for selected_hit in ranked_records}
    hit_updates = []
    materialize_count = min(solve_spec.search.materialize_top_k, len(report.hits))
    top_hit_linear_rows: list[dict[str, object]] = []
    top_hit_hairpin_rows: list[dict[str, object]] = []
    publication_by_hit_id: dict[str, _HitPublicationBundle] = {}

    for hit in report.hits:
        hit_record = hit_records_by_id[hit.hit_id]
        publication = _build_hit_publication_bundle(
            hit=hit,
            hit_record=hit_record,
            cassette_solve_id=cassette_solve_id,
            resolved_catalog_yaml=resolved_catalog_yaml,
            emit_visual_contracts=solve_spec.output.emit_visual_contracts,
        )
        publication_by_hit_id[hit.hit_id] = publication
        if publication.linear_view is not None and publication.hairpin_view is not None:
            top_hit_linear_rows.append(publication.linear_view)
            top_hit_hairpin_rows.append(publication.hairpin_view)

    for hit in report.hits[:materialize_count]:
        hit_record = hit_records_by_id[hit.hit_id]
        publication = publication_by_hit_id[hit.hit_id]
        hit_dir = solve_hit_dir(run_dir, rank=hit.rank, hit_id=hit.hit_id)
        resolved_spec_payload = _materialized_hit_spec_payload(
            canonical_spec_payload=publication.canonical_spec_payload,
            run_dir=run_dir,
        )
        explicit_dir = hit_dir / "explicit"
        resolved_candidate_spec_path = explicit_dir / "resolved_candidate.cassette.yaml"
        resolved_report = hit_record.report.model_copy(
            update={
                "spec_path": str(resolved_candidate_spec_path.resolve()),
                "catalog_path": str(solve_resolved_catalog_path(run_dir).resolve()),
                "run_dir": str(explicit_dir.resolve()),
            }
        )
        write_solve_hit_bundle(
            hit_dir=explicit_dir,
            resolved_spec_payload=resolved_spec_payload,
            report=resolved_report,
            markdown=render_markdown_report(resolved_report),
        )
        materialized_hit_runs.append(str(hit_dir.resolve()))
        if solve_spec.output.emit_visual_contracts:
            manifest = build_views_manifest(
                solution_id=hit.solution_id,
                rank=hit.rank,
                include_jobs=solve_spec.output.emit_baserender_jobs,
            )
            write_view_bundle(
                hit_dir,
                linear_duplex=publication.linear_view,
                hairpin=publication.hairpin_view,
                manifest=manifest.model_dump(mode="json"),
            )
        manifest_ref = _relative_to_run_if_exists(run_dir, views_manifest_path(hit_dir))
        if solve_spec.output.emit_baserender_jobs:
            if "duplex_qa" in solve_spec.output.baserender_profiles:
                write_baserender_job(
                    linear_duplex_job_path(hit_dir),
                    build_single_view_job(
                        input_filename=linear_duplex_view_path(hit_dir).name,
                        adapter_kind="duplex_sequence_v1",
                        renderer="sequence_rows",
                        style_preset="cassette_duplex_qa",
                        output_filename="linear_duplex.pdf",
                    ),
                )
            if "hairpin_qa" in solve_spec.output.baserender_profiles:
                write_baserender_job(
                    hairpin_job_path(hit_dir),
                    build_single_view_job(
                        input_filename=hairpin_view_path(hit_dir).name,
                        adapter_kind="hairpin_topology_v1",
                        renderer="hairpin_cartoon",
                        style_preset="cassette_hairpin_qa",
                        output_filename="ssdna_hairpin.pdf",
                    ),
                )
        linear_job_ref = _relative_to_run_if_exists(run_dir, linear_duplex_job_path(hit_dir))
        hairpin_job_ref = _relative_to_run_if_exists(run_dir, hairpin_job_path(hit_dir))
        hit_updates.append(
            hit.model_copy(
                update={
                    "materialized_run_dir": str(hit_dir.resolve()),
                    "explicit_design_id": publication.explicit_design_id,
                    "views_manifest_path": manifest_ref,
                    "linear_duplex_job_path": linear_job_ref,
                    "ssdna_hairpin_job_path": hairpin_job_ref,
                }
            )
        )
    hit_updates.extend(report.hits[materialize_count:])

    if solve_spec.output.emit_visual_contracts:
        write_jsonl_records(top_hits_linear_duplex_jsonl_path(run_dir), top_hit_linear_rows)
        write_jsonl_records(top_hits_hairpin_jsonl_path(run_dir), top_hit_hairpin_rows)
    if solve_spec.output.emit_baserender_jobs:
        if "top_hits_duplex_qa" in solve_spec.output.baserender_profiles:
            write_baserender_job(
                top_hits_duplex_job_path(run_dir),
                build_top_hits_job(
                    input_filename=top_hits_linear_duplex_jsonl_path(run_dir).name,
                    adapter_kind="duplex_sequence_v1",
                    renderer="sequence_rows",
                    style_preset="cassette_duplex_contact_sheet",
                    output_filename="top_hits_duplex_qa_sheet.pdf",
                ),
            )
        if "top_hits_hairpin_qa" in solve_spec.output.baserender_profiles:
            write_baserender_job(
                top_hits_hairpin_job_path(run_dir),
                build_top_hits_job(
                    input_filename=top_hits_hairpin_jsonl_path(run_dir).name,
                    adapter_kind="hairpin_topology_v1",
                    renderer="hairpin_cartoon",
                    style_preset="cassette_hairpin_qa",
                    output_filename="top_hits_hairpin_qa_sheet.pdf",
                ),
            )

    report = report.model_copy(
        update={
            "solve_id": cassette_solve_id,
            "run_dir": str(run_dir.resolve()),
            "hits": hit_updates,
            "materialized_hit_runs": materialized_hit_runs,
            "metadata": report.metadata.model_copy(update={"materialized_hit_count": len(materialized_hit_runs)}),
        }
    )
    _write_solve_bundle(
        run_dir=run_dir,
        report=report,
        workspace_root=workspace_root,
        spec_path=resolved_spec_path,
        status_message=(
            f"cassette solve {report.status} (hits={len(report.hits)}, materialized={len(materialized_hit_runs)})"
        ),
    )
    return run_dir, report
