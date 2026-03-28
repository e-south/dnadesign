"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/bundle.py

YIU bundle materialization, publication, and show payload helpers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from dnadesign.contracts.visual import YiuHairpinTopologyV1, YiuLinearStateV1, YiuTopologyCartoonV1
from dnadesign.cruncher.app.yiu_workflow.helpers import _annotation_collections, _item_end, _state_topology
from dnadesign.cruncher.app.yiu_workflow.report import _build_yiu_report
from dnadesign.cruncher.yiu.artifacts import (
    STATE_VIEW_SCHEMA_VERSION,
    annotations_path,
    baserender_jobs_dir,
    build_run_dir,
    catalog_fingerprint,
    design_id,
    fragments_path,
    input_fingerprint,
    parts_path,
    prepare_run_dir,
    published_views_dir,
    renders_dir,
    report_path,
    resolve_code_revision,
    state_view_path,
    status_path,
    trace_path,
    visual_manifest_path,
    write_csv,
    write_manifest,
    write_report,
    write_status,
    write_trace,
    write_trace_manifest,
)
from dnadesign.cruncher.yiu.catalog import load_yiu_catalogs
from dnadesign.cruncher.yiu.load import load_yiu_spec
from dnadesign.cruncher.yiu.models import YiuProcessSpec, YiuStateRecord, YiuValidationReport


def _catalog_bytes(catalog_paths: list[Path]) -> bytes:
    if not catalog_paths:
        return b""
    return b"\n".join(path.read_bytes() for path in catalog_paths if path.exists())


def _annotation_rows(spec: YiuProcessSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, collection in _annotation_collections(spec):
        for item in collection:
            rows.append(
                {
                    "category": category,
                    "id": item.id,
                    "start": item.start,
                    "end": _item_end(item),
                    "label": getattr(item, "enzyme", item.id),
                }
            )
    return rows


def _parts_rows(report: YiuValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in report.states:
        if state.primary_sequence:
            rows.append(
                {
                    "state_id": state.state_id,
                    "part_id": f"{state.state_id}_primary",
                    "role": state.kind,
                    "sequence": state.primary_sequence,
                }
            )
        if state.complement_sequence:
            rows.append(
                {
                    "state_id": state.state_id,
                    "part_id": f"{state.state_id}_complement",
                    "role": f"{state.kind}_complement",
                    "sequence": state.complement_sequence,
                }
            )
    return rows


def _fragment_rows(report: YiuValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in report.states:
        for index, length in enumerate(state.metadata.get("fragment_lengths", []), start=1):
            rows.append({"state_id": state.state_id, "fragment_id": f"{state.state_id}_{index}", "length_nt": length})
    return rows


def _split_view_contract_payload(state: YiuStateRecord) -> dict[str, Any]:
    topology_kind = state.topology_kind or _state_topology(state)
    alphabet = "dna" if state.sequence_mode == "concrete" else "iupac_dna"
    if state.state_id == "ligated_ssdna_hairpin":
        sequence = state.primary_sequence or ""
        paired_nt = int(state.metadata.get("paired_nt") or max(1, min(len(sequence) // 4, len(sequence) // 2)))
        loop_start = paired_nt
        loop_end = max(loop_start, len(sequence) - paired_nt)
        pair_map = [
            {"left_index": index, "right_index": len(sequence) - 1 - index}
            for index in range(max(0, min(paired_nt, len(sequence) // 2)))
        ]
        return YiuHairpinTopologyV1.model_validate(
            {
                "contract_kind": "yiu_hairpin_topology_v1",
                "state_id": state.state_id,
                "topology_kind": "ssdna_hairpin",
                "sequence": sequence,
                "stem_left_span": {"start": 0, "end": paired_nt},
                "stem_right_span": {"start": max(0, len(sequence) - paired_nt), "end": len(sequence)},
                "loop_span": {"start": loop_start, "end": loop_end},
                "pair_map": pair_map,
                "adapter_branches": [],
                "annotations": state.annotations,
                "display": {"title": state.state_id},
                "meta": {"evidence_mode": state.validation_mode, **state.metadata},
            }
        ).model_dump(mode="json")
    if topology_kind in {"circular_dsdna_candidate", "branched_y", "fragment_pool"}:
        subkind = {
            "circular_dsdna_candidate": "circular_duplex",
            "branched_y": "branched_y",
            "fragment_pool": "composite_retained_product",
        }.get(topology_kind, "composite_retained_product")
        return YiuTopologyCartoonV1.model_validate(
            {
                "contract_kind": "yiu_topology_cartoon_v1",
                "state_id": state.state_id,
                "topology_kind": subkind,
                "sequence": state.primary_sequence,
                "segments": state.segments,
                "annotations": state.annotations,
                "cuts": state.cuts,
                "junctions": state.junctions,
                "fragments": state.fragments,
                "display": {"title": state.state_id},
                "meta": {"evidence_mode": state.validation_mode, **state.metadata},
            }
        ).model_dump(mode="json")
    return YiuLinearStateV1.model_validate(
        {
            "contract_kind": "yiu_linear_state_v1",
            "state_id": state.state_id,
            "topology_kind": topology_kind,
            "alphabet": alphabet,
            "primary_sequence": state.primary_sequence,
            "complement_sequence": state.complement_sequence,
            "segments": state.segments,
            "annotations": state.annotations,
            "cuts": state.cuts,
            "junctions": state.junctions,
            "fragments": state.fragments,
            "display": {"title": state.state_id},
            "meta": {"evidence_mode": state.validation_mode, **state.metadata},
        }
    ).model_dump(mode="json")


def _split_view_job_payload(state: YiuStateRecord) -> dict[str, Any]:
    contract = _split_view_contract_payload(state)
    contract_kind = str(contract["contract_kind"])
    adapter_kind = {
        "yiu_linear_state_v1": "yiu_linear_state_v1",
        "yiu_hairpin_topology_v1": "yiu_hairpin_topology_v1",
        "yiu_topology_cartoon_v1": "yiu_topology_cartoon_v1",
    }[contract_kind]
    renderer_name = {
        "yiu_linear_state_v1": "sequence_rows",
        "yiu_hairpin_topology_v1": "hairpin_cartoon",
        "yiu_topology_cartoon_v1": "topology_cartoon",
    }[contract_kind]
    alphabet = str(contract.get("alphabet") or "dna").upper()
    return {
        "version": 3,
        "results_root": "..",
        "input": {
            "kind": "json",
            "path": f"../views/{state.state_id}.json",
            "adapter": {"kind": adapter_kind},
            "alphabet": "IUPAC_DNA" if alphabet == "IUPAC_DNA" else "DNA",
        },
        "render": {"renderer": renderer_name, "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "path": f"../renders/{state.state_id}.pdf", "fmt": "pdf"}],
        "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
    }


def _publish_split_visuals(run_dir: Path, report: YiuValidationReport, *, emit_baserender_jobs: bool) -> None:
    views_dir = published_views_dir(run_dir)
    jobs_dir = run_dir / "published" / "baserender_jobs"
    renders_dir = run_dir / "published" / "renders"
    views_dir.mkdir(parents=True, exist_ok=True)
    if emit_baserender_jobs:
        jobs_dir.mkdir(parents=True, exist_ok=True)
        renders_dir.mkdir(parents=True, exist_ok=True)
    manifest_views: list[dict[str, Any]] = []
    for state in report.states:
        payload = _split_view_contract_payload(state)
        view_path = state_view_path(run_dir, state.state_id)
        view_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        job_relpath = None
        if emit_baserender_jobs:
            job_path = jobs_dir / f"{state.state_id}.job.yaml"
            job_path.write_text(yaml.safe_dump(_split_view_job_payload(state), sort_keys=False), encoding="utf-8")
            job_relpath = f"published/baserender_jobs/{state.state_id}.job.yaml"
        manifest_views.append(
            {
                "state_id": state.state_id,
                "contract_kind": payload["contract_kind"],
                "path": f"published/views/{state.state_id}.json",
                "job_path": job_relpath,
            }
        )
    visual_manifest = {
        "contract_version": 3,
        "family": "yiu",
        "workflow": "yiu_explicit",
        "protocol": report.protocol,
        "protocol_template": report.protocol_template,
        "template_alias_used": report.template_alias_used,
        "template_alias_status": report.template_alias_status,
        "view_count": len(manifest_views),
        "job_count": sum(1 for view in manifest_views if view["job_path"] is not None),
        "render_count": sum(1 for path in renders_dir.glob("*") if path.is_file()) if renders_dir.exists() else 0,
        "views": manifest_views,
    }
    (run_dir / "published" / "visual_manifest.json").write_text(
        json.dumps(visual_manifest, indent=2),
        encoding="utf-8",
    )


def _publish_views(run_dir: Path, report: YiuValidationReport, *, emit_baserender_jobs: bool = False) -> None:
    published_views_dir(run_dir).mkdir(parents=True, exist_ok=True)
    if report.protocol_template == "yiu_circularized_payload_v1":
        _publish_split_visuals(run_dir, report, emit_baserender_jobs=emit_baserender_jobs)
        return
    view_contract_version = report.metadata.view_contract_version or STATE_VIEW_SCHEMA_VERSION
    manifest_views: list[dict[str, Any]] = []
    for state in report.states:
        payload = {
            "schema_version": view_contract_version,
            "view_contract_version": view_contract_version,
            "family": "yiu",
            "workflow": "yiu",
            "protocol": report.protocol,
            "protocol_template": report.protocol_template,
            "state_id": state.state_id,
            "state_kind": state.state_kind or state.kind,
            "kind": state.kind,
            "status": state.status,
            "molecule_topology": _state_topology(state),
            "topology_kind": state.topology_kind or _state_topology(state),
            "sequence_mode": state.sequence_mode,
            "validation_mode": state.validation_mode,
            "primary_sequence": state.primary_sequence,
            "complement_sequence": state.complement_sequence,
            "segments": state.segments,
            "annotations": state.annotations,
            "cuts": state.cuts,
            "junctions": state.junctions,
            "fragments": state.fragments,
            "meta": state.metadata,
        }
        state_view_path(run_dir, state.state_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifest_views.append(
            {
                "state_id": state.state_id,
                "path": f"published/views/{state.state_id}.json",
                "contract_kind": "yiu_state_view_v2" if view_contract_version >= 2 else "yiu_state_view_v1",
            }
        )
    (run_dir / "published" / "visual_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": view_contract_version,
                "family": "yiu",
                "workflow": "yiu_explicit",
                "protocol": report.protocol,
                "protocol_template": report.protocol_template,
                "template_alias_used": report.template_alias_used,
                "template_alias_status": report.template_alias_status,
                "view_count": len(manifest_views),
                "job_count": 0,
                "render_count": 0,
                "views": manifest_views,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _materialize_yiu_bundle(spec_path: str | Path, *, force_overwrite: bool) -> tuple[Path, YiuValidationReport]:
    spec, resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    catalogs = load_yiu_catalogs(spec, workspace_root=workspace_root)
    report = _build_yiu_report(spec, catalogs=catalogs)
    catalog_paths = list(catalogs.paths)
    spec_bytes = resolved_spec_path.read_bytes()
    catalog_bytes = _catalog_bytes(catalog_paths)
    run_id = design_id(spec_bytes=spec_bytes, catalog_bytes=catalog_bytes)
    input_fingerprint_value = input_fingerprint(spec_bytes=spec_bytes, catalog_bytes=catalog_bytes)
    catalog_fingerprint_value = catalog_fingerprint(catalog_bytes=catalog_bytes)
    code_revision = resolve_code_revision(workspace_root)
    run_dir = build_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
        spec_name=spec.name,
        run_id=run_id,
    )
    prepare_run_dir(
        run_dir,
        force_overwrite=force_overwrite,
        emit_view_contracts=spec.output.emit_view_contracts,
        emit_baserender_jobs=getattr(spec.output, "emit_baserender_jobs", False),
    )
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_report(run_dir, report)
    write_status(
        run_dir,
        report,
        input_fingerprint_value=input_fingerprint_value,
        catalog_fingerprint_value=catalog_fingerprint_value,
        code_revision=code_revision,
    )
    write_trace(run_dir, report.states)
    write_trace_manifest(run_dir, report)
    write_csv(
        parts_path(run_dir),
        fieldnames=["state_id", "part_id", "role", "sequence"],
        rows=_parts_rows(report),
    )
    write_csv(
        annotations_path(run_dir),
        fieldnames=["category", "id", "start", "end", "label"],
        rows=_annotation_rows(spec),
    )
    write_csv(
        fragments_path(run_dir),
        fieldnames=["state_id", "fragment_id", "length_nt"],
        rows=_fragment_rows(report),
    )
    if spec.output.emit_view_contracts:
        _publish_views(run_dir, report, emit_baserender_jobs=getattr(spec.output, "emit_baserender_jobs", False))
    write_manifest(
        run_dir,
        workspace_root=workspace_root,
        spec_path=resolved_spec_path,
        report=report,
        input_fingerprint_value=input_fingerprint_value,
        catalog_fingerprint_value=catalog_fingerprint_value,
        code_revision=code_revision,
        catalog_paths=catalog_paths,
    )
    return run_dir, report


def run_yiu_design(spec_path: str | Path, *, force_overwrite: bool = False) -> tuple[Path, YiuValidationReport]:
    return _materialize_yiu_bundle(spec_path, force_overwrite=force_overwrite)


def run_yiu_trace(spec_path: str | Path, *, force_overwrite: bool = False) -> tuple[Path, YiuValidationReport]:
    return _materialize_yiu_bundle(spec_path, force_overwrite=force_overwrite)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_path_if_exists(path: Path) -> str | None:
    return str(path.resolve()) if path.exists() else None


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for candidate in path.glob(pattern) if candidate.is_file())


def _top_hit_final_state_kind(hit_bundle_paths: list[str]) -> str | None:
    for hit_bundle_path in hit_bundle_paths:
        payload = _read_json_if_exists(Path(hit_bundle_path) / "yiu_report.json")
        if payload is None:
            continue
        states = payload.get("states")
        if isinstance(states, list) and states:
            final_state = states[-1]
            if isinstance(final_state, dict):
                return str(final_state.get("state_kind") or final_state.get("kind") or "")
    return None


def yiu_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved = Path(run_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"YIU run directory not found: {resolved}")
    visual_manifest = visual_manifest_path(resolved)
    published_views = resolved / "published" / "views"
    published_jobs = baserender_jobs_dir(resolved)
    published_renders = renders_dir(resolved)
    visual_manifest_payload = _read_json_if_exists(visual_manifest) or {}
    view_count = int(visual_manifest_payload.get("view_count") or _count_files(published_views, "*.json"))
    job_count = int(visual_manifest_payload.get("job_count") or _count_files(published_jobs, "*.job.yaml"))
    render_count = _count_files(published_renders, "*")
    if (resolved / "yiu_solve_manifest.json").exists():
        report = json.loads((resolved / "yiu_solve_report.json").read_text(encoding="utf-8"))
        status = json.loads((resolved / "yiu_solve_status.json").read_text(encoding="utf-8"))
        manifest = json.loads((resolved / "yiu_solve_manifest.json").read_text(encoding="utf-8"))
        first_hit_path = status.get("first_hit_path")
        top_hit_bundle_paths = [
            str(Path(path).resolve())
            for path in (
                status.get("top_hit_bundle_paths")
                or [
                    hit.get("materialized_run_dir") for hit in report.get("hits", []) if hit.get("materialized_run_dir")
                ]
            )
            if path
        ]
        paths = {
            "manifest": str((resolved / "yiu_solve_manifest.json").resolve()),
            "status": str((resolved / "yiu_solve_status.json").resolve()),
            "report": str((resolved / "yiu_solve_report.json").resolve()),
            "visual_manifest": _resolved_path_if_exists(visual_manifest),
            "published_views_dir": _resolved_path_if_exists(published_views),
            "published_jobs_dir": _resolved_path_if_exists(published_jobs),
            "published_renders_dir": _resolved_path_if_exists(published_renders),
            "accepted_hits": str((resolved / "accepted_hits.jsonl").resolve()),
            "hits_csv": str((resolved / "hits.csv").resolve()),
            "hits_root": _resolved_path_if_exists(resolved / "hits"),
            "first_hit": first_hit_path if first_hit_path and Path(first_hit_path).exists() else None,
        }
        return {
            "bundle_kind": "solve",
            "run_id": resolved.name,
            "spec_name": Path(report["spec_path"]).name,
            "run_dir": str(resolved),
            "status": status["status"],
            "status_message": f"{status['status']} (hits={status.get('hit_count', 0)})",
            "protocol": None,
            "protocol_template": None,
            "template_alias_used": None,
            "template_alias_status": None,
            "canonical_template_id": None,
            "view_contract_version": visual_manifest_payload.get("contract_version"),
            "step_count": None,
            "state_count": None,
            "issue_count": len(report.get("issues", [])),
            "emitted_view_count": view_count,
            "emitted_job_count": job_count,
            "emitted_render_count": render_count,
            "manifest_path": paths["manifest"],
            "status_path": paths["status"],
            "report_path": paths["report"],
            "trace_path": None,
            "trace_manifest_path": None,
            "published_views_manifest_path": None,
            "published_views_dir": paths["published_views_dir"],
            "visual_manifest_path": paths["visual_manifest"],
            "published_jobs_dir": paths["published_jobs_dir"],
            "published_renders_dir": paths["published_renders_dir"],
            "first_hit_path": paths["first_hit"],
            "accepted_hits_path": paths["accepted_hits"],
            "solve_id": report.get("solve_id"),
            "top_hit_ids": manifest.get("top_hit_ids", []),
            "top_hit_bundle_paths": top_hit_bundle_paths,
            "accepted_candidate_count": report.get("metadata", {}).get("accepted_candidate_count"),
            "returned_hit_count": report.get("metadata", {}).get("returned_hit_count"),
            "materialized_hit_count": report.get("metadata", {}).get("materialized_hit_count"),
            "search_node_count": report.get("metadata", {}).get("search_node_count"),
            "enumerated_candidate_count": report.get("metadata", {}).get("enumerated_candidate_count"),
            "warning_codes": report.get("metadata", {}).get("warning_codes", []),
            "warnings": report.get("metadata", {}).get("warnings", []),
            "search_truncated": report.get("metadata", {}).get("search_truncated"),
            "accepted_pool_truncated": report.get("metadata", {}).get("accepted_pool_truncated"),
            "final_state_kind": _top_hit_final_state_kind(top_hit_bundle_paths),
            "paths": paths,
            "visual_manifest": visual_manifest_payload or None,
            "manifest": manifest,
            "status_payload": status,
            "report_metadata": report.get("metadata", {}),
        }
    manifest = report = status = None
    if report_path(resolved).exists():
        report = json.loads(report_path(resolved).read_text(encoding="utf-8"))
    if status_path(resolved).exists():
        status = json.loads(status_path(resolved).read_text(encoding="utf-8"))
    if (resolved / "yiu_manifest.json").exists():
        manifest = json.loads((resolved / "yiu_manifest.json").read_text(encoding="utf-8"))
    if report is None or status is None or manifest is None:
        raise ValueError(f"Run directory does not contain a complete YIU bundle: {resolved}")
    paths = {
        "manifest": str((resolved / "yiu_manifest.json").resolve()),
        "status": str(status_path(resolved).resolve()),
        "report": str(report_path(resolved).resolve()),
        "trace": _resolved_path_if_exists(trace_path(resolved)),
        "trace_manifest": _resolved_path_if_exists(resolved / "yiu_trace_manifest.json"),
        "visual_manifest": _resolved_path_if_exists(visual_manifest),
        "published_views_dir": _resolved_path_if_exists(published_views_dir(resolved)),
        "published_jobs_dir": _resolved_path_if_exists(published_jobs),
        "published_renders_dir": _resolved_path_if_exists(published_renders),
    }
    return {
        "bundle_kind": "explicit",
        "run_id": resolved.name,
        "spec_name": report["spec_name"],
        "run_dir": str(resolved),
        "status": status["status"],
        "status_message": status["status_message"],
        "protocol": status.get("protocol"),
        "protocol_template": status.get("protocol_template"),
        "template_alias_used": status.get("template_alias_used"),
        "template_alias_status": status.get("template_alias_status"),
        "canonical_template_id": status.get("protocol_template") or status.get("protocol"),
        "view_contract_version": status.get("view_contract_version"),
        "step_count": report.get("metadata", {}).get("step_count"),
        "state_count": report.get("metadata", {}).get("state_count", len(report.get("states", []))),
        "issue_count": len(report.get("issues", [])),
        "emitted_view_count": view_count,
        "emitted_job_count": job_count,
        "emitted_render_count": render_count,
        "manifest_path": paths["manifest"],
        "status_path": paths["status"],
        "report_path": paths["report"],
        "trace_path": paths["trace"],
        "trace_manifest_path": paths["trace_manifest"],
        "published_views_manifest_path": None,
        "published_views_dir": paths["published_views_dir"],
        "visual_manifest_path": paths["visual_manifest"],
        "published_jobs_dir": paths["published_jobs_dir"],
        "published_renders_dir": paths["published_renders_dir"],
        "first_hit_path": None,
        "paths": paths,
        "visual_manifest": visual_manifest_payload or None,
        "manifest": manifest,
        "status_payload": status,
        "report_metadata": report.get("metadata", {}),
    }
