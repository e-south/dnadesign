"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/bundle.py

YIU v4 explicit bundle materialization and show helpers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from dnadesign.contracts.visual import SequenceEvidenceMapV1
from dnadesign.cruncher.app.yiu_workflow.report import _build_yiu_report
from dnadesign.cruncher.yiu.catalog import load_yiu_catalogs
from dnadesign.cruncher.yiu.load import load_yiu_spec
from dnadesign.cruncher.yiu.models import YiuProcessSpecV4, YiuStateRecord, YiuValidationReport


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    return path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _catalog_bytes(catalog_paths: list[Path]) -> bytes:
    if not catalog_paths:
        return b""
    return b"\n".join(path.read_bytes() for path in catalog_paths if path.exists())


def _annotation_rows(spec: YiuProcessSpecV4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for owner in spec.source_oligo.structural_owners:
        rows.append(
            {
                "category": "structural_owner",
                "id": owner.id,
                "start": owner.start,
                "end": owner.end,
            }
        )
    for tag in spec.source_oligo.effect_tags:
        rows.append(
            {
                "category": "effect_tag",
                "id": tag.id,
                "start": tag.start,
                "end": tag.end,
                "class": tag.class_,
            }
        )
    return rows


def _parts_rows(report: YiuValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in report.states:
        if state.primary_sequence:
            rows.append({"state_id": state.state_id, "row_id": "primary", "sequence": state.primary_sequence})
        if state.complement_sequence:
            rows.append({"state_id": state.state_id, "row_id": "complement", "sequence": state.complement_sequence})
    return rows


def _owner_rows(report: YiuValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in report.states:
        for annotation in state.annotations:
            if annotation.get("annotation_layer") != "structural_owner":
                continue
            rows.append(
                {
                    "state_id": state.state_id,
                    "row_id": annotation.get("row_id"),
                    "owner_id": annotation.get("id"),
                    "start": annotation.get("start"),
                    "end": annotation.get("end"),
                    "display_label": annotation.get("display_label"),
                    "short_label": annotation.get("short_label"),
                }
            )
    return rows


def _effect_rows(report: YiuValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in report.states:
        for annotation in state.annotations:
            if annotation.get("annotation_layer") != "effect_tag":
                continue
            rows.append(
                {
                    "state_id": state.state_id,
                    "row_id": annotation.get("row_id"),
                    "tag_id": annotation.get("id"),
                    "tag_kind": annotation.get("annotation_class"),
                    "start": annotation.get("start"),
                    "end": annotation.get("end"),
                    "display_label": annotation.get("display_label"),
                    "short_label": annotation.get("short_label"),
                }
            )
    return rows


def _fragment_rows(report: YiuValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in report.states:
        for fragment in state.fragments:
            rows.append(dict(fragment))
    return rows


def _inventory_render_status(views: list[dict[str, Any]]) -> str:
    if not views:
        return "not_requested"
    if all(bool(view.get("render_completed")) for view in views):
        return "rendered"
    return "not_requested"


def _contract_topology_kind(state: YiuStateRecord) -> str:
    return {
        "circular_dsdna_candidate": "circularized_linearized",
        "hairpin_ssdna": "hairpin_folded",
        "branched_y": "branched_adapter",
        "fragment_pool": "linear_ssdna",
    }.get(str(state.topology_kind or ""), str(state.topology_kind or "linear_ssdna"))


def _sequence_evidence_contract(state: YiuStateRecord) -> dict[str, Any]:
    owners = [
        {
            "owner_id": annotation["id"],
            "row_id": annotation["row_id"],
            "start": annotation["start"],
            "end": annotation["end"],
            "display_label": annotation.get("display_label") or annotation["id"],
            "short_label": annotation.get("short_label") or annotation["id"],
        }
        for annotation in state.annotations
        if annotation.get("annotation_layer") == "structural_owner"
    ]
    effect_tags = [
        {
            "tag_id": annotation["id"],
            "tag_kind": annotation["annotation_class"],
            "row_id": annotation["row_id"],
            "start": annotation["start"],
            "end": annotation["end"],
            "display_label": annotation.get("display_label") or annotation["annotation_class"],
            "short_label": annotation.get("short_label") or annotation["annotation_class"],
        }
        for annotation in state.annotations
        if annotation.get("annotation_layer") == "effect_tag"
    ]
    boundaries: list[dict[str, Any]] = []
    for junction in state.junctions:
        join_index = junction.get("join_index")
        if join_index is None:
            continue
        boundaries.append(
            {
                "boundary_id": junction.get("id") or "ligation_junction",
                "row_id": "primary",
                "boundary": int(join_index),
                "boundary_kind": "ligation_junction",
                "display_label": "Ligation junction",
                "short_label": "Lig",
            }
        )
    if state.state_id == "type_iis_cut_product_duplex" and state.primary_sequence:
        boundaries.extend(
            [
                {
                    "boundary_id": "left_cut",
                    "row_id": "primary",
                    "boundary": 0,
                    "boundary_kind": "cut",
                    "display_label": "Type IIS cut",
                    "short_label": "Cut",
                },
                {
                    "boundary_id": "right_cut",
                    "row_id": "primary",
                    "boundary": len(state.primary_sequence),
                    "boundary_kind": "cut",
                    "display_label": "Type IIS cut",
                    "short_label": "Cut",
                },
            ]
        )
    if state.state_id in {
        "post_fragment_cleanup",
        "snapback_adapter_complex",
        "ligated_ssdna_hairpin",
        "hairpin_pcr_linear_insert",
    }:
        boundaries.append(
            {
                "boundary_id": "nick_boundary",
                "row_id": "primary",
                "boundary": 33,
                "boundary_kind": "nick",
                "display_label": "Nick boundary",
                "short_label": "Nick",
            }
        )
    pairings: list[dict[str, Any]] = []
    return SequenceEvidenceMapV1.model_validate(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": state.state_id,
            "topology_kind": _contract_topology_kind(state),
            "alphabet": "iupac_dna" if state.sequence_mode != "concrete" else "dna",
            "primary_sequence": state.primary_sequence,
            "complement_sequence": state.complement_sequence,
            "owners": owners,
            "effect_tags": effect_tags,
            "boundaries": boundaries,
            "pairings": pairings,
            "display": {"title": state.state_id},
            "meta": {"state_kind": state.state_kind, **state.metadata},
        }
    ).model_dump(mode="json")


def _render_job_payload(*, view_relpath: str, render_relpath: str) -> dict[str, Any]:
    return {
        "version": 3,
        "results_root": ".",
        "input": {
            "kind": "json",
            "path": view_relpath,
            "adapter": {"kind": "sequence_evidence_map_v1"},
            "alphabet": "iupac_dna",
        },
        "render": {"renderer": "nucleotide_evidence_map", "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "path": render_relpath, "fmt": "pdf"}],
        "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
    }


def _publish_views(
    run_dir: Path, report: YiuValidationReport, *, persist_render_jobs_debug: bool = False
) -> dict[str, Any]:
    contracts_dir = run_dir / "contracts" / "visuals"
    jobs_dir = run_dir / "contracts" / "render_jobs"
    visuals_dir = run_dir / "visuals"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    if persist_render_jobs_debug:
        jobs_dir.mkdir(parents=True, exist_ok=True)
    views: list[dict[str, Any]] = []
    for state in report.states:
        contract = _sequence_evidence_contract(state)
        contract_path = contracts_dir / f"{state.state_id}.json"
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        render_job_path = None
        if persist_render_jobs_debug:
            render_job_path = jobs_dir / f"{state.state_id}.job.yaml"
            render_job_path.write_text(
                yaml.safe_dump(
                    _render_job_payload(
                        view_relpath=f"contracts/visuals/{state.state_id}.json",
                        render_relpath=f"visuals/{state.state_id}.pdf",
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
        views.append(
            {
                "state_id": state.state_id,
                "contract_kind": "sequence_evidence_map_v1",
                "view_contract_path": f"contracts/visuals/{state.state_id}.json",
                "render_artifact_path": f"visuals/{state.state_id}.pdf",
                "render_job_path": None
                if render_job_path is None
                else f"contracts/render_jobs/{state.state_id}.job.yaml",
                "renderer_kind": "nucleotide_evidence_map",
                "topology_kind": contract["topology_kind"],
                "render_requested": False,
                "render_completed": False,
                "last_rendered_at": None,
            }
        )
    inventory = {
        "bundle_kind": "explicit",
        "protocol_template": report.protocol_template,
        "renderer_kind": "nucleotide_evidence_map",
        "view_count": len(views),
        "render_count": 0,
        "render_status": "not_requested",
        "last_rendered_at": None,
        "views": views,
    }
    _write_json(run_dir / "visual_inventory.json", inventory)
    return inventory


def _run_id(spec_bytes: bytes, catalog_bytes: bytes) -> str:
    return sha256(spec_bytes + b"\n" + catalog_bytes).hexdigest()[:12]


def _write_explicit_bundle_from_report(
    run_dir: Path,
    *,
    spec: YiuProcessSpecV4,
    resolved_spec_path: Path,
    report: YiuValidationReport,
    catalog_paths: list[Path],
) -> tuple[YiuValidationReport, dict[str, Any] | None]:
    if run_dir.exists():
        for child in sorted(run_dir.glob("**/*"), reverse=True):
            if child.is_file():
                child.unlink()
        for child in sorted(run_dir.glob("**/*"), reverse=True):
            if child.is_dir():
                child.rmdir()
    run_dir.mkdir(parents=True, exist_ok=True)

    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    inventory = (
        _publish_views(run_dir, report, persist_render_jobs_debug=spec.output.persist_render_jobs_debug)
        if spec.output.emit_view_contracts
        else None
    )
    report = report.model_copy(
        update={
            "metadata": report.metadata.model_copy(
                update={"emitted_view_count": int(inventory["view_count"]) if inventory else 0}
            )
        }
    )

    _write_json(run_dir / "report.json", report.model_dump(mode="json"))
    _write_json(
        run_dir / "status.json",
        {
            "status": report.status,
            "schema_version": report.metadata.spec_schema_version,
            "protocol_template": report.protocol_template,
            "state_count": len(report.states),
            "explicit_final_state": report.states[-1].state_id if report.states else None,
        },
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "run_dir": str(run_dir.resolve()),
            "spec_path": str(resolved_spec_path.resolve()),
            "protocol_template": report.protocol_template,
            "state_trace_path": str((run_dir / "state_trace.jsonl").resolve()),
            "visual_inventory_path": str((run_dir / "visual_inventory.json").resolve()) if inventory else None,
        },
    )
    _write_jsonl(run_dir / "state_trace.jsonl", [state.model_dump(mode="json") for state in report.states])
    _write_csv(run_dir / "tables" / "state_sequences.csv", ["state_id", "row_id", "sequence"], _parts_rows(report))
    _write_csv(
        run_dir / "tables" / "state_owners.csv",
        ["state_id", "row_id", "owner_id", "start", "end", "display_label", "short_label"],
        _owner_rows(report),
    )
    _write_csv(
        run_dir / "tables" / "effect_tags.csv",
        ["state_id", "row_id", "tag_id", "tag_kind", "start", "end", "display_label", "short_label"],
        _effect_rows(report),
    )
    _write_csv(
        run_dir / "tables" / "fragment_summary.csv",
        [
            "state_id",
            "fragment_count",
            "max_fragment_nt",
            "fragment_lengths",
            "retained_product_sequence",
            "retained_owner_roster",
        ],
        _fragment_rows(report),
    )
    return report, inventory


def _materialize_yiu_bundle(spec_path: str | Path, *, force_overwrite: bool) -> tuple[Path, YiuValidationReport]:
    spec, resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    catalogs = load_yiu_catalogs(spec, workspace_root=workspace_root)
    report = _build_yiu_report(spec, catalogs=catalogs)
    spec_bytes = resolved_spec_path.read_bytes()
    catalog_bytes = _catalog_bytes(list(catalogs.paths))
    run_id = _run_id(spec_bytes, catalog_bytes)
    run_dir = workspace_root / spec.output.run_dir / spec.name / run_id
    if run_dir.exists() and not force_overwrite:
        raise ValueError(f"YIU run directory already exists: {run_dir}. Use --force-overwrite to replace it.")
    report, _inventory = _write_explicit_bundle_from_report(
        run_dir,
        spec=spec,
        resolved_spec_path=resolved_spec_path,
        report=report,
        catalog_paths=list(catalogs.paths),
    )
    return run_dir, report


def run_yiu_trace(spec_path: str | Path, *, force_overwrite: bool = False) -> tuple[Path, YiuValidationReport]:
    return _materialize_yiu_bundle(spec_path, force_overwrite=force_overwrite)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hard_invariant_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    states = report.get("states")
    if not isinstance(states, list) or not states:
        return None
    final_state = states[-1]
    if not isinstance(final_state, dict):
        return None
    metadata = final_state.get("metadata")
    if not isinstance(metadata, dict):
        return None
    invariants = metadata.get("hard_invariants")
    if not isinstance(invariants, list):
        return None
    statuses = [item.get("status") for item in invariants if isinstance(item, dict)]
    guaranteed = sum(status == "guaranteed" for status in statuses)
    impossible = sum(status == "impossible" for status in statuses)
    return {
        "state_id": final_state.get("state_id"),
        "total": len(invariants),
        "guaranteed": guaranteed,
        "impossible": impossible,
        "items": [
            {
                "id": item.get("id"),
                "class": item.get("class"),
                "status": item.get("status"),
            }
            for item in invariants
            if isinstance(item, dict)
        ],
    }


def _visual_render_summary(inventory_path: Path) -> dict[str, Any] | None:
    if not inventory_path.exists():
        return None
    payload = _read_json(inventory_path)
    return {
        "renderer_kind": payload.get("renderer_kind"),
        "view_count": payload.get("view_count"),
        "render_count": payload.get("render_count"),
        "render_status": payload.get("render_status"),
        "last_rendered_at": payload.get("last_rendered_at"),
    }


def yiu_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved = Path(run_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"YIU run directory not found: {resolved}")
    visual_inventory_path = resolved / "visual_inventory.json"
    visual_render_summary = _visual_render_summary(visual_inventory_path)
    if (resolved / "solve_report.json").exists():
        report = _read_json(resolved / "solve_report.json")
        status = _read_json(resolved / "solve_status.json")
        manifest = _read_json(resolved / "solve_manifest.json")
        selected_solution_path = report.get("selected_solution_path")
        selected_solution_dir = (
            Path(str(selected_solution_path)).expanduser().resolve() if selected_solution_path is not None else None
        )
        selected_hard_invariant_summary = None
        if selected_solution_dir is not None and (selected_solution_dir / "report.json").exists():
            selected_hard_invariant_summary = _hard_invariant_summary(_read_json(selected_solution_dir / "report.json"))
        return {
            "bundle_kind": "solve",
            "run_kind": "solve",
            "run_id": resolved.name,
            "run_dir": str(resolved),
            "protocol_template": status.get("protocol_template"),
            "canonical_template_id": status.get("protocol_template"),
            "schema_version": status.get("schema_version"),
            "solve_status": status.get("status"),
            "exhaustive_search": report.get("metadata", {}).get("exhaustive_search"),
            "satisfying_solution_count": report.get("satisfying_solution_count"),
            "comparison_solution_count": report.get("comparison_solution_count"),
            "selected_canonical_solution_path": selected_solution_path,
            "hard_invariant_summary": selected_hard_invariant_summary,
            "visual_inventory_path": str(visual_inventory_path.resolve()) if visual_inventory_path.exists() else None,
            "visual_render_summary": visual_render_summary,
            "key_artifact_paths": {
                "solve_report": str((resolved / "solve_report.json").resolve()),
                "solve_status": str((resolved / "solve_status.json").resolve()),
                "solve_manifest": str((resolved / "solve_manifest.json").resolve()),
                "selected_solution": str(selected_solution_dir) if selected_solution_dir is not None else None,
                "visual_inventory": str(visual_inventory_path.resolve()) if visual_inventory_path.exists() else None,
            },
            "manifest": manifest,
        }
    report = _read_json(resolved / "report.json")
    status = _read_json(resolved / "status.json")
    manifest = _read_json(resolved / "manifest.json")
    return {
        "bundle_kind": "explicit",
        "run_kind": "explicit",
        "run_id": resolved.name,
        "run_dir": str(resolved),
        "protocol_template": status.get("protocol_template"),
        "schema_version": status.get("schema_version"),
        "explicit_final_state": status.get("explicit_final_state"),
        "state_count": status.get("state_count"),
        "hard_invariant_summary": _hard_invariant_summary(report),
        "visual_inventory_path": str(visual_inventory_path.resolve()) if visual_inventory_path.exists() else None,
        "visual_render_summary": visual_render_summary,
        "key_artifact_paths": {
            "report": str((resolved / "report.json").resolve()),
            "status": str((resolved / "status.json").resolve()),
            "manifest": str((resolved / "manifest.json").resolve()),
            "state_trace": str((resolved / "state_trace.jsonl").resolve()),
            "visual_inventory": str(visual_inventory_path.resolve()) if visual_inventory_path.exists() else None,
        },
        "manifest": manifest,
        "report": report,
    }
