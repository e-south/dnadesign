"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_workflow.py

Application orchestration for v2 explicit snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from dnadesign.cruncher.app.snapback_publish import (
    build_publication_bundle,
    write_publication_bundle,
)
from dnadesign.cruncher.nickases.catalog import dump_nickase_catalog_yaml, load_merged_nickase_catalog
from dnadesign.cruncher.nickases.errors import NickaseCatalogError
from dnadesign.cruncher.snapback.artifacts import (
    baserender_jobs_dir,
    build_manifest,
    build_run_dir,
    candidate_table_path,
    catalog_snapshot_path,
    design_id,
    ensure_run_dirs,
    load_manifest,
    load_solve_manifest,
    load_solve_status,
    load_status,
    materialized_hits_dir,
    post_nick_exposed_view_path,
    post_nick_exposed_visual_contract_path,
    post_nick_foldback_view_path,
    post_nick_foldback_visual_contract_path,
    pre_nick_duplex_view_path,
    pre_nick_duplex_visual_contract_path,
    renders_dir,
    report_json_path,
    report_md_path,
    snapback_manifest_path,
    snapback_status_path,
    snapback_triptych_job_path,
    snapback_triptych_render_path,
    snapback_triptych_visual_contracts_path,
    snapshot_explicit_inputs,
    solve_frontier_table_path,
    solve_hits_table_path,
    solve_input_spec_path,
    solve_manifest_path,
    solve_report_json_path,
    solve_report_md_path,
    solve_resolved_catalog_path,
    solve_status_path,
    spec_snapshot_path,
    views_manifest_path,
    write_candidate_table,
    write_manifest,
    write_report,
    write_status,
)
from dnadesign.cruncher.snapback.catalog_sources import catalog_source_label
from dnadesign.cruncher.snapback.load import load_snapback_spec
from dnadesign.cruncher.snapback.planner import (
    build_invalid_catalog_report,
    build_snapback_report,
    render_markdown_report,
)


def _resolve_catalog(spec, *, workspace_root: Path):
    catalog, resolved_paths = load_merged_nickase_catalog(
        preset_id=spec.design.nickase.catalog.preset,
        additional_preset_ids=spec.design.nickase.catalog.additional_presets,
        additional_paths=spec.design.nickase.catalog.additional_paths,
        workspace_root=workspace_root,
    )
    return (
        catalog,
        resolved_paths,
        catalog_source_label(
            preset_ids=spec.design.nickase.catalog.resolved_preset_ids(),
            resolved_paths=resolved_paths,
        ),
    )


def _existing_triptych_render(run_dir: Path) -> Path | None:
    for fmt in ("png", "svg", "pdf"):
        candidate = snapback_triptych_render_path(run_dir, fmt=fmt)
        if candidate.exists():
            return candidate
    return None


def _required_existing_manifest_path(payload: dict[str, object], *, field: str, label: str) -> Path:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} drift detected.")
    resolved = Path(value).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} missing: {resolved}")
    return resolved


def _validate_explicit_publication_alignment(run_dir: Path, *, candidate: dict[str, object]) -> None:
    manifest_payload = json.loads(views_manifest_path(run_dir).read_text(encoding="utf-8"))
    solution_id = manifest_payload.get("solution_id")
    if not isinstance(solution_id, str) or not solution_id:
        raise ValueError("Snapback views manifest solution_id drift detected.")

    expected_state_ids = [
        f"{solution_id}.pre_nick_duplex",
        f"{solution_id}.post_nick_exposed",
        f"{solution_id}.post_nick_foldback",
    ]
    pre_visual = json.loads(pre_nick_duplex_visual_contract_path(run_dir).read_text(encoding="utf-8"))
    exposed_visual = json.loads(post_nick_exposed_visual_contract_path(run_dir).read_text(encoding="utf-8"))
    foldback_visual = json.loads(post_nick_foldback_visual_contract_path(run_dir).read_text(encoding="utf-8"))
    for expected_state_id, visual_payload in zip(
        expected_state_ids,
        [pre_visual, exposed_visual, foldback_visual],
        strict=True,
    ):
        if visual_payload.get("state_id") != expected_state_id:
            raise ValueError(f"Snapback visual state_id drift detected for {expected_state_id}.")

    designed_sequence = candidate.get("designed_sequence")
    post_nick_sequence = candidate.get("post_nick_sequence")
    if pre_visual.get("primary_sequence") != designed_sequence:
        raise ValueError("Snapback pre-nick visual primary_sequence drift detected.")
    if exposed_visual.get("primary_sequence") != designed_sequence:
        raise ValueError("Snapback exposed visual primary_sequence drift detected.")
    if foldback_visual.get("primary_sequence") != post_nick_sequence:
        raise ValueError("Snapback foldback visual primary_sequence drift detected.")
    if foldback_visual.get("meta", {}).get("cap_extension_nt") != candidate.get("cap_extension_nt"):
        raise ValueError("Snapback foldback visual cap_extension_nt drift detected.")
    if foldback_visual.get("meta", {}).get("terminal_ligatable_duplex_bp") != candidate.get(
        "terminal_ligatable_duplex_bp"
    ):
        raise ValueError("Snapback foldback visual terminal_ligatable_duplex_bp drift detected.")

    triptych_lines = [
        json.loads(line)
        for line in snapback_triptych_visual_contracts_path(run_dir).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(triptych_lines) != 3:
        raise ValueError("Snapback triptych visual contract count drift detected.")
    if [payload.get("state_id") for payload in triptych_lines] != expected_state_ids:
        raise ValueError("Snapback triptych visual state ordering drift detected.")


def _validate_materialized_hit_alignment(*, hit: dict[str, object], hit_run_dir: Path) -> None:
    report_payload = json.loads(report_json_path(hit_run_dir).read_text(encoding="utf-8"))
    candidate = report_payload.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"Materialized snapback hit bundle missing candidate payload: {hit_run_dir}")
    intended_nick = candidate.get("intended_nick")
    if not isinstance(intended_nick, dict):
        raise ValueError(f"Materialized snapback hit bundle missing intended_nick payload: {hit_run_dir}")
    if intended_nick.get("variant_id") != hit.get("variant_id"):
        raise ValueError(f"Materialized snapback hit variant drift detected: {hit_run_dir}")
    for candidate_key, hit_key in (
        ("cap_sequence", "cap_sequence"),
        ("foldback_arm", "foldback_arm"),
        ("nick_boundary", "nick_boundary"),
        ("paired_bp", "paired_bp"),
        ("cap_extension_nt", "cap_extension_nt"),
        ("site_mutation_count", "site_mutation_count"),
    ):
        if candidate.get(candidate_key) != hit.get(hit_key):
            raise ValueError(f"Materialized snapback hit {candidate_key} drift detected: {hit_run_dir}")


def validate_snapback_spec(path: str | Path):
    spec, spec_path, workspace_root = load_snapback_spec(path)
    catalog_source = catalog_source_label(
        preset_ids=spec.design.nickase.catalog.resolved_preset_ids(),
        resolved_paths=spec.design.nickase.catalog.additional_paths,
    )
    try:
        catalog, _resolved_paths, catalog_source = _resolve_catalog(spec, workspace_root=workspace_root)
    except (FileNotFoundError, NickaseCatalogError) as exc:
        return build_invalid_catalog_report(
            spec,
            spec_path=spec_path,
            workspace_root=workspace_root,
            catalog_source=catalog_source,
            code="CATALOG_LOAD_FAILED",
            message=str(exc),
        )
    return build_snapback_report(
        spec,
        spec_path=spec_path,
        workspace_root=workspace_root,
        catalog=catalog,
        catalog_source=catalog_source,
    )


def run_snapback_design(path: str | Path, *, force_overwrite: bool = False):
    spec, spec_path, workspace_root = load_snapback_spec(path)
    catalog, _resolved_paths, catalog_source = _resolve_catalog(spec, workspace_root=workspace_root)
    report = build_snapback_report(
        spec, spec_path=spec_path, workspace_root=workspace_root, catalog=catalog, catalog_source=catalog_source
    )
    catalog_yaml = dump_nickase_catalog_yaml(catalog)
    spec_bytes = spec_path.read_bytes()
    catalog_bytes = catalog_yaml.encode("utf-8")
    snapback_design_id = design_id(spec_bytes=spec_bytes, catalog_bytes=catalog_bytes)
    run_dir = build_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
        spec_name=spec.name,
        snapback_design_id=snapback_design_id,
    )
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(f"Snapback run directory already exists: {run_dir}. Use --force-overwrite to replace it.")
        shutil.rmtree(run_dir)
    ensure_run_dirs(
        run_dir,
        include_visual_contracts=spec.output.emit_visual_contracts,
        include_baserender_jobs=spec.output.emit_baserender_jobs,
    )
    snapshot_explicit_inputs(run_dir, spec_path=spec_path, catalog_yaml=catalog_yaml)
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_report(run_dir, report, markdown=render_markdown_report(report))
    write_candidate_table(run_dir, report)
    if spec.output.emit_visual_contracts and report.candidate is not None:
        write_publication_bundle(
            run_dir,
            bundle=build_publication_bundle(
                report=report,
                solution_id=snapback_design_id,
                include_jobs=spec.output.emit_baserender_jobs,
                render_format=spec.output.render_format,
            ),
        )
    manifest = build_manifest(
        run_dir=run_dir,
        workspace_root=workspace_root,
        spec_path=spec_path,
        report=report,
    )
    write_manifest(run_dir, manifest)
    write_status(run_dir, report=report)
    return run_dir, report


def _explicit_show_payload(run_dir: Path) -> dict[str, object]:
    manifest = load_manifest(run_dir)
    status = load_status(run_dir)
    expected_run_dir = str(run_dir)
    workspace_root = _required_existing_manifest_path(
        manifest,
        field="workspace_root",
        label="Snapback explicit manifest workspace_root",
    )
    expected_workspace_root = str(workspace_root)
    if manifest.get("kind") != "explicit":
        raise ValueError("Snapback explicit manifest kind drift detected.")
    if manifest.get("workflow") != "snapback_design":
        raise ValueError("Snapback explicit manifest workflow drift detected.")
    if manifest.get("contract") != "single_nick_snapback_v2":
        raise ValueError("Unsupported snapback explicit contract version.")
    if status.get("workflow") != "snapback_design":
        raise ValueError("Snapback explicit status workflow drift detected.")
    if status.get("contract") != "single_nick_snapback_v2":
        raise ValueError("Unsupported snapback explicit status contract version.")
    if manifest.get("stage") != "snapback" or status.get("stage") != "snapback":
        raise ValueError("Snapback explicit stage drift detected.")
    if manifest.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback manifest run_dir drift detected.")
    if status.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback status run_dir drift detected.")
    if manifest.get("spec_name") != status.get("spec_name"):
        raise ValueError("Snapback manifest/status spec_name drift detected.")
    if manifest.get("status") != status.get("status"):
        raise ValueError("Snapback manifest/status status drift detected.")
    required_paths = [
        report_json_path(run_dir),
        report_md_path(run_dir),
        spec_snapshot_path(run_dir),
        catalog_snapshot_path(run_dir),
        candidate_table_path(run_dir),
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required snapback artifact missing: {path}")
    declared_artifacts = {item["name"]: item["path"] for item in manifest.get("artifacts", [])}
    for key in ("pre_nick_duplex_view", "post_nick_exposed_view", "post_nick_foldback_view", "views_manifest"):
        if key in declared_artifacts:
            candidate = run_dir / declared_artifacts[key]
            if not candidate.exists():
                raise FileNotFoundError(f"Declared snapback visual artifact missing: {candidate}")
    for key in (
        "pre_nick_duplex_visual_contract",
        "post_nick_exposed_visual_contract",
        "post_nick_foldback_visual_contract",
        "snapback_triptych_visual_contracts",
        "snapback_triptych_job",
    ):
        if key in declared_artifacts:
            candidate = run_dir / declared_artifacts[key]
            if not candidate.exists():
                raise FileNotFoundError(f"Declared snapback visual artifact missing: {candidate}")
    report_payload = json.loads(report_json_path(run_dir).read_text(encoding="utf-8"))
    if report_payload.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback report run_dir drift detected.")
    if report_payload.get("workspace_root") != expected_workspace_root:
        raise ValueError("Snapback report workspace_root drift detected.")
    if views_manifest_path(run_dir).exists():
        candidate = report_payload.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("Snapback visual artifacts require a satisfied candidate payload.")
        _validate_explicit_publication_alignment(run_dir, candidate=candidate)
    triptych_render = _existing_triptych_render(run_dir)
    return {
        "kind": "explicit",
        "run_dir": str(run_dir),
        "spec_name": manifest.get("spec_name"),
        "status": status.get("status"),
        "status_message": status.get("status_message"),
        "manifest_path": str(snapback_manifest_path(run_dir).resolve()),
        "status_path": str(snapback_status_path(run_dir).resolve()),
        "report_json": str(report_json_path(run_dir).resolve()),
        "report_md": str(report_md_path(run_dir).resolve()),
        "spec_snapshot": str(spec_snapshot_path(run_dir).resolve()),
        "catalog_snapshot": str(catalog_snapshot_path(run_dir).resolve()),
        "views_manifest": (
            str(views_manifest_path(run_dir).resolve()) if views_manifest_path(run_dir).exists() else None
        ),
        "pre_nick_duplex_visual_contract": (
            str(pre_nick_duplex_visual_contract_path(run_dir).resolve())
            if pre_nick_duplex_visual_contract_path(run_dir).exists()
            else None
        ),
        "post_nick_exposed_visual_contract": (
            str(post_nick_exposed_visual_contract_path(run_dir).resolve())
            if post_nick_exposed_visual_contract_path(run_dir).exists()
            else None
        ),
        "post_nick_foldback_visual_contract": (
            str(post_nick_foldback_visual_contract_path(run_dir).resolve())
            if post_nick_foldback_visual_contract_path(run_dir).exists()
            else None
        ),
        "snapback_triptych_visual_contracts": (
            str(snapback_triptych_visual_contracts_path(run_dir).resolve())
            if snapback_triptych_visual_contracts_path(run_dir).exists()
            else None
        ),
        "pre_nick_duplex_view": (
            str(pre_nick_duplex_view_path(run_dir).resolve()) if pre_nick_duplex_view_path(run_dir).exists() else None
        ),
        "post_nick_exposed_view": (
            str(post_nick_exposed_view_path(run_dir).resolve())
            if post_nick_exposed_view_path(run_dir).exists()
            else None
        ),
        "post_nick_foldback_view": (
            str(post_nick_foldback_view_path(run_dir).resolve())
            if post_nick_foldback_view_path(run_dir).exists()
            else None
        ),
        "snapback_triptych_job": (
            str(snapback_triptych_job_path(run_dir).resolve()) if snapback_triptych_job_path(run_dir).exists() else None
        ),
        "baserender_jobs_dir": (
            str(baserender_jobs_dir(run_dir).resolve()) if baserender_jobs_dir(run_dir).exists() else None
        ),
        "plots_dir": str(renders_dir(run_dir).resolve()) if renders_dir(run_dir).exists() else None,
        "snapback_triptych_render": str(triptych_render.resolve()) if triptych_render is not None else None,
        "artifacts": manifest.get("artifacts", []),
    }


def _validate_materialized_hit_bundles(run_dir: Path) -> None:
    report_payload = json.loads(solve_report_json_path(run_dir).read_text(encoding="utf-8"))
    metadata = report_payload.get("metadata", {})
    hits = report_payload.get("hits", [])
    workspace_root = Path(report_payload["workspace_root"]).resolve()
    expected_hit_count = metadata.get("materialized_hit_count")
    if not isinstance(expected_hit_count, int):
        raise ValueError("Snapback solve materialized_hit_count drift detected.")
    materialized_hits = [hit for hit in hits if hit.get("materialized_run_dir") is not None]
    if len(materialized_hits) != expected_hit_count:
        raise ValueError("Snapback solve materialized_hit_count drift detected.")
    observed_ranks = {hit.get("rank") for hit in materialized_hits}
    if observed_ranks != set(range(1, expected_hit_count + 1)):
        raise ValueError("Snapback solve materialized hit rank coverage drift detected.")
    seen_materialized_dirs: set[str] = set()
    for hit in materialized_hits:
        rank = hit.get("rank")
        materialized_run_dir = hit["materialized_run_dir"]
        if materialized_run_dir in seen_materialized_dirs:
            raise ValueError("Duplicate materialized snapback hit bundle path detected.")
        seen_materialized_dirs.add(materialized_run_dir)
        expected_name = f"hit_{int(rank):02d}"
        if Path(materialized_run_dir).name != expected_name:
            raise ValueError("Snapback solve materialized hit path/rank drift detected.")
        hit_run_dir = (workspace_root / materialized_run_dir).resolve()
        try:
            hit_run_dir.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError(f"Materialized snapback hit bundle escaped workspace_root: {hit_run_dir}") from exc
        if not hit_run_dir.exists():
            raise FileNotFoundError(f"Materialized snapback hit bundle missing: {hit_run_dir}")
        _explicit_show_payload(hit_run_dir)
        _validate_materialized_hit_alignment(hit=hit, hit_run_dir=hit_run_dir)


def _solve_show_payload(run_dir: Path) -> dict[str, object]:
    manifest = load_solve_manifest(run_dir)
    status = load_solve_status(run_dir)
    expected_run_dir = str(run_dir)
    workspace_root = _required_existing_manifest_path(
        manifest,
        field="workspace_root",
        label="Snapback solve manifest workspace_root",
    )
    expected_workspace_root = str(workspace_root)
    if manifest.get("kind") != "solve":
        raise ValueError("Snapback solve manifest kind drift detected.")
    if manifest.get("workflow") != "snapback_solve":
        raise ValueError("Snapback solve manifest workflow drift detected.")
    if manifest.get("contract") != "single_nick_snapback_solve_v3":
        raise ValueError("Unsupported snapback solve contract version.")
    if status.get("workflow") != "snapback_solve":
        raise ValueError("Snapback solve status workflow drift detected.")
    if status.get("contract") != "single_nick_snapback_solve_v3":
        raise ValueError("Unsupported snapback solve status contract version.")
    if manifest.get("stage") != "snapback" or status.get("stage") != "snapback":
        raise ValueError("Snapback solve stage drift detected.")
    if manifest.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback solve manifest run_dir drift detected.")
    if status.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback solve status run_dir drift detected.")
    if manifest.get("spec_name") != status.get("spec_name"):
        raise ValueError("Snapback solve manifest/status spec_name drift detected.")
    if manifest.get("status") != status.get("status"):
        raise ValueError("Snapback solve manifest/status status drift detected.")
    required_paths = [
        solve_report_json_path(run_dir),
        solve_report_md_path(run_dir),
        solve_manifest_path(run_dir),
        solve_status_path(run_dir),
        solve_input_spec_path(run_dir),
        solve_resolved_catalog_path(run_dir),
        solve_hits_table_path(run_dir),
        solve_frontier_table_path(run_dir),
        materialized_hits_dir(run_dir),
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required snapback solve artifact missing: {path}")
    solve_report_payload = json.loads(solve_report_json_path(run_dir).read_text(encoding="utf-8"))
    if solve_report_payload.get("run_dir") != expected_run_dir:
        raise ValueError("Snapback solve report run_dir drift detected.")
    if solve_report_payload.get("workspace_root") != expected_workspace_root:
        raise ValueError("Snapback solve report workspace_root drift detected.")
    _validate_materialized_hit_bundles(run_dir)
    return {
        "kind": "solve",
        "run_dir": str(run_dir),
        "spec_name": manifest.get("spec_name"),
        "status": status.get("status"),
        "status_message": status.get("status_message"),
        "solve_report": str(solve_report_json_path(run_dir).resolve()),
        "solve_report_md": str(solve_report_md_path(run_dir).resolve()),
        "solve_manifest": str(solve_manifest_path(run_dir).resolve()),
        "solve_status": str(solve_status_path(run_dir).resolve()),
        "input_solve_spec": str(solve_input_spec_path(run_dir).resolve()),
        "resolved_catalog": str(solve_resolved_catalog_path(run_dir).resolve()),
        "table__hits": str(solve_hits_table_path(run_dir).resolve()),
        "table__frontier": str(solve_frontier_table_path(run_dir).resolve()),
        "materialized_hits_dir": str(materialized_hits_dir(run_dir).resolve()),
    }


def snapback_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved = Path(run_dir).expanduser().resolve()
    explicit_manifest_exists = snapback_manifest_path(resolved).exists()
    solve_manifest_exists = solve_manifest_path(resolved).exists()
    if explicit_manifest_exists and solve_manifest_exists:
        raise ValueError(f"Ambiguous snapback run directory contains explicit and solve manifests: {resolved}")
    if explicit_manifest_exists:
        return _explicit_show_payload(resolved)
    if solve_manifest_exists:
        return _solve_show_payload(resolved)
    raise FileNotFoundError(f"Unsupported snapback run directory: {resolved}")
