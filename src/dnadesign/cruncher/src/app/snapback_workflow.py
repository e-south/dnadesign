"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_workflow.py

Application orchestration for v2 explicit snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

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
    post_nick_exposed_job_path,
    post_nick_exposed_view_path,
    post_nick_exposed_visual_contract_path,
    post_nick_foldback_job_path,
    post_nick_foldback_view_path,
    post_nick_foldback_visual_contract_path,
    pre_nick_duplex_job_path,
    pre_nick_duplex_view_path,
    pre_nick_duplex_visual_contract_path,
    renders_dir,
    report_json_path,
    report_md_path,
    snapshot_explicit_inputs,
    solve_hits_table_path,
    solve_input_spec_path,
    solve_manifest_path,
    solve_report_json_path,
    solve_report_md_path,
    solve_resolved_catalog_path,
    solve_status_path,
    spec_snapshot_path,
    views_manifest_path,
    write_baserender_job,
    write_candidate_table,
    write_manifest,
    write_report,
    write_status,
    write_view_bundle,
)
from dnadesign.cruncher.snapback.load import load_snapback_spec
from dnadesign.cruncher.snapback.planner import (
    build_invalid_catalog_report,
    build_snapback_report,
    render_markdown_report,
)
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
        preset_id=spec.design.nickase.catalog.preset,
        additional_paths=spec.design.nickase.catalog.additional_paths,
        workspace_root=workspace_root,
    )
    return (
        catalog,
        resolved_paths,
        _catalog_source_label(
            preset=spec.design.nickase.catalog.preset,
            resolved_paths=resolved_paths,
        ),
    )


def _view_title(*, spec_name: str, solution_id: str, state_label: str) -> str:
    _ = (spec_name, solution_id)
    return state_label


def validate_snapback_spec(path: str | Path):
    spec, spec_path, workspace_root = load_snapback_spec(path)
    catalog_source = _catalog_source_label(
        preset=spec.design.nickase.catalog.preset,
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
    ensure_run_dirs(run_dir)
    snapshot_explicit_inputs(run_dir, spec_path=spec_path, catalog_yaml=catalog_yaml)
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_report(run_dir, report, markdown=render_markdown_report(report))
    write_candidate_table(run_dir, report)
    if spec.output.emit_visual_contracts and report.candidate is not None:
        pre_nick_duplex = build_pre_nick_duplex_view(
            report=report,
            solution_id=snapback_design_id,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=snapback_design_id,
                state_label="pre-nick duplex",
            ),
        )
        post_nick_exposed = build_post_nick_exposed_view(
            report=report,
            solution_id=snapback_design_id,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=snapback_design_id,
                state_label="post-nick exposed",
            ),
        )
        post_nick_foldback = build_post_nick_foldback_view(
            report=report,
            solution_id=snapback_design_id,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=snapback_design_id,
                state_label="post-nick foldback",
            ),
        )
        pre_nick_duplex_visual_contract = build_pre_nick_snapback_visual(
            report=report,
            solution_id=snapback_design_id,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=snapback_design_id,
                state_label="pre-nick duplex",
            ),
        )
        post_nick_exposed_visual_contract = build_post_nick_exposed_snapback_visual(
            report=report,
            solution_id=snapback_design_id,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=snapback_design_id,
                state_label="post-nick exposed",
            ),
        )
        post_nick_foldback_visual_contract = build_post_nick_foldback_snapback_visual(
            report=report,
            solution_id=snapback_design_id,
            title=_view_title(
                spec_name=report.spec_name,
                solution_id=snapback_design_id,
                state_label="post-nick foldback",
            ),
        )
        write_view_bundle(
            run_dir,
            pre_nick_duplex=pre_nick_duplex,
            post_nick_exposed=post_nick_exposed,
            post_nick_foldback=post_nick_foldback,
            pre_nick_duplex_visual_contract=pre_nick_duplex_visual_contract,
            post_nick_exposed_visual_contract=post_nick_exposed_visual_contract,
            post_nick_foldback_visual_contract=post_nick_foldback_visual_contract,
            manifest=build_views_manifest(
                solution_id=snapback_design_id,
                include_jobs=spec.output.emit_baserender_jobs,
            ),
        )
        if spec.output.emit_baserender_jobs:
            write_baserender_job(
                pre_nick_duplex_job_path(run_dir),
                build_single_view_job(
                    input_filename=pre_nick_duplex_visual_contract_path(run_dir).name,
                    output_filename="pre_nick_duplex.png",
                ),
            )
            write_baserender_job(
                post_nick_exposed_job_path(run_dir),
                build_single_view_job(
                    input_filename=post_nick_exposed_visual_contract_path(run_dir).name,
                    output_filename="post_nick_exposed.png",
                ),
            )
            write_baserender_job(
                post_nick_foldback_job_path(run_dir),
                build_single_view_job(
                    input_filename=post_nick_foldback_visual_contract_path(run_dir).name,
                    output_filename="post_nick_foldback.png",
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
        "pre_nick_duplex_job",
        "post_nick_exposed_job",
        "post_nick_foldback_job",
    ):
        if key in declared_artifacts:
            candidate = run_dir / declared_artifacts[key]
            if not candidate.exists():
                raise FileNotFoundError(f"Declared snapback visual artifact missing: {candidate}")
    return {
        "kind": "explicit",
        "run_dir": str(run_dir),
        "spec_name": manifest.get("spec_name"),
        "status": status.get("status"),
        "status_message": status.get("status_message"),
        "manifest_path": str((run_dir / "meta" / "snapback_manifest.json").resolve()),
        "status_path": str((run_dir / "meta" / "snapback_status.json").resolve()),
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
        "pre_nick_duplex_job": (
            str(pre_nick_duplex_job_path(run_dir).resolve()) if pre_nick_duplex_job_path(run_dir).exists() else None
        ),
        "post_nick_exposed_job": (
            str(post_nick_exposed_job_path(run_dir).resolve()) if post_nick_exposed_job_path(run_dir).exists() else None
        ),
        "post_nick_foldback_job": (
            str(post_nick_foldback_job_path(run_dir).resolve())
            if post_nick_foldback_job_path(run_dir).exists()
            else None
        ),
        "baserender_jobs_dir": (
            str(baserender_jobs_dir(run_dir).resolve()) if baserender_jobs_dir(run_dir).exists() else None
        ),
        "renders_dir": str(renders_dir(run_dir).resolve()),
        "pre_nick_duplex_render": (
            str((renders_dir(run_dir) / "pre_nick_duplex.png").resolve())
            if (renders_dir(run_dir) / "pre_nick_duplex.png").exists()
            else None
        ),
        "post_nick_exposed_render": (
            str((renders_dir(run_dir) / "post_nick_exposed.png").resolve())
            if (renders_dir(run_dir) / "post_nick_exposed.png").exists()
            else None
        ),
        "post_nick_foldback_render": (
            str((renders_dir(run_dir) / "post_nick_foldback.png").resolve())
            if (renders_dir(run_dir) / "post_nick_foldback.png").exists()
            else None
        ),
        "artifacts": manifest.get("artifacts", []),
    }


def _solve_show_payload(run_dir: Path) -> dict[str, object]:
    manifest = load_solve_manifest(run_dir)
    status = load_solve_status(run_dir)
    expected_run_dir = str(run_dir)
    if manifest.get("kind") != "solve":
        raise ValueError("Snapback solve manifest kind drift detected.")
    if manifest.get("workflow") != "snapback_solve":
        raise ValueError("Snapback solve manifest workflow drift detected.")
    if manifest.get("contract") != "single_nick_snapback_solve_v2":
        raise ValueError("Unsupported snapback solve contract version.")
    if status.get("workflow") != "snapback_solve":
        raise ValueError("Snapback solve status workflow drift detected.")
    if status.get("contract") != "single_nick_snapback_solve_v2":
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
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required snapback solve artifact missing: {path}")
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
        "hits_root": str((run_dir / "hits").resolve()),
    }


def snapback_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved = Path(run_dir).expanduser().resolve()
    explicit_manifest_exists = (resolved / "meta" / "snapback_manifest.json").exists()
    solve_manifest_exists = (resolved / "solve_manifest.json").exists()
    if explicit_manifest_exists and solve_manifest_exists:
        raise ValueError(f"Ambiguous snapback run directory contains explicit and solve manifests: {resolved}")
    if explicit_manifest_exists:
        return _explicit_show_payload(resolved)
    if solve_manifest_exists:
        return _solve_show_payload(resolved)
    raise FileNotFoundError(f"Unsupported snapback run directory: {resolved}")
