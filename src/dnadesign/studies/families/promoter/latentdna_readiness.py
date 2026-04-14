"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/promoter/latentdna_readiness.py

Read-only latentdna readiness inspection for promoter-study status surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dnadesign.ops.status.paths import resolve_repo_relative_path

from .record_normalizer import PromoterStudyResolvedContext

_STATUS_KIND = "promoter-study-status"
_DEFAULT_NOTEBOOK_ID = "browser"
_DEFAULT_DOC_PATH = "src/dnadesign/latentdna/docs/workflows/promoter-study-latent-atlas.md"


def inspect_promoter_latentdna_readiness(
    *,
    study_context: PromoterStudyResolvedContext,
) -> dict[str, object]:
    latentdna_config = study_context.study_pipeline.get("latentdna")
    if not isinstance(latentdna_config, Mapping):
        return _build_empty_payload(doc_path=_DEFAULT_DOC_PATH)
    workspace_raw = _string_or_none(latentdna_config.get("workspace"))
    doc_path = _string_or_none(latentdna_config.get("doc")) or _DEFAULT_DOC_PATH

    expected_deliverables = _string_list(latentdna_config.get("expected_deliverables"))
    required_leiden_runs = _string_list(latentdna_config.get("required_leiden_runs"))
    required_exports = _string_list(latentdna_config.get("required_exports"))
    notebook_id = _string_or_none(latentdna_config.get("notebook")) or _DEFAULT_NOTEBOOK_ID
    if workspace_raw is None or study_context.study_repo_root is None:
        return _build_empty_payload(
            doc_path=doc_path,
            expected_deliverables=expected_deliverables,
            required_leiden_runs=required_leiden_runs,
            required_exports=required_exports,
        )

    workspace_path = resolve_repo_relative_path(
        repo_root=study_context.study_repo_root,
        raw_path=workspace_raw,
        status_kind=_STATUS_KIND,
    )
    payload = _build_empty_payload(
        doc_path=doc_path,
        surface_ref=str(workspace_path),
        configured=True,
        state="missing",
        workspace_id=workspace_path.name,
        expected_deliverables=expected_deliverables,
        required_leiden_runs=required_leiden_runs,
        required_exports=required_exports,
    )
    if not workspace_path.exists():
        return payload

    from dnadesign.latentdna.src.io.json_io import read_json
    from dnadesign.latentdna.src.services.deliverable_service import deliverable_status
    from dnadesign.latentdna.src.workspaces.loader import load_workspace_config

    try:
        context = load_workspace_config(workspace_path)
    except Exception:
        payload["state"] = "error"
        return payload

    payload["workspace_id"] = context.workspace_id
    if not expected_deliverables:
        expected_deliverables = sorted(context.config.deliverables)
        payload["expected_deliverables"] = list(expected_deliverables)
        payload["missing_deliverables"] = list(expected_deliverables)

    ok_deliverables: list[str] = []
    missing_deliverables: list[str] = []
    deliverable_error = False
    for deliverable_id in expected_deliverables:
        if deliverable_id not in context.config.deliverables:
            missing_deliverables.append(deliverable_id)
            continue
        try:
            status = deliverable_status(context.workspace_dir, deliverable_id).status
        except Exception:
            deliverable_error = True
            missing_deliverables.append(deliverable_id)
            continue
        if status == "ok":
            ok_deliverables.append(deliverable_id)
            continue
        if status == "error":
            deliverable_error = True
        missing_deliverables.append(deliverable_id)
    payload["ok_deliverables"] = ok_deliverables
    payload["missing_deliverables"] = missing_deliverables

    payload["rendered_plot_count"] = _rendered_plot_count(
        plots_root=context.output_root / "plots",
        read_json=read_json,
    )
    payload["notebook_generated"] = (context.output_root / "notebooks" / f"{notebook_id}.py").is_file()
    notebook_smoke_ok, notebook_error = _notebook_smoke_ok(
        health_path=context.output_root / "notebooks" / "health.json",
        read_json=read_json,
    )
    payload["notebook_smoke_ok"] = notebook_smoke_ok

    leiden_runs_ok, any_leiden_artifacts, leiden_error = _required_artifacts_ok(
        artifacts_root=context.output_root / "clusters",
        artifact_ids=required_leiden_runs,
        read_json=read_json,
    )
    payload["leiden_runs_ok"] = leiden_runs_ok

    exports_ok, any_export_artifacts, export_error = _required_artifacts_ok(
        artifacts_root=context.output_root / "exports",
        artifact_ids=required_exports,
        read_json=read_json,
    )
    payload["exports_ok"] = exports_ok

    has_materialized_outputs = any(
        (
            bool(ok_deliverables),
            bool(payload["rendered_plot_count"]),
            bool(payload["notebook_generated"]),
            any_leiden_artifacts,
            any_export_artifacts,
            _has_any_artifact_manifests(context.output_root),
        )
    )
    all_main_ready = (
        len(ok_deliverables) == len(expected_deliverables)
        and payload["notebook_smoke_ok"] is True
        and payload["leiden_runs_ok"] is True
        and payload["exports_ok"] is True
    )
    if deliverable_error or notebook_error or leiden_error or export_error:
        payload["state"] = "error"
    elif all_main_ready:
        payload["state"] = "ok"
    elif has_materialized_outputs:
        payload["state"] = "attention"
    else:
        payload["state"] = "missing"
    return payload


def _build_empty_payload(
    *,
    doc_path: str,
    surface_ref: str | None = None,
    configured: bool = False,
    state: str = "not_configured",
    workspace_id: str | None = None,
    expected_deliverables: list[str] | None = None,
    required_leiden_runs: list[str] | None = None,
    required_exports: list[str] | None = None,
) -> dict[str, object]:
    expected = list(expected_deliverables or [])
    leiden_runs = list(required_leiden_runs or [])
    exports = list(required_exports or [])
    return {
        "configured": configured,
        "state": state,
        "doc": doc_path,
        "surface_ref": surface_ref,
        "workspace_id": workspace_id,
        "expected_deliverables": expected,
        "ok_deliverables": [],
        "missing_deliverables": list(expected),
        "rendered_plot_count": 0,
        "notebook_generated": False,
        "notebook_smoke_ok": False,
        "leiden_runs_ok": False if leiden_runs else True,
        "exports_ok": False if exports else True,
        "required_leiden_runs": leiden_runs,
        "required_exports": exports,
    }


def _has_any_artifact_manifests(output_root: Path) -> bool:
    for manifest_path in output_root.rglob("manifest.json"):
        if "logs" not in manifest_path.parts:
            return True
    return False


def _rendered_plot_count(
    *,
    plots_root: Path,
    read_json,
) -> int:
    index_path = plots_root / "index.json"
    if index_path.is_file():
        try:
            payload = read_json(index_path)
        except Exception:
            return 0
        rows = payload.get("plots")
        if isinstance(rows, list):
            return sum(
                1
                for row in rows
                if isinstance(row, dict)
                and str(row.get("status") or "").strip() == "ok"
                and bool(row.get("output_paths"))
            )

    rendered = 0
    if not plots_root.is_dir():
        return rendered
    for plot_dir in sorted(candidate for candidate in plots_root.iterdir() if candidate.is_dir()):
        manifest_path = plot_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        outputs = manifest.get("outputs")
        if str(manifest.get("status") or "").strip() == "ok" and isinstance(outputs, list) and outputs:
            rendered += 1
    return rendered


def _notebook_smoke_ok(
    *,
    health_path: Path,
    read_json,
) -> tuple[bool, bool]:
    if not health_path.is_file():
        return False, False
    try:
        payload = read_json(health_path)
    except Exception:
        return False, True
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        return False, True
    all_checks = all(bool(value) for value in checks.values())
    status_text = str(payload.get("status") or "").strip()
    return status_text == "ok" and all_checks, status_text == "error"


def _required_artifacts_ok(
    *,
    artifacts_root: Path,
    artifact_ids: list[str],
    read_json,
) -> tuple[bool, bool, bool]:
    if not artifact_ids:
        return True, False, False
    any_present = False
    saw_error = False
    for artifact_id in artifact_ids:
        manifest_path = artifacts_root / artifact_id / "manifest.json"
        if not manifest_path.is_file():
            return False, any_present, saw_error
        any_present = True
        try:
            manifest = read_json(manifest_path)
        except Exception:
            return False, any_present, True
        status_text = str(manifest.get("status") or "").strip()
        if status_text == "error":
            saw_error = True
            return False, any_present, saw_error
        if status_text != "ok":
            return False, any_present, saw_error
    return True, any_present, saw_error


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _string_or_none(item)
        if text is not None:
            result.append(text)
    return result
