"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/promoter/analysis_surfaces.py

Study-owned exploratory-analysis route inventory for promoter-study snapshots.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dnadesign.densegen import inspect_analysis_surface
from dnadesign.ops.status.paths import resolve_repo_relative_path

from .latentdna_contract import latentdna_doc_path
from .record_normalizer import PromoterStudyResolvedContext

_STATUS_KIND = "promoter-study-status"
_DEFAULT_DENSEGEN_WORKSPACE_DOC = "README.md"
_DEFAULT_CLUSTER_DOC = "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md"
_DEFAULT_CLUSTER_WORKSPACE_EXAMPLE = "src/dnadesign/cluster/workspaces/promoter_clusters_v1/config.yaml"
_CLUSTER_PLOT_FAMILIES = [
    "umap_png",
    "composition_proportions_png",
    "diversity_png",
    "numeric_violin_png",
    "resolution_sweep_png",
]


def inspect_promoter_exploratory_analysis(
    *,
    study_context: PromoterStudyResolvedContext,
    latentdna_state: Mapping[str, object] | None,
    downstream_surfaces: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    return {
        "densegen": _inspect_densegen_surface(study_context=study_context),
        "latentdna": _inspect_latentdna_surface(
            study_context=study_context,
            latentdna_state=latentdna_state,
        ),
        "cluster": _inspect_cluster_surface(
            study_context=study_context,
            cluster_state=(downstream_surfaces or {}).get("cluster"),
        ),
    }


def _inspect_densegen_surface(
    *,
    study_context: PromoterStudyResolvedContext,
) -> dict[str, object]:
    densegen_config = _pipeline_mapping(study_context=study_context, key="densegen")
    densegen_analysis_surface = (
        densegen_config.get("analysis_surface") if isinstance(densegen_config.get("analysis_surface"), Mapping) else {}
    )
    expected_contract_ref = (
        _string_or_none(densegen_analysis_surface.get("contract_ref")) or "densegen.analysis_surface.v2"
    )
    densegen_config_path = _densegen_config_path(study_context=study_context, densegen_config=densegen_config)
    workspace_path = _resolve_repo_path(
        study_context=study_context,
        raw_path=_string_or_none(densegen_config.get("workspace")),
    )
    if workspace_path is None and densegen_config_path is not None:
        workspace_path = densegen_config_path.parent
    doc_path = _resolve_repo_path(
        study_context=study_context,
        raw_path=_string_or_none(densegen_config.get("doc")),
    )
    if doc_path is None and workspace_path is not None:
        doc_path = workspace_path / _DEFAULT_DENSEGEN_WORKSPACE_DOC

    plots_root = workspace_path / "outputs" / "plots" if workspace_path is not None else None
    current_inventory = plots_root / "current_inventory.json" if plots_root is not None else None
    artifact_ledger = plots_root / "artifact_ledger.json" if plots_root is not None else None
    plot_manifest_path = plots_root / "plot_manifest.json" if plots_root is not None else None

    config_arg = _command_path(study_context=study_context, path=densegen_config_path)
    payload: dict[str, object] = {
        "configured": workspace_path is not None or densegen_config_path is not None,
        "state": "not_configured",
        "surface_status": "not_configured",
        "entry_artifact": study_context.densegen_dataset_id,
        "contract_ref": expected_contract_ref,
        "doc": _path_string(doc_path),
        "workspace_ref": _path_string(workspace_path),
        "config_ref": _path_string(densegen_config_path),
        "generated_surface": [],
        "operator_visible_surface": [],
        "optional_surface": [],
        "internal_or_hidden_surface": [],
        "historical_ledger_surface": [],
        "default_plot_ids": [],
        "optional_plot_ids": [],
        "read_only_plot_ids": [],
        "local_artifact_plot_ids": [],
        "rendered_plot_count": 0,
        "rendered_plots": [],
        "notebook_generated": False,
        "artifact_paths": {
            "plots_root": _path_string(plots_root),
            "current_inventory": _path_string(current_inventory),
            "artifact_ledger": _path_string(artifact_ledger),
            "plots_manifest": _path_string(plot_manifest_path),
            "notebook_path": None,
            "records_export_prefix": _path_string(workspace_path / "outputs" / "notebooks" / "records_preview")
            if workspace_path is not None
            else None,
        },
    }
    if config_arg is not None:
        payload["commands"] = {
            "plot": f"uv run dense plot -c {config_arg}",
            "notebook_generate": f"uv run dense notebook generate -c {config_arg}",
            "notebook_run": f"uv run dense notebook run -c {config_arg}",
        }
    if workspace_path is None and densegen_config_path is None:
        return payload

    if densegen_config_path is None or not densegen_config_path.is_file():
        payload.update(
            {
                "state": "degraded",
                "surface_status": "degraded",
                "reason_code": "densegen_analysis_surface_unavailable",
                "reason_message": _densegen_surface_config_error_message(densegen_config_path),
                "blocking": True,
            }
        )
        return payload

    try:
        surface = inspect_analysis_surface(densegen_config_path)
    except Exception as exc:
        payload.update(
            {
                "state": "degraded",
                "surface_status": "degraded",
                "reason_code": "densegen_analysis_surface_unavailable",
                "reason_message": (
                    f"DenseGen analysis surface config could not be loaded from {densegen_config_path}: {exc}"
                ),
                "blocking": True,
            }
        )
        return payload

    payload["contract_ref"] = surface.contract_version
    if surface.contract_version != expected_contract_ref:
        payload.update(
            {
                "state": "degraded",
                "surface_status": "degraded",
                "reason_code": "densegen_analysis_surface_contract_mismatch",
                "reason_message": (
                    "DenseGen analysis surface contract mismatch: "
                    f"expected {expected_contract_ref}, got {surface.contract_version}."
                ),
                "blocking": True,
            }
        )
        return payload

    rendered_plots = [_artifact_record_to_snapshot(record) for record in surface.current_inventory]
    surface_payload = surface.to_dict()
    surface_status, reason_code, reason_message, blocking = _densegen_surface_health(surface)
    payload.update(
        {
            "state": surface_status,
            "surface_status": surface_status,
            "reason_code": reason_code,
            "reason_message": reason_message,
            "blocking": blocking,
            "generated_surface": list(surface.generated_surface),
            "operator_visible_surface": list(surface.operator_visible_surface),
            "optional_surface": list(surface.optional_surface),
            "internal_or_hidden_surface": list(surface.internal_or_hidden_surface),
            "historical_ledger_surface": list(surface.historical_ledger_surface),
            "default_plot_ids": list(surface.generated_surface),
            "optional_plot_ids": list(surface.optional_surface),
            "read_only_plot_ids": sorted(
                entry.plot_id for entry in surface.taxonomy if "recoverable_read_only" in entry.degraded_modes
            ),
            "local_artifact_plot_ids": sorted(
                entry.plot_id for entry in surface.taxonomy if "requires_local_artifacts" in entry.degraded_modes
            ),
            "rendered_plot_count": len(rendered_plots),
            "rendered_plots": rendered_plots,
            "notebook_generated": surface.notebook.notebook_path is not None,
            "notebook_visible_plot_ids": list(surface.notebook.gallery_visible_artifact_ids),
            "diagnostics": list(surface_payload["diagnostics"]),
            "freshness": dict(surface_payload["freshness"]),
        }
    )
    payload["artifact_paths"]["notebook_path"] = surface.notebook.notebook_path
    return payload


def _inspect_latentdna_surface(
    *,
    study_context: PromoterStudyResolvedContext,
    latentdna_state: Mapping[str, object] | None,
) -> dict[str, object]:
    workspace_id = _string_or_none((latentdna_state or {}).get("workspace_id"))
    payload: dict[str, object] = {
        "configured": bool(latentdna_state and latentdna_state.get("configured")),
        "state": _string_or_none((latentdna_state or {}).get("state")) or "not_configured",
        "doc": _string_or_none((latentdna_state or {}).get("doc")) or latentdna_doc_path(study_context),
        "binding_ref": _string_or_none((latentdna_state or {}).get("binding_ref")),
        "workspace_ref": _string_or_none((latentdna_state or {}).get("workspace_ref")),
        "snapshot_ref": _string_or_none((latentdna_state or {}).get("snapshot_ref")),
        "entry_artifacts": dict((latentdna_state or {}).get("source_datasets") or {}),
        "workspace_id": workspace_id,
        "deliverable_ids": list((latentdna_state or {}).get("decision_deliverables") or []),
        "ok_deliverables": list((latentdna_state or {}).get("ok_deliverables") or []),
        "pending_deliverables": list((latentdna_state or {}).get("pending_deliverables") or []),
        "export_ids": list((latentdna_state or {}).get("export_ids") or []),
        "exports_ok": bool((latentdna_state or {}).get("exports_ok")),
        "supported_model_families": list((latentdna_state or {}).get("supported_model_families") or []),
        "default_model_family": _string_or_none((latentdna_state or {}).get("default_model_family")),
        "required_wildtype_references": list((latentdna_state or {}).get("required_wildtype_references") or []),
        "browser_default_geometry_ids": list((latentdna_state or {}).get("browser_default_geometry_ids") or []),
        "browser_preferred_hues": list((latentdna_state or {}).get("browser_preferred_hues") or []),
        "last_updated_at": _string_or_none((latentdna_state or {}).get("last_updated_at")),
    }
    if workspace_id is not None:
        commands = {
            "snapshot": f"uv run latentdna workspace snapshot --workspace {workspace_id} --json",
            "validate": f"uv run latentdna validate workspace --workspace {workspace_id} --deep",
            "deliverable_status_template": (
                f"uv run latentdna deliverable status <deliverable-id> --workspace {workspace_id}"
            ),
            "deliverable_run_template": (
                f"uv run latentdna deliverable run <deliverable-id> --workspace {workspace_id}"
            ),
            "notebook_generate_template": (
                f"uv run latentdna notebook generate <notebook-id> --workspace {workspace_id}"
            ),
        }
        payload["commands"] = commands
    error_text = _string_or_none((latentdna_state or {}).get("error"))
    if error_text is not None:
        payload["error"] = error_text
    return payload


def _inspect_cluster_surface(
    *,
    study_context: PromoterStudyResolvedContext,
    cluster_state: Mapping[str, object] | None,
) -> dict[str, object]:
    cluster_config = _pipeline_mapping(study_context=study_context, key="cluster")
    doc_path = _resolve_repo_path(
        study_context=study_context,
        raw_path=_string_or_none(cluster_config.get("doc")) or _string_or_none((cluster_state or {}).get("doc")),
    )
    workspace_example = _resolve_repo_path(
        study_context=study_context,
        raw_path=_string_or_none(cluster_config.get("workspace_example")) or _DEFAULT_CLUSTER_WORKSPACE_EXAMPLE,
    )
    workspace_example_root = workspace_example.parent if workspace_example is not None else None
    payload: dict[str, object] = {
        "configured": bool(cluster_state and cluster_state.get("configured")),
        "state": _string_or_none((cluster_state or {}).get("state")) or "planned",
        "doc": _path_string(doc_path) or _DEFAULT_CLUSTER_DOC,
        "entry_artifact": _string_or_none((cluster_state or {}).get("entry_artifact")),
        "workspace_example_ref": _path_string(workspace_example),
        "workspace_results_root": _path_string(
            workspace_example_root / "outputs" / "cluster" if workspace_example_root is not None else None
        ),
        "plot_families": list(_CLUSTER_PLOT_FAMILIES),
        "artifact_path_templates": {
            "workspace_results_root": _path_string(
                workspace_example_root / "outputs" / "cluster" if workspace_example_root is not None else None
            ),
            "standalone_results_root": "<results-root>/<alias>",
            "fit_run_json": "<results-root>/<alias>/fits/<run-slug>/run.json",
            "fit_labels_parquet": "<results-root>/<alias>/fits/<run-slug>/labels.parquet",
            "umap_json": "<results-root>/<alias>/umap/<run-slug>/umap.json",
            "analysis_json": "<results-root>/<alias>/analysis/<run-slug>/analysis.json",
            "sweep_json": "<results-root>/<alias>/sweeps/<run-slug>/sweep.json",
        },
        "notebook_surface": None,
        "commands": {
            "catalog_show": "uv run ops catalog show cluster.downstream.exploratory-clustering",
            "workspace_list": "uv run cluster workspace list",
            "fit_template": (
                "uv run cluster fit --results-root <results-root> --dataset <feature-dataset> "
                "--x-col <feature-column> --preset method.leiden.fine --name <alias> --write --allow-overwrite"
            ),
            "umap_template": (
                "uv run cluster umap --results-root <results-root> --dataset <feature-dataset> --name <alias>"
            ),
            "analyze_template": (
                "uv run cluster analyze --results-root <results-root> --dataset <feature-dataset> --name <alias>"
            ),
        },
    }
    return payload


def _pipeline_mapping(
    *,
    study_context: PromoterStudyResolvedContext,
    key: str,
) -> Mapping[str, object]:
    payload = study_context.study_pipeline.get(key)
    return payload if isinstance(payload, Mapping) else {}


def _artifact_repo_path(
    *,
    study_context: PromoterStudyResolvedContext,
    artifact_id: str,
) -> Path | None:
    if study_context.ops_contract is None:
        return None
    artifact = study_context.ops_contract.artifacts.get(artifact_id) or {}
    raw_ref = _string_or_none(artifact.get("ref"))
    if raw_ref is None or not raw_ref.startswith("repo:"):
        return None
    return _resolve_repo_path(
        study_context=study_context,
        raw_path=raw_ref.removeprefix("repo:"),
    )


def _resolve_repo_path(
    *,
    study_context: PromoterStudyResolvedContext,
    raw_path: str | None,
) -> Path | None:
    if raw_path is None or study_context.study_repo_root is None:
        return None
    return resolve_repo_relative_path(
        repo_root=study_context.study_repo_root,
        raw_path=raw_path,
        status_kind=_STATUS_KIND,
    )


def _command_path(
    *,
    study_context: PromoterStudyResolvedContext,
    path: Path | None,
) -> str | None:
    if path is None:
        return None
    repo_root = study_context.study_repo_root
    if repo_root is None:
        return str(path)
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _densegen_config_path(
    *,
    study_context: PromoterStudyResolvedContext,
    densegen_config: Mapping[str, object],
) -> Path | None:
    config_path = _artifact_repo_path(study_context=study_context, artifact_id="densegen_config")
    if config_path is not None:
        return config_path
    workspace_path = _resolve_repo_path(
        study_context=study_context,
        raw_path=_string_or_none(densegen_config.get("workspace")),
    )
    if workspace_path is None:
        return None
    return workspace_path / "config.yaml"


def _artifact_record_to_snapshot(record) -> dict[str, object]:
    return {
        "artifact_id": record.artifact_id,
        "plot_id": record.plot_id,
        "variant": record.variant or "default",
        "path": record.relative_path,
        "state": record.state,
        "visible": record.visible,
        "current": record.current,
        "stale": record.stale,
    }


def _densegen_surface_config_error_message(config_path: Path | None) -> str:
    if config_path is None:
        return "DenseGen analysis surface config is not declared for this study."
    return f"DenseGen analysis surface config is missing or unreadable: {config_path}"


def _densegen_surface_health(surface) -> tuple[str, str | None, str | None, bool]:
    visible_inventory = [record for record in surface.current_inventory if record.visible]
    visible_by_plot: dict[str, list[object]] = {}
    for record in visible_inventory:
        visible_by_plot.setdefault(record.plot_id, []).append(record)
    missing_visible = [
        plot_id
        for plot_id in surface.operator_visible_surface
        if not any(record.materialized for record in visible_by_plot.get(plot_id, []))
    ]
    degraded_visible = [
        record.artifact_id
        for record in visible_inventory
        if record.state in {"partial", "degraded", "historical_only", "missing"}
    ]
    stale_visible = [record.artifact_id for record in visible_inventory if record.state == "stale"]
    blocking_diagnostics = [diagnostic for diagnostic in surface.diagnostics if diagnostic.blocking]
    non_blocking_diagnostics = [diagnostic for diagnostic in surface.diagnostics if not diagnostic.blocking]
    if blocking_diagnostics or missing_visible or degraded_visible:
        reasons = []
        if missing_visible:
            reasons.append(f"missing operator-visible plots: {', '.join(sorted(missing_visible))}")
        if degraded_visible:
            reasons.append(f"degraded operator-visible artifacts: {', '.join(sorted(degraded_visible))}")
        if blocking_diagnostics:
            reasons.extend(diagnostic.message for diagnostic in blocking_diagnostics)
        return (
            "degraded",
            "densegen_analysis_surface_incomplete",
            "; ".join(reasons) or "DenseGen analysis surface is incomplete.",
            True,
        )
    attention_reasons = []
    if stale_visible:
        attention_reasons.append(f"stale operator-visible artifacts: {', '.join(sorted(stale_visible))}")
    if surface.notebook.fresh is False:
        attention_reasons.append("DenseGen notebook is stale relative to the current inventory.")
    if surface.freshness.manifest_freshness != "current":
        attention_reasons.append(f"DenseGen current inventory freshness is {surface.freshness.manifest_freshness}.")
    if non_blocking_diagnostics:
        attention_reasons.extend(diagnostic.message for diagnostic in non_blocking_diagnostics)
    if attention_reasons:
        return (
            "attention",
            "densegen_analysis_surface_attention",
            "; ".join(attention_reasons),
            False,
        )
    return ("ok", None, None, False)


def _mapping_keys(value: object) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    return sorted(key for key in value.keys() if _string_or_none(key) is not None)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _string_or_none(item)
        if text is not None:
            result.append(text)
    return result


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _path_string(path: Path | None) -> str | None:
    return str(path) if path is not None else None
