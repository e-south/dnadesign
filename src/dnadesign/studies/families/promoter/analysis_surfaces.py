"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/promoter/analysis_surfaces.py

Study-owned exploratory-analysis route inventory for promoter-study snapshots.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from dnadesign.ops.status.paths import resolve_repo_relative_path

from .record_normalizer import PromoterStudyResolvedContext

_STATUS_KIND = "promoter-study-status"
_DEFAULT_DENSEGEN_WORKSPACE_DOC = "README.md"
_DEFAULT_DENSEGEN_NOTEBOOK_ID = "densegen_run_overview"
_DEFAULT_CLUSTER_DOC = "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md"
_DEFAULT_CLUSTER_WORKSPACE_EXAMPLE = "src/dnadesign/cluster/workspaces/promoter_clusters_v1/config.yaml"
_LATENTDNA_ARTIFACT_ROOTS = [
    "alignments",
    "agreements",
    "clusters",
    "distances",
    "enrichments",
    "exports",
    "neighbors",
    "notebooks",
    "plots",
    "projections",
    "reducers",
    "reduced_views",
    "samples",
    "scalars",
    "snapshots",
    "views",
]
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
    densegen_config_path = _artifact_repo_path(study_context=study_context, artifact_id="densegen_config")
    densegen_config = _pipeline_mapping(study_context=study_context, key="densegen")
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

    notebook_id = _string_or_none(densegen_config.get("notebook")) or _DEFAULT_DENSEGEN_NOTEBOOK_ID
    plots_root = workspace_path / "outputs" / "plots" if workspace_path is not None else None
    plot_manifest_path = plots_root / "plot_manifest.json" if plots_root is not None else None
    rendered_plots = _load_densegen_rendered_plots(plot_manifest_path)
    notebook_path = (
        workspace_path / "outputs" / "notebooks" / f"{notebook_id}.py" if workspace_path is not None else None
    )

    densegen_plot_ids = _string_list(densegen_config.get("default_plot_ids")) or _default_densegen_plot_ids()
    optional_plot_ids = _string_list(densegen_config.get("optional_plot_ids")) or ["dense_array_video_showcase"]
    plot_specs = _densegen_plot_specs()

    if workspace_path is None and densegen_config_path is None:
        state = "not_configured"
    elif rendered_plots or (notebook_path is not None and notebook_path.is_file()):
        state = "ok"
    else:
        state = "configured"

    config_arg = _command_path(study_context=study_context, path=densegen_config_path)
    payload: dict[str, object] = {
        "configured": workspace_path is not None or densegen_config_path is not None,
        "state": state,
        "entry_artifact": study_context.densegen_dataset_id,
        "doc": _path_string(doc_path),
        "workspace_ref": _path_string(workspace_path),
        "config_ref": _path_string(densegen_config_path),
        "default_plot_ids": densegen_plot_ids,
        "optional_plot_ids": optional_plot_ids,
        "read_only_plot_ids": sorted(
            plot_id
            for plot_id, spec in plot_specs.items()
            if str(spec.get("missing_state") or "").strip() == "recoverable_read_only"
        ),
        "local_artifact_plot_ids": sorted(
            plot_id
            for plot_id, spec in plot_specs.items()
            if str(spec.get("missing_state") or "").strip() == "requires_local_artifacts"
        ),
        "rendered_plot_count": len(rendered_plots),
        "rendered_plots": rendered_plots,
        "notebook_generated": bool(notebook_path is not None and notebook_path.is_file()),
        "artifact_paths": {
            "plots_root": _path_string(plots_root),
            "plots_manifest": _path_string(plot_manifest_path),
            "notebook_path": _path_string(notebook_path),
            "records_export_prefix": _path_string(
                workspace_path / "outputs" / "notebooks" / "records_preview" if workspace_path is not None else None
            ),
        },
    }
    if config_arg is not None:
        payload["commands"] = {
            "plot": f"uv run dense plot -c {config_arg}",
            "notebook_generate": f"uv run dense notebook generate -c {config_arg}",
            "notebook_run": f"uv run dense notebook run -c {config_arg}",
        }
    return payload


def _inspect_latentdna_surface(
    *,
    study_context: PromoterStudyResolvedContext,
    latentdna_state: Mapping[str, object] | None,
) -> dict[str, object]:
    latentdna_config = _pipeline_mapping(study_context=study_context, key="latentdna")
    workspace_path = _resolve_repo_path(
        study_context=study_context,
        raw_path=_string_or_none(latentdna_config.get("workspace")),
    )
    doc_path = _resolve_repo_path(
        study_context=study_context,
        raw_path=_string_or_none(latentdna_config.get("doc")),
    )
    workspace_yaml = _load_yaml_mapping(workspace_path / "config.yaml" if workspace_path is not None else None)
    notebooks = _mapping_keys(workspace_yaml.get("notebooks"))
    plots = _mapping_keys(workspace_yaml.get("plots"))
    deliverables = _mapping_keys(workspace_yaml.get("deliverables")) or list(
        (latentdna_state or {}).get("expected_deliverables") or []
    )
    exports = _mapping_keys(workspace_yaml.get("exports")) or list(
        (latentdna_state or {}).get("required_exports") or []
    )
    notebook_id = _string_or_none(latentdna_config.get("notebook")) or next(iter(notebooks), None)
    output_root = _latentdna_output_root(workspace_path=workspace_path, workspace_yaml=workspace_yaml)

    workspace_arg = _command_path(study_context=study_context, path=workspace_path)
    notebook_path = (
        output_root / "notebooks" / notebook_id / "notebook.py"
        if output_root is not None and notebook_id is not None
        else None
    )

    payload: dict[str, object] = {
        "configured": bool(latentdna_state and latentdna_state.get("configured")),
        "state": _string_or_none((latentdna_state or {}).get("state")) or "not_configured",
        "doc": _path_string(doc_path) or _string_or_none((latentdna_state or {}).get("doc")),
        "workspace_ref": _path_string(workspace_path) or _string_or_none((latentdna_state or {}).get("surface_ref")),
        "entry_artifacts": [
            dataset_id
            for dataset_id in (study_context.merged_anchor_dataset_id, study_context.construct_context_dataset_id)
            if dataset_id is not None
        ],
        "notebook_id": notebook_id,
        "plot_ids": plots,
        "deliverable_ids": deliverables,
        "export_ids": exports,
        "expected_deliverables": list((latentdna_state or {}).get("expected_deliverables") or []),
        "required_leiden_runs": list((latentdna_state or {}).get("required_leiden_runs") or []),
        "required_exports": list((latentdna_state or {}).get("required_exports") or []),
        "rendered_plot_count": int((latentdna_state or {}).get("rendered_plot_count") or 0),
        "notebook_generated": bool((latentdna_state or {}).get("notebook_generated")),
        "notebook_smoke_ok": bool((latentdna_state or {}).get("notebook_smoke_ok")),
        "leiden_runs_ok": bool((latentdna_state or {}).get("leiden_runs_ok")),
        "exports_ok": bool((latentdna_state or {}).get("exports_ok")),
        "artifact_roots": list(_LATENTDNA_ARTIFACT_ROOTS),
        "artifact_paths": {
            "output_root": _path_string(output_root),
            "plots_root": _path_string(output_root / "plots" if output_root is not None else None),
            "plots_index": _path_string(output_root / "plots" / "index.json" if output_root is not None else None),
            "plot_manifest_pattern": _path_string(
                output_root / "plots" / "<plot-id>" / "manifest.json" if output_root is not None else None
            ),
            "notebooks_root": _path_string(output_root / "notebooks" if output_root is not None else None),
            "notebook_script": _path_string(notebook_path),
            "notebook_manifest_pattern": _path_string(
                output_root / "notebooks" / "<notebook-id>" / "manifest.json" if output_root is not None else None
            ),
            "artifact_manifest_pattern": _path_string(
                output_root / "<artifact-kind>" / "<artifact-id>" / "manifest.json" if output_root is not None else None
            ),
        },
    }
    if workspace_arg is not None:
        commands = {
            "validate": f"uv run latentdna validate workspace --workspace {workspace_arg} --deep",
            "inspect_artifacts": f"uv run latentdna inspect artifacts --workspace {workspace_arg}",
            "runs_list": f"uv run latentdna runs list --workspace {workspace_arg} --json",
            "deliverable_status_template": (
                f"uv run latentdna deliverable status <deliverable-id> --workspace {workspace_arg}"
            ),
            "deliverable_run_template": (
                f"uv run latentdna deliverable run <deliverable-id> --workspace {workspace_arg}"
            ),
            "notebook_generate": (
                f"uv run latentdna notebook generate {notebook_id or '<notebook-id>'} --workspace {workspace_arg}"
            ),
        }
        if notebook_path is not None:
            notebook_run_path = _command_path(study_context=study_context, path=notebook_path)
            commands["notebook_run"] = f"uv run marimo run {notebook_run_path}"
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


def _densegen_plot_specs() -> dict[str, Mapping[str, object]]:
    try:
        from dnadesign.densegen.src.viz.plot_registry import PLOT_SPECS
    except Exception:
        return {}
    return {str(plot_id): spec for plot_id, spec in dict(PLOT_SPECS).items()}


def _default_densegen_plot_ids() -> list[str]:
    plot_specs = _densegen_plot_specs()
    preferred_order = [
        "dataset_source_inventory",
        "dataset_metadata_heatmap",
        "stage_a_summary",
        "placement_map",
        "run_health",
        "tfbs_usage",
    ]
    return [plot_id for plot_id in preferred_order if plot_id in plot_specs] or preferred_order


def _load_densegen_rendered_plots(plot_manifest_path: Path | None) -> list[dict[str, str]]:
    payload = _load_json_mapping(plot_manifest_path)
    rows = payload.get("plots")
    if not isinstance(rows, list):
        return []
    rendered: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        entry = {
            "plot_id": _string_or_none(row.get("plot_id")) or _string_or_none(row.get("name")) or "unknown",
            "variant": _string_or_none(row.get("variant")) or "default",
            "path": _string_or_none(row.get("path")) or "",
            "plan_name": _string_or_none(row.get("plan_name")) or "unscoped",
        }
        rendered.append(entry)
    return rendered


def _latentdna_output_root(
    *,
    workspace_path: Path | None,
    workspace_yaml: Mapping[str, object],
) -> Path | None:
    if workspace_path is None:
        return None
    workspace_payload = workspace_yaml.get("workspace")
    if isinstance(workspace_payload, Mapping):
        raw_output_root = _string_or_none(workspace_payload.get("output_root"))
        if raw_output_root is not None:
            return (workspace_path / raw_output_root).resolve()
    return (workspace_path / "outputs" / "latentdna").resolve()


def _load_yaml_mapping(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_mapping(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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
