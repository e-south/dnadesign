"""
Dry-run preview helpers for mutating latentdna CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.errors import ArtifactConflictError, MissingArtifactError, WorkspaceValidationError
from ..contracts.recipe import expected_step_artifacts, topological_step_order
from ..contracts.result import CommandResult
from ..services._artifacts import artifact_dir, artifact_kind_for_category
from ..services.plot_service import resolve_plot_request
from ..workspaces.loader import builtin_templates_dir, load_workspace_config, resolve_repo_path

_ARTIFACT_LABELS = {
    "alignment_set": "alignment",
    "cluster_set": "cluster",
    "distance_set": "distance",
    "enrichment_set": "enrichment",
    "export_bundle": "export",
    "neighbor_set": "neighbors",
    "notebook": "notebook",
    "plot": "plot",
    "projection": "projection",
    "reducer": "reducer",
    "reduced_view": "reduced view",
    "sample_set": "sample",
    "scalar_table": "scalar",
    "snapshot": "snapshot",
    "view": "view",
}


def _artifact_label(artifact_kind: str) -> str:
    return _ARTIFACT_LABELS.get(artifact_kind, artifact_kind.replace("_", " "))


def _ensure_preview_targets_available(output_paths: list[Path], *, artifact_kind: str, force: bool) -> None:
    if force:
        return
    for output_path in output_paths:
        if output_path.exists():
            raise ArtifactConflictError(f"{_artifact_label(artifact_kind)} artifact already exists: {output_path}")


def _preview_payload(
    *,
    workspace_id: str,
    command: str,
    artifact_kind: str | None,
    artifact_id: str | None,
    outputs: list[Path],
    inputs: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    result = CommandResult(
        command=command,
        workspace_id=workspace_id,
        status="ok",
        dry_run=True,
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        outputs=[output.as_posix() for output in outputs],
        inputs=inputs,
        warnings=["dry-run only; no artifacts were written", *(warnings or [])],
        metrics=metrics or {},
    )
    return result.model_dump(mode="json")


def preview_workspace_init(
    *,
    workspace: str | Path,
    template: str,
    from_study_dir: str | Path | None = None,
) -> dict[str, Any]:
    workspace_dir = Path(workspace).resolve()
    template_dir = builtin_templates_dir() / template
    if not template_dir.is_dir():
        raise WorkspaceValidationError(f"unknown workspace template: {template}")
    if workspace_dir.exists():
        raise WorkspaceValidationError(f"workspace already exists: {workspace_dir}")
    if from_study_dir is not None:
        study_dir = resolve_repo_path(from_study_dir)
        if not study_dir.exists():
            raise WorkspaceValidationError(f"study directory not found: {study_dir}")
    payload = _preview_payload(
        workspace_id=workspace_dir.name,
        command="workspace init",
        artifact_kind="workspace",
        artifact_id=workspace_dir.name,
        outputs=[workspace_dir],
        inputs={
            "template": template,
            **({"study_dir": str(from_study_dir)} if from_study_dir is not None else {}),
        },
    )
    payload["config_path"] = (workspace_dir / "config.yaml").as_posix()
    return payload


def preview_snapshot_build(workspace: str | Path, snapshot_id: str, *, source_id: str, force: bool) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_source(source_id)
    output_dir = artifact_dir(context, artifact_kind="snapshot", artifact_id=snapshot_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="snapshot", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="snapshot build",
        artifact_kind="snapshot",
        artifact_id=snapshot_id,
        outputs=[output_dir],
        inputs={"snapshot": snapshot_id, "source": source_id},
    )


def preview_alignment_build(workspace: str | Path, alignment_id: str, *, force: bool) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_alignment(alignment_id)
    output_dir = artifact_dir(context, artifact_kind="alignment_set", artifact_id=alignment_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="alignment_set", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="alignment build",
        artifact_kind="alignment_set",
        artifact_id=alignment_id,
        outputs=[output_dir],
        inputs={"alignment": alignment_id},
    )


def preview_view_materialize(workspace: str | Path, view_id: str, *, force: bool) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    view = context.require_source_view(view_id)
    output_dir = artifact_dir(context, artifact_kind="view", artifact_id=view_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="view", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="view materialize",
        artifact_kind="view",
        artifact_id=view_id,
        outputs=[output_dir],
        inputs={"view": view_id, "source": view.source},
    )


def preview_view_derive(workspace: str | Path, view_id: str, *, force: bool) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_view(view_id)
    output_dir = artifact_dir(context, artifact_kind="view", artifact_id=view_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="view", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="view derive",
        artifact_kind="view",
        artifact_id=view_id,
        outputs=[output_dir],
        inputs={"view": view_id},
    )


def preview_view_reduce(
    workspace: str | Path,
    view_id: str,
    *,
    reducer_id: str,
    sample_id: str | None,
    alignment_id: str | None,
    reduced_view_id: str | None,
    force: bool,
) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_view(view_id)
    output_dirs = [artifact_dir(context, artifact_kind="reducer", artifact_id=reducer_id)]
    if reduced_view_id is not None:
        output_dirs.append(artifact_dir(context, artifact_kind="reduced_view", artifact_id=reduced_view_id))
    _ensure_preview_targets_available(output_dirs, artifact_kind="reducer", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="view reduce",
        artifact_kind="reducer",
        artifact_id=reducer_id,
        outputs=output_dirs,
        inputs={"view": view_id, "sample": sample_id, "alignment": alignment_id, "reduced_view": reduced_view_id},
    )


def preview_scalar_derive(workspace: str | Path, scalar_id: str, *, force: bool) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_scalar(scalar_id)
    output_dir = artifact_dir(context, artifact_kind="scalar_table", artifact_id=scalar_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="scalar_table", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="scalar derive",
        artifact_kind="scalar_table",
        artifact_id=scalar_id,
        outputs=[output_dir],
        inputs={"scalar": scalar_id},
    )


def preview_sample_build(
    workspace: str | Path,
    sample_id: str,
    *,
    view_id: str | None,
    strategy: str,
    explicit_ids: list[str] | None,
    input_sample_ids: list[str] | None,
    force: bool,
) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    if view_id is not None:
        context.require_view(view_id)
    output_dir = artifact_dir(context, artifact_kind="sample_set", artifact_id=sample_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="sample_set", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="sample build",
        artifact_kind="sample_set",
        artifact_id=sample_id,
        outputs=[output_dir],
        inputs={
            "view": view_id,
            "strategy": strategy,
            "explicit_ids": explicit_ids or [],
            "input_samples": input_sample_ids or [],
        },
    )


def preview_neighbors_fit(
    workspace: str | Path,
    neighbors_id: str,
    *,
    view_id: str,
    sample_id: str | None,
    alignment_id: str | None,
    force: bool,
) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_view(view_id)
    output_dir = artifact_dir(context, artifact_kind="neighbor_set", artifact_id=neighbors_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="neighbor_set", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="neighbors fit",
        artifact_kind="neighbor_set",
        artifact_id=neighbors_id,
        outputs=[output_dir],
        inputs={"view": view_id, "sample": sample_id, "alignment": alignment_id},
    )


def preview_cluster_fit(
    workspace: str | Path,
    cluster_id: str,
    *,
    view_id: str,
    sample_id: str | None,
    alignment_id: str | None,
    force: bool,
) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_view(view_id)
    output_dir = artifact_dir(context, artifact_kind="cluster_set", artifact_id=cluster_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="cluster_set", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="cluster fit",
        artifact_kind="cluster_set",
        artifact_id=cluster_id,
        outputs=[output_dir],
        inputs={"view": view_id, "sample": sample_id, "alignment": alignment_id},
    )


def preview_projection_fit(
    workspace: str | Path,
    projection_id: str,
    *,
    view_id: str,
    sample_id: str,
    force: bool,
) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_view(view_id)
    output_dir = artifact_dir(context, artifact_kind="projection", artifact_id=projection_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="projection", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="projection fit",
        artifact_kind="projection",
        artifact_id=projection_id,
        outputs=[output_dir],
        inputs={"view": view_id, "sample": sample_id},
    )


def preview_distance_score(
    workspace: str | Path,
    distance_id: str,
    *,
    view_id: str,
    landmark_ids: list[str],
    force: bool,
) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_view(view_id)
    for landmark_id in landmark_ids:
        context.require_landmark(landmark_id)
    output_dir = artifact_dir(context, artifact_kind="distance_set", artifact_id=distance_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="distance_set", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="distance score",
        artifact_kind="distance_set",
        artifact_id=distance_id,
        outputs=[output_dir],
        inputs={"view": view_id, "landmarks": landmark_ids},
    )


def preview_enrich_score(
    workspace: str | Path,
    enrichment_id: str,
    *,
    neighbors_id: str,
    cohort_id: str,
    landmark_ids: list[str],
    force: bool,
) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    context.require_cohort(cohort_id)
    for landmark_id in landmark_ids:
        context.require_landmark(landmark_id)
    output_dir = artifact_dir(context, artifact_kind="enrichment_set", artifact_id=enrichment_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="enrichment_set", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="enrich score",
        artifact_kind="enrichment_set",
        artifact_id=enrichment_id,
        outputs=[output_dir],
        inputs={"neighbors": neighbors_id, "cohort": cohort_id, "landmarks": landmark_ids},
    )


def preview_agreement_compare(
    workspace: str | Path,
    agreement_id: str,
    *,
    left_neighbors_id: str | None,
    right_neighbors_id: str | None,
    left_cluster_id: str | None,
    right_cluster_id: str | None,
    landmark_ids: list[str],
    force: bool,
) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    for landmark_id in landmark_ids:
        context.require_landmark(landmark_id)
    output_dir = artifact_dir(context, artifact_kind="agreement_set", artifact_id=agreement_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="agreement_set", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="agreement compare",
        artifact_kind="agreement_set",
        artifact_id=agreement_id,
        outputs=[output_dir],
        inputs={
            "left_neighbors": left_neighbors_id,
            "right_neighbors": right_neighbors_id,
            "left_clusters": left_cluster_id,
            "right_clusters": right_cluster_id,
            "landmarks": landmark_ids,
        },
    )


def preview_plot_render(
    workspace: str | Path,
    plot_id: str,
    *,
    kind: str | None,
    projection_ids: list[str],
    enrichment_id: str | None,
    distance_id: str | None,
    scalar_id: str | None,
    agreement_id: str | None,
    value_column: str | None,
    x_column: str | None,
    y_column: str | None,
    color_column: str | None,
    force: bool,
) -> dict[str, Any]:
    context, spec = resolve_plot_request(
        workspace,
        plot_id,
        kind=kind,
        projection_ids=projection_ids,
        enrichment_id=enrichment_id,
        distance_id=distance_id,
        scalar_id=scalar_id,
        agreement_id=agreement_id,
        value_column=value_column,
        x_column=x_column,
        y_column=y_column,
        color_column=color_column,
    )
    output_dir = artifact_dir(context, artifact_kind="plot", artifact_id=plot_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="plot", force=force)
    inputs = {
        "kind": spec.kind,
        "projections": spec.projection_ids,
        "enrichment": spec.enrichment_id,
        "distance": spec.distance_id,
        "scalar": spec.scalar_id,
        "agreement": spec.agreement_id,
    }
    if spec.config_id is not None:
        inputs["plot_recipe"] = spec.config_id
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="plot render",
        artifact_kind="plot",
        artifact_id=plot_id,
        outputs=[output_dir],
        inputs=inputs,
    )


def preview_export(
    workspace: str | Path,
    export_id: str,
    *,
    command: str,
    force: bool,
) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    export = context.require_export(export_id)
    output_dir = artifact_dir(context, artifact_kind="export_bundle", artifact_id=export_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="export_bundle", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command=command,
        artifact_kind="export_bundle",
        artifact_id=export_id,
        outputs=[output_dir],
        inputs={"export": export_id, "row_basis": export.row_basis},
    )


def preview_notebook_generate(workspace: str | Path, notebook_id: str, *, force: bool) -> dict[str, Any]:
    context = load_workspace_config(workspace)
    notebook = context.require_notebook(notebook_id)
    output_dir = artifact_dir(context, artifact_kind="notebook", artifact_id=notebook_id)
    _ensure_preview_targets_available([output_dir], artifact_kind="notebook", force=force)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="notebook generate",
        artifact_kind="notebook",
        artifact_id=notebook_id,
        outputs=[output_dir],
        inputs={"notebook": notebook_id, "artifacts": [artifact.id for artifact in notebook.artifacts]},
    )


def preview_recipe_run(workspace: str | Path, recipe_id: str, *, force: bool) -> dict[str, Any]:
    del force
    context = load_workspace_config(workspace)
    recipe = context.require_recipe(recipe_id)
    order = topological_step_order(recipe.steps)
    outputs: list[Path] = []
    step_results: list[dict[str, Any]] = []
    for step_id in order:
        step = next(step for step in recipe.steps if step.id == step_id)
        refs = expected_step_artifacts(step.op, step.params)
        for artifact_kind, artifact_id in refs:
            outputs.append(artifact_dir(context, artifact_kind=artifact_kind, artifact_id=artifact_id))
        step_results.append(
            {
                "step_id": step_id,
                "op": step.op,
                "expected_outputs": [f"{artifact_kind}:{artifact_id}" for artifact_kind, artifact_id in refs],
            }
        )
    deduped_outputs = list(dict.fromkeys(output.as_posix() for output in outputs))
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="recipe run",
        artifact_kind="recipe",
        artifact_id=recipe_id,
        outputs=[Path(path) for path in deduped_outputs],
        inputs={"recipe": recipe_id},
        metrics={"steps": len(order), "step_order": order, "step_results": step_results},
    )


def preview_deliverable_run(workspace: str | Path, deliverable_id: str, *, force: bool) -> dict[str, Any]:
    del force
    context = load_workspace_config(workspace)
    deliverable = context.require_deliverable(deliverable_id)
    outputs: list[Path] = []
    supported_categories = {
        "views",
        "alignments",
        "scalars",
        "samples",
        "neighbors",
        "clusters",
        "projections",
        "distances",
        "enrichments",
        "agreements",
        "plots",
        "exports",
        "reducers",
        "reduced_views",
        "snapshots",
        "notebooks",
    }
    for category, ids in deliverable.outputs.items():
        if category not in supported_categories:
            continue
        artifact_kind = artifact_kind_for_category(category)
        outputs.extend(artifact_dir(context, artifact_kind=artifact_kind, artifact_id=item_id) for item_id in ids)
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="deliverable run",
        artifact_kind="deliverable",
        artifact_id=deliverable_id,
        outputs=outputs,
        inputs={"deliverable": deliverable_id, "recipe": deliverable.recipe},
        metrics={"outputs": len(outputs)},
    )


def preview_runs_prune(
    workspace: str | Path,
    artifact_kind: str,
    artifact_id: str,
    *,
    force: bool,
) -> dict[str, Any]:
    del force
    context = load_workspace_config(workspace)
    target_dir = artifact_dir(context, artifact_kind=artifact_kind, artifact_id=artifact_id)
    if not target_dir.exists():
        raise MissingArtifactError(f"artifact run not found: {artifact_kind}:{artifact_id}")
    return _preview_payload(
        workspace_id=context.workspace_id,
        command="runs prune",
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        outputs=[],
        inputs={"artifact_kind": artifact_kind, "artifact_id": artifact_id},
    )
