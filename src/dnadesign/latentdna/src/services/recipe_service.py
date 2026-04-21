"""
Recipe validation and execution services for latentdna.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from ..contracts.errors import ArtifactConflictError, ContractViolationError
from ..contracts.recipe import RecipeValidationResult, expected_step_artifacts, topological_step_order
from ..contracts.result import CommandResult
from ..runs.recorder import record_audit
from ..workspaces.loader import load_workspace_config
from ._artifacts import artifact_exists
from .agreement_service import compare_agreement
from .alignment_service import build_alignment
from .cluster_service import fit_cluster
from .distance_service import score_distance
from .enrichment_service import score_enrichment
from .export_service import export_matrix, export_table
from .freshness_service import FreshnessCache, evaluate_artifact_freshness
from .neighbors_service import fit_neighbors
from .notebook_service import generate_notebook
from .plot_service import render_plot
from .progress_service import build_run_id, heartbeat_scope, start_run_progress
from .projection_service import fit_projection
from .sample_service import build_sample
from .scalar_service import build_scalar, derive_scalar
from .snapshot_service import build_snapshot
from .view_service import derive_view, materialize_view, reduce_view


def _require_param(params: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in params:
            return params[key]
    expected = ", ".join(keys)
    raise ContractViolationError(f"recipe step is missing required params: {expected}")


def _optional_param(params: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in params:
            return params[key]
    return default


def _list_param(params: dict[str, Any], *keys: str) -> list[str]:
    value = _optional_param(params, *keys, default=[])
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _materialize_view_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    return materialize_view(
        workspace,
        str(_require_param(params, "view_id", "view")),
        allow_memory_overage=allow_memory_overage,
        force=force,
    )


def _derive_view_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return derive_view(workspace, str(_require_param(params, "view_id", "view")), force=force)


def _build_alignment_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return build_alignment(workspace, str(_require_param(params, "alignment_id", "alignment")), force=force)


def _derive_scalar_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return derive_scalar(workspace, str(_require_param(params, "scalar_id", "scalar")), force=force)


def _build_scalar_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    builder_kind = str(_require_param(params, "kind"))
    scalar_id = str(_require_param(params, "scalar_id", "scalar"))
    builder_params = {key: value for key, value in params.items() if key not in {"scalar_id", "scalar", "kind"}}
    return build_scalar(
        workspace,
        scalar_id,
        builder_kind=builder_kind,
        params=builder_params,
        force=force,
    )


def _build_sample_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return build_sample(
        workspace,
        str(_require_param(params, "sample_id", "sample")),
        view_id=(str(_require_param(params, "view_id", "view")) if "view_id" in params or "view" in params else None),
        strategy=str(_optional_param(params, "strategy", default="all")),
        n=_optional_param(params, "n", default=None),
        group_column=_optional_param(params, "group_column", "group_by", default=None),
        seed=int(_optional_param(params, "seed", default=17)),
        reference_set_id=_optional_param(params, "reference_set_id", "reference_set", default=None),
        explicit_ids=_list_param(params, "explicit_ids", "record_ids", "record_id"),
        input_sample_ids=_list_param(params, "input_sample_ids", "input_samples", "input_sample"),
        force=force,
    )


def _fit_projection_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    return fit_projection(
        workspace,
        str(_require_param(params, "view_id", "view")),
        projection_id=str(_require_param(params, "projection_id", "run_id")),
        sample_id=str(_require_param(params, "sample_id", "sample")),
        metric=_optional_param(params, "metric", default=None),
        seed=int(_optional_param(params, "seed", default=17)),
        allow_memory_overage=allow_memory_overage,
        force=force,
    )


def _fit_neighbors_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    return fit_neighbors(
        workspace,
        str(_require_param(params, "neighbor_id", "neighbors_id", "neighbors")),
        view_id=_optional_param(params, "view_id", "view", default=None),
        reduced_view_id=_optional_param(params, "reduced_view_id", "reduced_view", default=None),
        k=int(_require_param(params, "k")),
        metric=_optional_param(params, "metric", default=None),
        backend=_optional_param(params, "backend", default=None),
        sample_id=_optional_param(params, "sample_id", "sample", default=None),
        alignment_id=_optional_param(params, "alignment_id", "alignment", default=None),
        seed=_optional_param(params, "seed", default=None),
        allow_memory_overage=allow_memory_overage,
        force=force,
    )


def _generate_notebook_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return generate_notebook(
        workspace,
        str(_require_param(params, "notebook_id", "notebook")),
        force=bool(_optional_param(params, "force", default=force)),
    )


def _score_distance_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return score_distance(
        workspace,
        str(_require_param(params, "distance_id", "distance")),
        view_id=str(_require_param(params, "view_id", "view")),
        landmark_ids=_list_param(params, "landmark_ids", "landmarks", "landmark"),
        alignment_id=_optional_param(params, "alignment_id", "alignment", default=None),
        metric=_optional_param(params, "metric", default=None),
        force=force,
    )


def _score_enrichment_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return score_enrichment(
        workspace,
        str(_require_param(params, "enrichment_id", "enrichment")),
        neighbors_id=str(_require_param(params, "neighbors_id", "neighbors")),
        cohort_id=str(_require_param(params, "cohort_id", "cohort")),
        landmark_ids=_list_param(params, "landmark_ids", "landmarks", "landmark"),
        force=force,
    )


def _fit_cluster_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    return fit_cluster(
        workspace,
        str(_require_param(params, "cluster_id", "cluster")),
        view_id=_optional_param(params, "view_id", "view", default=None),
        reduced_view_id=_optional_param(params, "reduced_view_id", "reduced_view", default=None),
        method=str(_optional_param(params, "method", default="kmeans")),
        n_clusters=(
            int(_require_param(params, "n_clusters"))
            if _optional_param(params, "n_clusters", default=None) is not None
            else None
        ),
        seed=_optional_param(params, "seed", default=None),
        max_iter=int(_optional_param(params, "max_iter", default=100)),
        sample_id=_optional_param(params, "sample_id", "sample", default=None),
        alignment_id=_optional_param(params, "alignment_id", "alignment", default=None),
        neighbor_set_id=_optional_param(params, "neighbor_set_id", "neighbor_set", default=None),
        metric=_optional_param(params, "metric", default=None),
        k=int(_optional_param(params, "k", default=30)),
        resolution=float(_optional_param(params, "resolution", default=1.0)),
        allow_memory_overage=allow_memory_overage,
        force=force,
    )


def _compare_agreement_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return compare_agreement(
        workspace,
        str(_require_param(params, "agreement_id", "agreement")),
        left_neighbors_id=_optional_param(params, "left_neighbors_id", "left_neighbors", default=None),
        right_neighbors_id=_optional_param(params, "right_neighbors_id", "right_neighbors", default=None),
        left_cluster_id=_optional_param(params, "left_cluster_id", "left_clusters", default=None),
        right_cluster_id=_optional_param(params, "right_cluster_id", "right_clusters", default=None),
        landmark_ids=_list_param(params, "landmark_ids", "landmarks", "landmark"),
        force=force,
    )


def _render_plot_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return render_plot(
        workspace,
        str(_require_param(params, "plot_id", "plot")),
        kind=(_optional_param(params, "kind", default=None)),
        projection_ids=_list_param(params, "projection_ids", "projections", "projection"),
        panel_titles=_list_param(params, "panel_titles", "panel_title"),
        enrichment_id=_optional_param(params, "enrichment_id", "enrichment", default=None),
        distance_id=_optional_param(params, "distance_id", "distance", default=None),
        scalar_id=_optional_param(params, "scalar_id", "scalar", default=None),
        scalar_ids=_list_param(params, "scalar_ids", "scalars", "scalar_panel"),
        agreement_id=_optional_param(params, "agreement_id", "agreement", default=None),
        agreement_ids=_list_param(params, "agreement_ids", "agreements", "agreement_panel"),
        reducer_id=_optional_param(params, "reducer_id", "reducer", default=None),
        left_cluster_id=_optional_param(params, "left_cluster_id", "left_cluster", default=None),
        right_cluster_id=_optional_param(params, "right_cluster_id", "right_cluster", default=None),
        value_column=_optional_param(params, "value_column", default=None),
        x_column=_optional_param(params, "x_column", default=None),
        y_column=_optional_param(params, "y_column", default=None),
        color_column=_optional_param(params, "color_column", default=None),
        shape_column=_optional_param(params, "shape_column", default=None),
        render_mode=_optional_param(params, "render_mode", default=None),
        label_column=_optional_param(params, "label_column", default=None),
        label_values=_list_param(params, "label_values", "label_value"),
        force=force,
    )


def _reduce_view_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    return reduce_view(
        workspace,
        str(_require_param(params, "view_id", "view")),
        reducer_id=str(_require_param(params, "reducer_id", "run_id")),
        dims=int(_require_param(params, "dims")),
        sample_id=_optional_param(params, "sample_id", "sample", default=None),
        alignment_id=_optional_param(params, "alignment_id", "alignment", default=None),
        reduced_view_id=_optional_param(params, "reduced_view_id", default=None),
        allow_memory_overage=allow_memory_overage,
        force=force,
    )


def _export_matrix_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    return export_matrix(
        workspace,
        str(_require_param(params, "export_id", "export")),
        allow_memory_overage=allow_memory_overage,
        force=force,
    )


def _export_table_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    return export_table(
        workspace,
        str(_require_param(params, "export_id", "export")),
        allow_memory_overage=allow_memory_overage,
        force=force,
    )


def _build_snapshot_step(
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    del allow_memory_overage
    return build_snapshot(
        workspace,
        str(_require_param(params, "snapshot_id", "snapshot")),
        source_id=str(_require_param(params, "source_id", "source")),
        force=force,
    )


STEP_EXECUTORS: dict[str, Callable[..., CommandResult]] = {
    "agreement.compare": _compare_agreement_step,
    "alignment.build": _build_alignment_step,
    "cluster.fit": _fit_cluster_step,
    "distance.score": _score_distance_step,
    "enrich.score": _score_enrichment_step,
    "export.matrix": _export_matrix_step,
    "export.table": _export_table_step,
    "neighbors.fit": _fit_neighbors_step,
    "notebook.generate": _generate_notebook_step,
    "plot.render": _render_plot_step,
    "projection.fit": _fit_projection_step,
    "sample.build": _build_sample_step,
    "scalar.build": _build_scalar_step,
    "scalar.derive": _derive_scalar_step,
    "snapshot.build": _build_snapshot_step,
    "view.derive": _derive_view_step,
    "view.materialize": _materialize_view_step,
    "view.reduce": _reduce_view_step,
}


def _invoke_executor(
    executor: Callable[..., CommandResult],
    workspace: str | Path,
    params: dict[str, Any],
    *,
    force: bool,
    allow_memory_overage: bool,
) -> CommandResult:
    if "allow_memory_overage" in inspect.signature(executor).parameters:
        return executor(workspace, params, force=force, allow_memory_overage=allow_memory_overage)
    return executor(workspace, params, force=force)


def validate_recipe(workspace: str | Path, recipe_id: str) -> RecipeValidationResult:
    context = load_workspace_config(workspace)
    recipe = context.require_recipe(recipe_id)
    return RecipeValidationResult(
        workspace_id=context.workspace_id,
        recipe_id=recipe_id,
        status="ok",
        step_order=topological_step_order(recipe.steps),
    )


def run_recipe(
    workspace: str | Path,
    recipe_id: str,
    *,
    force: bool = False,
    allow_memory_overage: bool = False,
    refresh_catalog: bool = True,
    event_sink: Callable[[dict[str, object]], None] | None = None,
) -> CommandResult:
    context = load_workspace_config(workspace)
    run_id = build_run_id(kind="recipe", name=recipe_id)
    recipe = context.require_recipe(recipe_id)
    steps_by_id = {step.id: step for step in recipe.steps}
    order = topological_step_order(recipe.steps)
    outputs: list[str] = []
    executed_steps = 0
    rebuilt_steps = 0
    skipped_steps = 0
    warnings: list[str] = []
    step_summaries: list[dict[str, Any]] = []
    freshness_cache = FreshnessCache()
    progress = start_run_progress(
        context,
        command="recipe run",
        run_id=run_id,
        current_stage=recipe_id,
        expected_steps=len(order),
        event_sink=event_sink,
    )

    try:
        for step_id in order:
            step = steps_by_id[step_id]
            progress.step_started(current_step=step_id)
            try:
                refs = expected_step_artifacts(step.op, step.params)
            except ValueError as exc:
                raise ContractViolationError(str(exc)) from exc
            existence = [
                artifact_exists(context, artifact_kind=kind, artifact_id=artifact_id) for kind, artifact_id in refs
            ]
            step_force = force or bool(step.params.get("force", False))
            rebuild_reasons: list[str] = []
            if not step_force and existence and all(existence):
                freshness = [
                    evaluate_artifact_freshness(
                        context,
                        artifact_kind=kind,
                        artifact_id=artifact_id,
                        cache=freshness_cache,
                    )
                    for kind, artifact_id in refs
                ]
                if all(entry["status"] == "ok" for entry in freshness):
                    skipped_steps += 1
                    step_summaries.append({"step_id": step_id, "op": step.op, "status": "skipped"})
                    progress.step_finished(current_step=step_id, status="skipped")
                    continue
                step_force = True
                rebuild_reasons = [str(entry.get("reason") or "freshness requires attention") for entry in freshness]
            if not step_force and any(existence) and not all(existence):
                raise ArtifactConflictError(
                    f"recipe step {step_id} has partial existing outputs; rerun with --force to rebuild"
                )

            executor = STEP_EXECUTORS[step.op]
            if rebuild_reasons:
                progress.step_progress(
                    current_step=step_id,
                    message="rebuild required because upstream freshness needs attention",
                )
            with heartbeat_scope(progress, current_step=step_id):
                step_result = _invoke_executor(
                    executor,
                    context.workspace_dir,
                    dict(step.params),
                    force=step_force,
                    allow_memory_overage=allow_memory_overage,
                )
            executed_steps += 1
            rebuilt = bool(rebuild_reasons)
            if rebuilt:
                rebuilt_steps += 1
            outputs.extend(step_result.outputs)
            summary_status = "attention" if step_result.status == "attention" else ("rebuilt" if rebuilt else "ok")
            summary = {
                "step_id": step_id,
                "op": step.op,
                "status": summary_status,
                "artifact_kind": step_result.artifact_kind,
                "artifact_id": step_result.artifact_id,
            }
            if rebuilt:
                summary["rebuild_reasons"] = rebuild_reasons
            if step_result.warnings:
                step_warnings = [f"{step_id}: {warning}" for warning in step_result.warnings]
                warnings.extend(step_warnings)
                summary["warnings"] = step_result.warnings
                for warning in step_warnings:
                    progress.warning(warning)
            step_summaries.append(summary)
            progress.step_finished(current_step=step_id, status=str(summary_status))
    except Exception as exc:
        progress.fail(current_step=step_id if "step_id" in locals() else None, message=str(exc))
        raise

    result = CommandResult(
        command="recipe run",
        workspace_id=context.workspace_id,
        status="attention" if warnings else "ok",
        run_id=run_id,
        artifact_kind="recipe",
        artifact_id=recipe_id,
        outputs=outputs,
        inputs={"recipe": recipe_id},
        warnings=warnings,
        metrics={
            "steps": len(order),
            "executed_steps": executed_steps,
            "rebuilt_steps": rebuilt_steps,
            "skipped_steps": skipped_steps,
            "step_order": order,
            "step_results": step_summaries,
            "finished_at": datetime.now(UTC).isoformat(),
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="recipe_run",
        artifact_id=recipe_id,
    )
    progress.succeed()
    if refresh_catalog:
        from .catalog_service import workspace_catalog_from_context

        workspace_catalog_from_context(context)
    return result
