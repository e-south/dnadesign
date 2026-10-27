"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/services/deliverable_service.py

Deliverable list, status, and run services for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..contracts.deliverable import (
    ARTIFACT_REFERENCE_CATEGORIES,
    SINGULAR_REFERENCE_NAMES,
    DeliverableEntryStatus,
    DeliverableStatusResult,
)
from ..contracts.recipe import expected_step_artifacts
from ..contracts.result import CommandResult
from ..runs.recorder import record_audit
from ..studies.docs_refs import resolve_docs_ref
from ..workspaces.loader import WorkspaceContext, load_workspace_config
from ._artifacts import artifact_dir, artifact_exists, artifact_kind_for_category
from ._status import merge_statuses
from .freshness_service import FreshnessCache, evaluate_artifact_freshness
from .progress_service import build_run_id, heartbeat_scope, start_run_progress
from .recipe_service import run_recipe


def _config_section(context: WorkspaceContext, category: str) -> dict[str, object] | None:
    return {
        "cohorts": context.config.cohorts,
        "landmarks": context.config.landmarks,
        "notebooks": context.config.notebooks,
        "recipes": context.config.recipes,
        "sources": context.config.sources,
        "views": context.config.views,
        "alignments": context.config.alignments,
        "scalars": context.config.scalars,
        "exports": context.config.exports,
    }.get(category)


def _artifact_reason(category: str, item_id: str) -> str:
    singular = SINGULAR_REFERENCE_NAMES.get(category, category.rstrip("s"))
    if category == "views":
        return f"{singular} artifact not materialized: {item_id}"
    return f"{singular} artifact is missing: {item_id}"


def _entry_status(
    context: WorkspaceContext,
    category: str,
    item_id: str,
    *,
    freshness_cache: FreshnessCache | None = None,
    allow_recipe_output: bool = False,
) -> DeliverableEntryStatus:
    label = SINGULAR_REFERENCE_NAMES.get(category, category.rstrip("s"))
    name = f"{label}:{item_id}"
    if category in ARTIFACT_REFERENCE_CATEGORIES:
        section = _config_section(context, category)
        if section is not None and item_id not in section and not allow_recipe_output:
            return DeliverableEntryStatus(
                name=name,
                status="missing",
                reason=f"{label} is not declared in workspace config",
            )
        artifact_kind = artifact_kind_for_category(category)
        if not artifact_exists(context, artifact_kind=artifact_kind, artifact_id=item_id):
            return DeliverableEntryStatus(name=name, status="missing", reason=_artifact_reason(category, item_id))
        freshness = evaluate_artifact_freshness(
            context,
            artifact_kind=artifact_kind,
            artifact_id=item_id,
            cache=freshness_cache,
        )
        if freshness["status"] != "ok":
            return DeliverableEntryStatus(
                name=name,
                status="attention",
                reason=str(freshness.get("reason") or f"{label} freshness requires attention"),
                path=artifact_dir(context, artifact_kind=artifact_kind, artifact_id=item_id).as_posix(),
            )
        return DeliverableEntryStatus(
            name=name,
            status="ok",
            path=artifact_dir(context, artifact_kind=artifact_kind, artifact_id=item_id).as_posix(),
        )

    section = _config_section(context, category)
    if section is None or item_id not in section:
        return DeliverableEntryStatus(
            name=name,
            status="missing",
            reason=f"{label} is not declared in workspace config",
        )
    return DeliverableEntryStatus(name=name, status="ok")


def _recipe_expected_outputs(context: WorkspaceContext, recipe_id: str) -> set[tuple[str, str]]:
    recipe = context.require_recipe(recipe_id)
    expected_outputs: set[tuple[str, str]] = set()
    for step in recipe.steps:
        expected_outputs.update(expected_step_artifacts(step.op, step.params))
    return expected_outputs


def _status_from_entries(checks: list[DeliverableEntryStatus], outputs: list[DeliverableEntryStatus]) -> str:
    entries = [*checks, *outputs]
    if any(entry.status == "error" for entry in entries):
        return "error"
    if entries and all(entry.status == "ok" for entry in entries):
        return "ok"
    if any(entry.status == "attention" for entry in entries):
        return "attention"
    if any(entry.status == "ok" for entry in outputs):
        return "attention"
    if outputs:
        return "missing"
    if any(entry.status != "ok" for entry in checks):
        return "missing"
    return "ok"


def list_deliverables(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace, validate_plot_semantics=False)
    deliverables = [
        {
            "deliverable_id": deliverable_id,
            "title": deliverable.title,
            "section": deliverable.section,
            "question": deliverable.question,
            "summary": deliverable.summary,
            "recipe": deliverable.recipe,
            "docs_refs": list(deliverable.docs_refs),
            "acceptance_checks": [item.model_dump(mode="json") for item in deliverable.acceptance_checks],
        }
        for deliverable_id, deliverable in sorted(context.config.deliverables.items())
    ]
    return {
        "schema_version": "latentdna.deliverable_list.v1",
        "workspace_id": context.workspace_id,
        "deliverables": deliverables,
    }


def deliverable_status(
    workspace: str | Path,
    deliverable_id: str,
    *,
    freshness_cache: FreshnessCache | None = None,
) -> DeliverableStatusResult:
    context = load_workspace_config(workspace, validate_plot_semantics=False)
    return deliverable_status_from_context(context, deliverable_id, freshness_cache=freshness_cache)


def deliverable_status_from_context(
    context: WorkspaceContext,
    deliverable_id: str,
    *,
    freshness_cache: FreshnessCache | None = None,
) -> DeliverableStatusResult:
    deliverable = context.require_deliverable(deliverable_id)
    if freshness_cache is None:
        freshness_cache = FreshnessCache()
    expected_outputs = _recipe_expected_outputs(context, deliverable.recipe)
    checks = [
        _entry_status(context, category, item_id, freshness_cache=freshness_cache)
        for category, ids in deliverable.requires.items()
        for item_id in ids
    ]
    outputs = [
        _entry_status(
            context,
            category,
            item_id,
            freshness_cache=freshness_cache,
            allow_recipe_output=(ARTIFACT_REFERENCE_CATEGORIES.get(category), item_id) in expected_outputs,
        )
        for category, ids in deliverable.outputs.items()
        for item_id in ids
    ]
    acceptance_results: list[dict[str, object]] = []
    acceptance_statuses: list[str] = []
    warnings: list[str] = []
    for acceptance_check in deliverable.acceptance_checks:
        result = _evaluate_acceptance_check(context, deliverable, acceptance_check)
        acceptance_results.append(result)
        result_status = str(result["status"])
        if result_status != "ok":
            acceptance_statuses.append(result_status)
            warnings.append(str(result["reason"]))
    base_status = _status_from_entries(checks, outputs)
    return DeliverableStatusResult(
        deliverable_id=deliverable_id,
        title=deliverable.title,
        section=deliverable.section,
        question=deliverable.question,
        summary=deliverable.summary,
        status=merge_statuses(base_status, *acceptance_statuses),
        checks=checks,
        outputs=outputs,
        docs_refs=[resolve_docs_ref(context, docs_ref) for docs_ref in deliverable.docs_refs],
        acceptance_checks=acceptance_results,
        warnings=warnings,
    )


def _evaluate_acceptance_check(context: WorkspaceContext, deliverable, acceptance_check) -> dict[str, object]:
    result: dict[str, object] = {
        "kind": acceptance_check.kind,
        "value": acceptance_check.value,
        "status": "ok",
    }
    if acceptance_check.kind == "required_plot_kind":
        plot_ids = deliverable.outputs.get("plots", [])
        mismatched = [
            plot_id for plot_id in plot_ids if context.require_plot(plot_id).kind != str(acceptance_check.value)
        ]
        if mismatched:
            result["status"] = "error"
            result["reason"] = f"plots do not match required kind {acceptance_check.value!r}: {mismatched}"
        return result
    if acceptance_check.kind == "required_reference_set":
        plot_ids = deliverable.outputs.get("plots", [])
        mismatched = []
        for plot_id in plot_ids:
            annotation = getattr(context.require_plot(plot_id), "annotation", None)
            if annotation is None or annotation.reference_set != str(acceptance_check.value):
                mismatched.append(plot_id)
        if mismatched:
            result["status"] = "error"
            result["reason"] = f"plots do not declare reference_set {acceptance_check.value!r}: {mismatched}"
        return result
    if acceptance_check.kind == "require_reference_set_in_every_panel":
        if bool(acceptance_check.value) is False:
            return result
        incomplete = []
        for plot_id in deliverable.outputs.get("plots", []):
            manifest_path = artifact_dir(context, artifact_kind="plot", artifact_id=plot_id) / "manifest.json"
            if not manifest_path.is_file():
                incomplete.append(plot_id)
                continue
            manifest = context.read_manifest(manifest_path)
            if not bool(manifest.get("stats", {}).get("reference_set_complete")):
                incomplete.append(plot_id)
        if incomplete:
            result["status"] = "attention"
            result["reason"] = f"reference_set completeness not satisfied for plots: {incomplete}"
        return result
    result["status"] = "error"
    result["reason"] = f"unsupported acceptance check: {acceptance_check.kind}"
    return result


def run_deliverable(
    workspace: str | Path,
    deliverable_id: str,
    *,
    force: bool = False,
    allow_memory_overage: bool = False,
    event_sink: Callable[[dict[str, object]], None] | None = None,
) -> CommandResult:
    context = load_workspace_config(workspace)
    deliverable = context.require_deliverable(deliverable_id)
    run_id = build_run_id(kind="deliverable", name=deliverable_id)
    progress = start_run_progress(
        context,
        command="deliverable run",
        run_id=run_id,
        current_stage=deliverable_id,
        expected_steps=1,
        event_sink=event_sink,
    )
    try:
        progress.step_started(current_step=deliverable.recipe)
        with heartbeat_scope(progress, current_step=deliverable.recipe):
            recipe_result = run_recipe(
                context.workspace_dir,
                deliverable.recipe,
                force=force,
                allow_memory_overage=allow_memory_overage,
                refresh_catalog=False,
                event_sink=event_sink,
            )
        progress.step_finished(current_step=deliverable.recipe, status=recipe_result.status)
        status = deliverable_status(context.workspace_dir, deliverable_id)
    except Exception as exc:
        progress.fail(current_step=deliverable.recipe, message=str(exc))
        raise
    output_paths = [entry.path for entry in status.outputs if entry.path is not None and entry.status == "ok"]
    warnings = [*recipe_result.warnings, *status.warnings]
    result_status = merge_statuses(recipe_result.status, status.status)
    result = CommandResult(
        command="deliverable run",
        workspace_id=context.workspace_id,
        status=result_status,
        run_id=run_id,
        artifact_kind="deliverable",
        artifact_id=deliverable_id,
        outputs=output_paths,
        inputs={"deliverable": deliverable_id, "recipe": deliverable.recipe},
        warnings=warnings,
        metrics={
            "executed_steps": recipe_result.metrics["executed_steps"],
            "skipped_steps": recipe_result.metrics["skipped_steps"],
            "steps": recipe_result.metrics["steps"],
            "outputs": len(output_paths),
            "recipe_warnings": len(recipe_result.warnings),
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="deliverable_run",
        artifact_id=deliverable_id,
    )
    progress.succeed()
    from .catalog_service import workspace_catalog_from_context

    workspace_catalog_from_context(context)
    return result
