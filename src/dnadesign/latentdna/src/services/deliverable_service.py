"""
Deliverable list, status, and run services for latentdna.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.deliverable import (
    ARTIFACT_REFERENCE_CATEGORIES,
    SINGULAR_REFERENCE_NAMES,
    DeliverableEntryStatus,
    DeliverableStatusResult,
)
from ..contracts.result import CommandResult
from ..runs.recorder import record_audit
from ..workspaces.loader import WorkspaceContext, load_workspace_config
from ._artifacts import artifact_dir, artifact_exists, artifact_kind_for_category
from .freshness_service import FreshnessCache, evaluate_artifact_freshness
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
) -> DeliverableEntryStatus:
    label = SINGULAR_REFERENCE_NAMES.get(category, category.rstrip("s"))
    name = f"{label}:{item_id}"
    if category in ARTIFACT_REFERENCE_CATEGORIES:
        section = _config_section(context, category)
        if section is not None and item_id not in section:
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
    context = load_workspace_config(workspace)
    deliverables = [
        {
            "deliverable_id": deliverable_id,
            "kind": deliverable.kind,
            "description": deliverable.description,
            "recipe": deliverable.recipe,
        }
        for deliverable_id, deliverable in sorted(context.config.deliverables.items())
    ]
    return {
        "schema_version": "latentdna.deliverable_list.v1",
        "workspace_id": context.workspace_id,
        "deliverables": deliverables,
    }


def deliverable_status(workspace: str | Path, deliverable_id: str) -> DeliverableStatusResult:
    context = load_workspace_config(workspace)
    deliverable = context.require_deliverable(deliverable_id)
    freshness_cache = FreshnessCache()
    checks = [
        _entry_status(context, category, item_id, freshness_cache=freshness_cache)
        for category, ids in deliverable.requires.items()
        for item_id in ids
    ]
    outputs = [
        _entry_status(context, category, item_id, freshness_cache=freshness_cache)
        for category, ids in deliverable.outputs.items()
        for item_id in ids
    ]
    return DeliverableStatusResult(
        deliverable_id=deliverable_id,
        status=_status_from_entries(checks, outputs),
        checks=checks,
        outputs=outputs,
    )


def run_deliverable(workspace: str | Path, deliverable_id: str, *, force: bool = False) -> CommandResult:
    context = load_workspace_config(workspace)
    deliverable = context.require_deliverable(deliverable_id)
    recipe_result = run_recipe(context.workspace_dir, deliverable.recipe, force=force)
    status = deliverable_status(context.workspace_dir, deliverable_id)
    output_paths = [entry.path for entry in status.outputs if entry.path is not None and entry.status == "ok"]
    result = CommandResult(
        command="deliverable run",
        workspace_id=context.workspace_id,
        status=status.status,
        artifact_kind="deliverable",
        artifact_id=deliverable_id,
        outputs=output_paths,
        inputs={"deliverable": deliverable_id, "recipe": deliverable.recipe},
        metrics={
            "executed_steps": recipe_result.metrics["executed_steps"],
            "skipped_steps": recipe_result.metrics["skipped_steps"],
            "steps": recipe_result.metrics["steps"],
            "outputs": len(output_paths),
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="deliverable_run",
        artifact_id=deliverable_id,
    )
    return result
