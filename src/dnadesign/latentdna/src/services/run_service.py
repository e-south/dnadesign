"""
Artifact inventory, show, and prune services for latentdna.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..contracts.deliverable import ARTIFACT_REFERENCE_CATEGORIES
from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..contracts.result import CommandResult
from ..runs.recorder import record_audit
from ..workspaces.loader import WorkspaceContext, load_workspace_config
from ._artifacts import ARTIFACT_KIND_DIRS, artifact_dir, artifact_exists, artifact_manifest_path
from .freshness_service import FreshnessCache, evaluate_artifact_freshness


def artifact_inventory(
    context: WorkspaceContext,
    *,
    freshness_cache: FreshnessCache | None = None,
) -> list[dict[str, object]]:
    cache = freshness_cache or FreshnessCache()
    items: list[dict[str, object]] = []
    for artifact_kind, relative_dir in sorted(ARTIFACT_KIND_DIRS.items()):
        base_dir = context.output_root / relative_dir
        if not base_dir.is_dir():
            continue
        for candidate in sorted(base_dir.iterdir()):
            manifest_path = candidate / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = context.read_manifest(manifest_path)
            freshness = evaluate_artifact_freshness(
                context,
                artifact_kind=artifact_kind,
                artifact_id=str(manifest["artifact_id"]),
                cache=cache,
            )
            status = str(manifest["status"])
            if status == "ok":
                status = str(freshness["status"])
            items.append(
                {
                    "artifact_kind": artifact_kind,
                    "artifact_id": manifest["artifact_id"],
                    "status": status,
                    "command": manifest["command"],
                    "created_at": manifest["created_at"],
                    "path": candidate.as_posix(),
                    "reason": freshness.get("reason"),
                    "freshness_known": freshness.get("known"),
                }
            )
    items.sort(
        key=lambda item: (str(item["created_at"]), str(item["artifact_kind"]), str(item["artifact_id"])),
        reverse=True,
    )
    return items


def _deliverable_references(context: WorkspaceContext, *, artifact_kind: str, artifact_id: str) -> list[dict[str, str]]:
    categories = [category for category, kind in ARTIFACT_REFERENCE_CATEGORIES.items() if kind == artifact_kind]
    references: list[dict[str, str]] = []
    for deliverable_id, deliverable in context.config.deliverables.items():
        for section_name, section in [("requires", deliverable.requires), ("outputs", deliverable.outputs)]:
            for category in categories:
                if artifact_id in section.get(category, []):
                    references.append(
                        {
                            "deliverable_id": deliverable_id,
                            "section": section_name,
                            "category": category,
                        }
                    )
    return references


def list_runs(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    return {
        "schema_version": "latentdna.run_list.v1",
        "workspace_id": context.workspace_id,
        "runs": artifact_inventory(context),
    }


def show_run(workspace: str | Path, artifact_kind: str, artifact_id: str) -> dict[str, object]:
    context = load_workspace_config(workspace)
    if not artifact_exists(context, artifact_kind=artifact_kind, artifact_id=artifact_id):
        raise MissingArtifactError(f"artifact run not found: {artifact_kind}:{artifact_id}")
    manifest_path = artifact_manifest_path(context, artifact_kind=artifact_kind, artifact_id=artifact_id)
    freshness = evaluate_artifact_freshness(context, artifact_kind=artifact_kind, artifact_id=artifact_id)
    return {
        "schema_version": "latentdna.run_show.v1",
        "workspace_id": context.workspace_id,
        "path": artifact_dir(context, artifact_kind=artifact_kind, artifact_id=artifact_id).as_posix(),
        "artifact": context.read_manifest(manifest_path),
        "freshness": freshness,
        "deliverable_references": _deliverable_references(
            context,
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
        ),
    }


def prune_run(
    workspace: str | Path,
    artifact_kind: str,
    artifact_id: str,
    *,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    if not artifact_exists(context, artifact_kind=artifact_kind, artifact_id=artifact_id):
        raise MissingArtifactError(f"artifact run not found: {artifact_kind}:{artifact_id}")
    references = _deliverable_references(context, artifact_kind=artifact_kind, artifact_id=artifact_id)
    if references and not force:
        ref = references[0]
        raise ContractViolationError(
            "artifact is still referenced by deliverables; rerun with --force to prune "
            f"{artifact_kind}:{artifact_id} (for example {ref['deliverable_id']}:{ref['section']}:{ref['category']})"
        )
    target_dir = artifact_dir(context, artifact_kind=artifact_kind, artifact_id=artifact_id)
    shutil.rmtree(target_dir)
    result = CommandResult(
        command="runs prune",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        outputs=[],
        inputs={"artifact_kind": artifact_kind, "artifact_id": artifact_id},
        metrics={"deliverable_references": len(references)},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="runs_prune",
        artifact_id=artifact_id,
    )
    return result
