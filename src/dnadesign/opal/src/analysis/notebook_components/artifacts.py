from __future__ import annotations

from typing import Any, Mapping

from ._support import compact_path, mapping, sequence


def build_notebook_artifact_garden_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build artifact-garden status lines for generated notebooks."""

    return [f"{row['field']}: `{row['value']}`" for row in build_notebook_artifact_garden_summary_rows(view_model)]


def build_notebook_artifact_garden_summary_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build artifact-garden status rows for generated notebooks."""

    audit = mapping(view_model.get("artifact_garden"))
    if not audit:
        return [
            {"field": "Artifact garden audit", "value": "unavailable"},
            {
                "field": "Next command",
                "value": "uv run opal artifacts audit -c <campaign.yaml>",
            },
        ]
    bytes_row = mapping(audit.get("bytes"))
    prune_plan = mapping(audit.get("prune_plan"))
    roots = sequence(audit.get("artifact_roots"))
    active_manifests = sequence(audit.get("active_manifests"))
    stale = sequence(audit.get("stale_artifacts"))
    local_only = "yes (local-only)" if audit.get("local_only") else "no"
    root = audit.get("root")
    return [
        {"field": "Artifact garden schema", "value": audit.get("schema_version")},
        {"field": "Root", "value": compact_path(root, max_parts=1)},
        {"field": "Local-only root", "value": local_only},
        {"field": "Artifact roots", "value": len(roots)},
        {"field": "Active manifests", "value": len(active_manifests)},
        {"field": "Stale artifacts", "value": len(stale)},
        {"field": "Artifact bytes", "value": bytes_row.get("artifact_roots", 0)},
        {"field": "Stale bytes", "value": bytes_row.get("stale_artifacts", 0)},
        {"field": "Prune plan items", "value": prune_plan.get("item_count", 0)},
        {"field": "Prune requires apply", "value": prune_plan.get("requires_apply", True)},
    ]


def build_notebook_artifact_garden_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return artifact root, stale artifact, and prune-plan rows for notebooks."""

    audit = mapping(view_model.get("artifact_garden"))
    if not audit:
        return []
    root_path = audit.get("root")
    rows: list[dict[str, Any]] = []
    for root in sequence(audit.get("artifact_roots")):
        if not isinstance(root, Mapping):
            continue
        rows.append(
            {
                "source": "artifact_root",
                "name": root.get("name"),
                "path": compact_path(root.get("path"), base=root_path),
                "exists": root.get("exists"),
                "file_count": root.get("file_count"),
                "size_bytes": root.get("size_bytes"),
                "scope": None,
                "reason": None,
            }
        )
    for artifact in sequence(audit.get("stale_artifacts")):
        if not isinstance(artifact, Mapping):
            continue
        rows.append(
            {
                "source": "stale_artifact",
                "name": None,
                "path": compact_path(artifact.get("path"), base=root_path),
                "exists": True,
                "file_count": None,
                "size_bytes": artifact.get("size_bytes"),
                "scope": artifact.get("scope"),
                "reason": artifact.get("reason"),
            }
        )
    prune_plan = mapping(audit.get("prune_plan"))
    if prune_plan:
        rows.append(
            {
                "source": "prune_plan",
                "name": "stale_artifacts_only",
                "path": "",
                "exists": None,
                "file_count": prune_plan.get("item_count"),
                "size_bytes": prune_plan.get("bytes_to_delete"),
                "scope": prune_plan.get("mode"),
                "reason": "dry-run unless --apply is explicit",
            }
        )
    return rows
