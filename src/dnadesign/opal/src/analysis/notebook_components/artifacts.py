from __future__ import annotations

from typing import Any, Mapping

from ._support import mapping, sequence


def build_notebook_artifact_garden_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build artifact-garden status lines for generated notebooks."""

    audit = mapping(view_model.get("artifact_garden"))
    if not audit:
        return [
            "### Artifacts",
            "",
            "- Artifact garden audit: `unavailable`",
            "- Run `uv run opal artifacts audit -c <campaign.yaml>` for a manifest-authoritative artifact inventory.",
        ]
    bytes_row = mapping(audit.get("bytes"))
    prune_plan = mapping(audit.get("prune_plan"))
    roots = sequence(audit.get("artifact_roots"))
    active_manifests = sequence(audit.get("active_manifests"))
    stale = sequence(audit.get("stale_artifacts"))
    local_only = "yes (local-only)" if audit.get("local_only") else "no"
    return [
        "### Artifacts",
        "",
        f"- Artifact garden schema: `{audit.get('schema_version')}`",
        f"- Root: `{audit.get('root')}`",
        f"- Local-only root: `{local_only}`",
        f"- Artifact roots: `{len(roots)}`",
        f"- Active manifests: `{len(active_manifests)}`",
        f"- Stale artifacts: `{len(stale)}`",
        f"- Artifact bytes: `{bytes_row.get('artifact_roots', 0)}`",
        f"- Stale bytes: `{bytes_row.get('stale_artifacts', 0)}`",
        f"- Prune plan items: `{prune_plan.get('item_count', 0)}`",
        f"- Prune requires apply: `{prune_plan.get('requires_apply', True)}`",
    ]


def build_notebook_artifact_garden_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return artifact root, stale artifact, and prune-plan rows for notebooks."""

    audit = mapping(view_model.get("artifact_garden"))
    if not audit:
        return []
    rows: list[dict[str, Any]] = []
    for root in sequence(audit.get("artifact_roots")):
        if not isinstance(root, Mapping):
            continue
        rows.append(
            {
                "source": "artifact_root",
                "name": root.get("name"),
                "path": root.get("path"),
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
                "path": artifact.get("path"),
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
