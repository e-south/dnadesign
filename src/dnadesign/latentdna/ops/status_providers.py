"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/ops/status_providers.py

Provider-owned LatentDNA status builders.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dnadesign.latentdna.src.services.workspace_snapshot_service import workspace_snapshot


def provide_latentdna_workspace_snapshot_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    del repo_root
    workspace = Path(inputs["workspace"])
    if not workspace.exists():
        return (
            "missing",
            f"LatentDNA workspace not found: {workspace.name}",
            {"workspace": str(workspace)},
        )

    snapshot = workspace_snapshot(workspace)
    workspace_id = str(snapshot.get("workspace_id") or workspace.name)
    deliverables = dict(snapshot.get("deliverables") or {})
    exports = dict(snapshot.get("exports") or {})
    deliverable_statuses = [
        str(entry.get("status") or "") for entry in deliverables.values() if isinstance(entry, Mapping)
    ]
    export_statuses = [str(entry.get("status") or "") for entry in exports.values() if isinstance(entry, Mapping)]
    state = (
        "attention"
        if any(status in {"attention", "missing", "error"} for status in [*deliverable_statuses, *export_statuses])
        else "ok"
    )
    summary = (
        f"LatentDNA workspace snapshot published for {workspace_id}"
        if state == "ok"
        else f"LatentDNA workspace snapshot published for {workspace_id} with follow-up required"
    )
    return (
        state,
        summary,
        {
            "workspace": str(workspace),
            "workspace_id": workspace_id,
            "snapshot_path": str(workspace / "outputs" / "status" / "workspace_snapshot.json"),
            "decision_ladder": list(snapshot.get("decision_ladder") or []),
            "deliverable_statuses": {
                name: dict(entry) for name, entry in deliverables.items() if isinstance(entry, Mapping)
            },
            "export_statuses": {name: dict(entry) for name, entry in exports.items() if isinstance(entry, Mapping)},
        },
    )


__all__ = ["provide_latentdna_workspace_snapshot_status"]
