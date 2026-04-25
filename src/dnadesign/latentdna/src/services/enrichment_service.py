"""
Enrichment services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..enrichments.score import score_enrichment_artifact
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config


def score_enrichment(
    workspace: str | Path,
    enrichment_id: str,
    *,
    neighbors_id: str,
    cohort_id: str,
    landmark_ids: list[str],
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    enrichment_dir = context.output_root / "enrichments" / enrichment_id
    if enrichment_dir.exists() and not force:
        raise ArtifactConflictError(f"enrichment artifact already exists: {enrichment_dir}")
    if force and enrichment_dir.exists():
        import shutil

        shutil.rmtree(enrichment_dir)

    artifact_dir, rows, summary = score_enrichment_artifact(
        context,
        enrichment_id=enrichment_id,
        neighbors_id=neighbors_id,
        cohort_id=cohort_id,
        landmark_ids=landmark_ids,
    )
    manifest = ArtifactManifest(
        artifact_kind="enrichment_set",
        artifact_id=enrichment_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="enrich score",
        inputs=[
            ArtifactInput(
                kind="neighbor_set",
                id=neighbors_id,
                digest=sha256_file(context.output_root / "neighbors" / neighbors_id / "indices.npy"),
            ),
            ArtifactInput(
                kind="neighbor_rows",
                id=neighbors_id,
                digest=sha256_file(context.output_root / "neighbors" / neighbors_id / "rows.parquet"),
            ),
        ],
        params={
            "method": summary["method"],
            "neighbors_id": neighbors_id,
            "view_id": summary["view_id"],
            "cohort_id": cohort_id,
            "cohort_column": summary["cohort_column"],
            "landmarks": landmark_ids,
            "k": summary["k"],
        },
        outputs=[
            ArtifactOutput(path="table.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="summary.json", media_type="application/json"),
        ],
        stats={
            "rows": rows,
            "k": summary["k"],
            "landmarks": len(landmark_ids),
            "cohorts": len(summary["cohort_values"]),
        },
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="enrich score",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="enrichment_set",
        artifact_id=enrichment_id,
        outputs=[artifact_dir.as_posix()],
        inputs={"neighbors": neighbors_id, "cohort": cohort_id, "landmarks": landmark_ids},
        metrics={"rows": rows, "k": summary["k"], "landmarks": len(landmark_ids)},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="enrich_score",
        artifact_id=enrichment_id,
    )
    return result
