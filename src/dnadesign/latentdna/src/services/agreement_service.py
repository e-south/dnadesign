"""
Agreement services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..agreements.compare import compare_agreement_artifact
from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..runs.recorder import record_audit
from ..sources.resolver import resolve_source
from ..version import __version__
from ..workspaces.loader import load_workspace_config


def compare_agreement(
    workspace: str | Path,
    agreement_id: str,
    *,
    left_neighbors_id: str | None = None,
    right_neighbors_id: str | None = None,
    left_cluster_id: str | None = None,
    right_cluster_id: str | None = None,
    landmark_ids: list[str] | None = None,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    agreement_dir = context.output_root / "agreements" / agreement_id
    if agreement_dir.exists() and not force:
        raise ArtifactConflictError(f"agreement artifact already exists: {agreement_dir}")
    if force and agreement_dir.exists():
        import shutil

        shutil.rmtree(agreement_dir)

    artifact_dir, rows, summary = compare_agreement_artifact(
        context,
        agreement_id=agreement_id,
        left_neighbors_id=left_neighbors_id,
        right_neighbors_id=right_neighbors_id,
        left_cluster_id=left_cluster_id,
        right_cluster_id=right_cluster_id,
        landmark_ids=landmark_ids,
    )
    inputs: list[ArtifactInput] = []
    params: dict[str, object] = {
        "methods": summary["methods"],
        "landmarks": landmark_ids or [],
    }
    metrics: dict[str, object] = {"rows": rows, "methods": summary["methods"]}

    if left_neighbors_id is not None and right_neighbors_id is not None:
        inputs.extend(
            [
                ArtifactInput(
                    kind="neighbor_set",
                    id=left_neighbors_id,
                    digest=sha256_file(context.output_root / "neighbors" / left_neighbors_id / "indices.npy"),
                ),
                ArtifactInput(
                    kind="neighbor_rows",
                    id=left_neighbors_id,
                    digest=sha256_file(context.output_root / "neighbors" / left_neighbors_id / "rows.parquet"),
                ),
                ArtifactInput(
                    kind="neighbor_set",
                    id=right_neighbors_id,
                    digest=sha256_file(context.output_root / "neighbors" / right_neighbors_id / "indices.npy"),
                ),
                ArtifactInput(
                    kind="neighbor_rows",
                    id=right_neighbors_id,
                    digest=sha256_file(context.output_root / "neighbors" / right_neighbors_id / "rows.parquet"),
                ),
            ]
        )
        params["left_neighbors"] = left_neighbors_id
        params["right_neighbors"] = right_neighbors_id
        if "knn_overlap" in summary:
            metrics["k"] = int(summary["knn_overlap"]["k"])
            metrics["mean_overlap_fraction"] = float(summary["knn_overlap"]["mean_overlap_fraction"])

    if left_cluster_id is not None and right_cluster_id is not None:
        inputs.extend(
            [
                ArtifactInput(
                    kind="cluster_set",
                    id=left_cluster_id,
                    digest=sha256_file(context.output_root / "clusters" / left_cluster_id / "assignments.parquet"),
                ),
                ArtifactInput(
                    kind="cluster_set",
                    id=right_cluster_id,
                    digest=sha256_file(context.output_root / "clusters" / right_cluster_id / "assignments.parquet"),
                ),
            ]
        )
        params["left_clusters"] = left_cluster_id
        params["right_clusters"] = right_cluster_id
        if "cluster_agreement" in summary:
            metrics["adjusted_rand_index"] = float(summary["cluster_agreement"]["adjusted_rand_index"])
            metrics["normalized_mutual_information"] = float(
                summary["cluster_agreement"]["normalized_mutual_information"]
            )

    if landmark_ids:
        seen_sources: set[str] = set()
        for landmark_id in landmark_ids:
            landmark = context.require_landmark(landmark_id)
            if landmark.source in seen_sources:
                continue
            seen_sources.add(landmark.source)
            source = context.require_source(landmark.source)
            resolved_source = resolve_source(landmark.source, source, workspace_dir=context.workspace_dir)
            if resolved_source.records_path is not None:
                inputs.append(
                    ArtifactInput(
                        kind="landmark_source",
                        id=landmark.source,
                        digest=sha256_file(resolved_source.records_path),
                    )
                )

    manifest = ArtifactManifest(
        artifact_kind="agreement_set",
        artifact_id=agreement_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="agreement compare",
        inputs=inputs,
        params=params,
        outputs=[
            ArtifactOutput(path="table.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="summary.json", media_type="application/json"),
        ],
        stats=metrics,
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="agreement compare",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="agreement_set",
        artifact_id=agreement_id,
        outputs=[artifact_dir.as_posix()],
        inputs={
            "left_neighbors": left_neighbors_id,
            "right_neighbors": right_neighbors_id,
            "left_clusters": left_cluster_id,
            "right_clusters": right_cluster_id,
            "landmarks": landmark_ids or [],
        },
        metrics=metrics,
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="agreement_compare",
        artifact_id=agreement_id,
    )
    return result
