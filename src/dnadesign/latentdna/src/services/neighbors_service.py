"""
Neighbor services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..neighbors.fit import fit_neighbor_artifact
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config


def _scope_input_digest_path(
    context,
    *,
    view_id: str,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[str, str, Path]:
    if alignment_id is not None:
        return "alignment_set", alignment_id, context.output_root / "alignments" / alignment_id / "rows.parquet"
    if sample_id is not None:
        return "sample_set", sample_id, context.output_root / "samples" / sample_id / "rows.parquet"
    return "view_rows", view_id, context.output_root / "views" / view_id / "rows.parquet"


def fit_neighbors(
    workspace: str | Path,
    neighbor_id: str,
    *,
    view_id: str,
    k: int,
    metric: str | None,
    backend: str | None,
    sample_id: str | None,
    alignment_id: str | None,
    seed: int | None,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    neighbor_dir = context.output_root / "neighbors" / neighbor_id
    if neighbor_dir.exists() and not force:
        raise ArtifactConflictError(f"neighbor artifact already exists: {neighbor_dir}")
    if force and neighbor_dir.exists():
        import shutil

        shutil.rmtree(neighbor_dir)

    metric_value = metric or context.config.defaults.metric
    backend_value = backend or context.config.defaults.neighbor_backend
    seed_value = seed if seed is not None else context.config.defaults.random_seed
    artifact_dir, rows, resolved_backend, approximate, scope_kind, scope_id = fit_neighbor_artifact(
        context,
        neighbor_id=neighbor_id,
        view_id=view_id,
        k=k,
        metric=metric_value,
        backend=backend_value,
        seed=seed_value,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    scope_kind_input, scope_id_input, scope_digest_path = _scope_input_digest_path(
        context,
        view_id=view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    manifest = ArtifactManifest(
        artifact_kind="neighbor_set",
        artifact_id=neighbor_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="neighbors fit",
        inputs=[
            ArtifactInput(
                kind="view_matrix",
                id=view_id,
                digest=sha256_file(context.output_root / "views" / view_id / "matrix.npy"),
            ),
            ArtifactInput(kind=scope_kind_input, id=scope_id_input, digest=sha256_file(scope_digest_path)),
        ],
        params={
            "view_id": view_id,
            "k": k,
            "metric": metric_value,
            "backend": resolved_backend,
            "requested_backend": backend_value,
            "approximate": approximate,
            "seed": seed_value,
            "scope_kind": scope_kind,
            "scope_id": scope_id,
        },
        outputs=[
            ArtifactOutput(path="indices.npy", media_type="application/x-npy"),
            ArtifactOutput(path="distances.npy", media_type="application/x-npy"),
            ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
        ],
        stats={"rows": rows, "k": k},
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="neighbors fit",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="neighbor_set",
        artifact_id=neighbor_id,
        outputs=[artifact_dir.as_posix()],
        inputs={"view": view_id, "sample": sample_id, "alignment": alignment_id},
        metrics={"rows": rows, "k": k, "backend": resolved_backend},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="neighbors_fit",
        artifact_id=neighbor_id,
    )
    return result
