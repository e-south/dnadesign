"""
Scalar services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError, MissingArtifactError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..runs.recorder import record_audit
from ..scalars.derive import derive_scalar_artifact
from ..version import __version__
from ..workspaces.loader import load_workspace_config


def _resolve_table_source(context, source_id: str) -> tuple[str, Path]:
    scalar_source_path = context.output_root / "scalars" / source_id / "table.parquet"
    distance_source_path = context.output_root / "distances" / source_id / "table.parquet"
    if scalar_source_path.exists():
        return "scalar_table", scalar_source_path
    if distance_source_path.exists():
        return "distance_set", distance_source_path
    raise MissingArtifactError(f"scalar source table not found for {source_id!r}")


def derive_scalar(workspace: str | Path, scalar_id: str, *, force: bool = False) -> CommandResult:
    context = load_workspace_config(workspace)
    scalar = context.require_scalar(scalar_id)
    scalar_dir = context.output_root / "scalars" / scalar_id
    if scalar_dir.exists() and not force:
        raise ArtifactConflictError(f"scalar artifact already exists: {scalar_dir}")
    if force and scalar_dir.exists():
        import shutil

        shutil.rmtree(scalar_dir)

    artifact_dir, rows, columns = derive_scalar_artifact(context, scalar_id=scalar_id)
    if scalar.derive.kind == "vector_norm":
        input_entries = [
            ArtifactInput(
                kind="view_matrix",
                id=scalar.derive.view,
                digest=sha256_file(context.output_root / "views" / scalar.derive.view / "matrix.npy"),
            )
        ]
        input_payload = {"view": scalar.derive.view}
        params = {
            "derive_kind": scalar.derive.kind,
            "norm": scalar.derive.norm,
            "output_column": scalar.derive.output_column or scalar_id,
        }
    elif scalar.derive.kind in {"column_expression", "select_columns", "rename_columns"}:
        input_kind, source_path = _resolve_table_source(context, scalar.derive.source)
        input_entries = [
            ArtifactInput(
                kind=input_kind,
                id=scalar.derive.source,
                digest=sha256_file(source_path),
                path=source_path.as_posix(),
            )
        ]
        input_payload = {"source": scalar.derive.source}
        if scalar.derive.kind == "column_expression":
            params = {
                "derive_kind": scalar.derive.kind,
                "expression": scalar.derive.expression,
                "output_column": scalar.derive.output_column,
            }
        elif scalar.derive.kind == "select_columns":
            params = {
                "derive_kind": scalar.derive.kind,
                "columns": scalar.derive.columns,
            }
        elif scalar.derive.kind == "rename_columns":
            params = {
                "derive_kind": scalar.derive.kind,
                "renames": scalar.derive.renames,
            }
    else:
        input_entries = []
        for source_id in scalar.derive.sources:
            input_kind, source_path = _resolve_table_source(context, source_id)
            input_entries.append(
                ArtifactInput(
                    kind=input_kind,
                    id=source_id,
                    digest=sha256_file(source_path),
                    path=source_path.as_posix(),
                )
            )
        input_payload = {"sources": scalar.derive.sources}
        params = {
            "derive_kind": scalar.derive.kind,
            "sources": scalar.derive.sources,
            "on": scalar.derive.on,
        }

    manifest = ArtifactManifest(
        artifact_kind="scalar_table",
        artifact_id=scalar_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="scalar derive",
        inputs=input_entries,
        params=params,
        outputs=[ArtifactOutput(path="table.parquet", media_type="application/x-parquet")],
        stats={"rows": rows, "columns": len(columns)},
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="scalar derive",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="scalar_table",
        artifact_id=scalar_id,
        outputs=[artifact_dir.as_posix()],
        inputs=input_payload,
        metrics={"rows": rows, "columns": len(columns)},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="scalar_derive",
        artifact_id=scalar_id,
    )
    return result
