"""
Export services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..exports.anndata import build_export_anndata_artifact
from ..exports.matrix import build_export_matrix_artifact
from ..exports.table import build_export_table_artifact
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config
from .memory_service import apply_memory_preflight, evaluate_export_preflight


def _prepare_export_dir(context, export_id: str, *, force: bool) -> Path:
    export_dir = context.output_root / "exports" / export_id
    if export_dir.exists() and not force:
        raise ArtifactConflictError(f"export artifact already exists: {export_dir}")
    if force and export_dir.exists():
        import shutil

        shutil.rmtree(export_dir)
    return export_dir


def _build_export_inputs(export, basis_path: Path, block_rows: list[dict[str, object]]) -> list[ArtifactInput]:
    inputs = [
        ArtifactInput(
            kind="row_basis",
            id=export.row_basis,
            digest=sha256_file(basis_path),
            path=basis_path.as_posix(),
        )
    ]
    seen_sources = {basis_path.as_posix()}
    for block_row in block_rows:
        source_path = Path(str(block_row["source_path"]))
        rows_path = Path(str(block_row["rows_path"]))
        for kind, input_id, path in [
            ("export_block", str(block_row["source_artifact_id"]), source_path),
            ("export_block_rows", str(block_row["source_artifact_id"]), rows_path),
        ]:
            source_key = path.as_posix()
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            inputs.append(ArtifactInput(kind=kind, id=input_id, digest=sha256_file(path), path=path.as_posix()))
        alignment_path = block_row.get("alignment_path")
        alignment_id = block_row.get("alignment_id")
        if alignment_path is not None and alignment_id is not None:
            path = Path(str(alignment_path))
            source_key = path.as_posix()
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                inputs.append(
                    ArtifactInput(
                        kind="alignment_set",
                        id=str(alignment_id),
                        digest=sha256_file(path),
                        path=path.as_posix(),
                    )
                )
    return inputs


def _build_supplemental_export_inputs(supplemental_inputs: list[dict[str, object]]) -> list[ArtifactInput]:
    inputs: list[ArtifactInput] = []
    seen_paths: set[str] = set()
    for row in supplemental_inputs:
        kind = str(row["kind"])
        input_id = str(row["id"])
        for key in ["path", "rows_path", "indices_path", "distances_path"]:
            path_value = row.get(key)
            if path_value is None:
                continue
            path = Path(str(path_value))
            path_key = path.as_posix()
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            inputs.append(ArtifactInput(kind=kind, id=input_id, digest=sha256_file(path), path=path_key))
    return inputs


def export_matrix(
    workspace: str | Path,
    export_id: str,
    *,
    allow_memory_overage: bool = False,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    export = context.require_export(export_id)
    _prepare_export_dir(context, export_id, force=force)
    preflight = evaluate_export_preflight(context, export_id=export_id, export_kind="matrix")
    status, warnings = apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)

    export_dir, basis_path, rows, dims, feature_rows, block_rows = build_export_matrix_artifact(
        context,
        export_id=export_id,
    )
    inputs = _build_export_inputs(export, basis_path, block_rows)
    manifest = ArtifactManifest(
        artifact_kind="export_bundle",
        artifact_id=export_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="export matrix",
        status=status,
        inputs=inputs,
        params={
            "row_basis": export.row_basis,
            "block_count": len(export.blocks),
            "blocks": block_rows,
            "output_matrix_dtype": export.matrix_dtype or context.analysis_dtype,
            "memory_preflight": preflight.as_payload(),
        },
        outputs=[
            ArtifactOutput(path="matrix.npy", media_type="application/x-npy"),
            ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="features.parquet", media_type="application/x-parquet"),
        ],
        stats={"rows": rows, "dims": dims, "features": len(feature_rows)},
        warnings=warnings,
    )
    write_manifest(export_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="export matrix",
        workspace_id=context.workspace_id,
        status=status,
        artifact_kind="export_bundle",
        artifact_id=export_id,
        outputs=[export_dir.as_posix()],
        inputs={"export": export_id, "row_basis": export.row_basis},
        warnings=warnings,
        metrics={"rows": rows, "dims": dims, "features": len(feature_rows), "memory_preflight": preflight.as_payload()},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="export_matrix",
        artifact_id=export_id,
    )
    return result


def export_table(
    workspace: str | Path,
    export_id: str,
    *,
    allow_memory_overage: bool = False,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    export = context.require_export(export_id)
    _prepare_export_dir(context, export_id, force=force)
    preflight = evaluate_export_preflight(context, export_id=export_id, export_kind="table")
    status, warnings = apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)

    export_dir, basis_path, rows, features, feature_rows, block_rows = build_export_table_artifact(
        context,
        export_id=export_id,
    )
    inputs = _build_export_inputs(export, basis_path, block_rows)
    manifest = ArtifactManifest(
        artifact_kind="export_bundle",
        artifact_id=export_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="export table",
        status=status,
        inputs=inputs,
        params={
            "row_basis": export.row_basis,
            "block_count": len(export.blocks),
            "blocks": block_rows,
            "output_table_kind": "aligned_table",
            "memory_preflight": preflight.as_payload(),
        },
        outputs=[
            ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="table.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="features.parquet", media_type="application/x-parquet"),
        ],
        stats={"rows": rows, "features": features},
        warnings=warnings,
    )
    write_manifest(export_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="export table",
        workspace_id=context.workspace_id,
        status=status,
        artifact_kind="export_bundle",
        artifact_id=export_id,
        outputs=[export_dir.as_posix()],
        inputs={"export": export_id, "row_basis": export.row_basis},
        warnings=warnings,
        metrics={"rows": rows, "features": features, "memory_preflight": preflight.as_payload()},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="export_table",
        artifact_id=export_id,
    )
    return result


def export_anndata(
    workspace: str | Path,
    export_id: str,
    *,
    projection_ids: list[str] | None = None,
    neighbor_ids: list[str] | None = None,
    allow_memory_overage: bool = False,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    export = context.require_export(export_id)
    _prepare_export_dir(context, export_id, force=force)
    preflight = evaluate_export_preflight(context, export_id=export_id, export_kind="anndata")
    status, warnings = apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)

    export_dir, basis_path, bundle_path, rows, dims, feature_rows, block_rows, supplemental_inputs = (
        build_export_anndata_artifact(
            context,
            export_id=export_id,
            projection_ids=projection_ids,
            neighbor_ids=neighbor_ids,
        )
    )
    inputs = [
        *_build_export_inputs(export, basis_path, block_rows),
        *_build_supplemental_export_inputs(supplemental_inputs),
    ]
    manifest = ArtifactManifest(
        artifact_kind="export_bundle",
        artifact_id=export_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="export anndata",
        status=status,
        inputs=inputs,
        params={
            "row_basis": export.row_basis,
            "block_count": len(export.blocks),
            "blocks": block_rows,
            "output_matrix_dtype": export.matrix_dtype or context.analysis_dtype,
            "anndata_schema_version": "latentdna.anndata_export.v1",
            "projection_ids": list(projection_ids or []),
            "neighbor_ids": list(neighbor_ids or []),
            "supplemental_inputs": supplemental_inputs,
            "memory_preflight": preflight.as_payload(),
        },
        outputs=[
            ArtifactOutput(path="bundle.h5ad", media_type="application/x-hdf5"),
            ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="features.parquet", media_type="application/x-parquet"),
        ],
        stats={
            "rows": rows,
            "dims": dims,
            "features": len(feature_rows),
            "obsm": len(projection_ids or []),
            "obsp": len(neighbor_ids or []),
        },
        warnings=warnings,
    )
    write_manifest(export_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="export anndata",
        workspace_id=context.workspace_id,
        status=status,
        artifact_kind="export_bundle",
        artifact_id=export_id,
        outputs=[export_dir.as_posix(), bundle_path.as_posix()],
        inputs={
            "export": export_id,
            "row_basis": export.row_basis,
            "projections": list(projection_ids or []),
            "neighbors": list(neighbor_ids or []),
        },
        warnings=warnings,
        metrics={
            "rows": rows,
            "dims": dims,
            "features": len(feature_rows),
            "obsm": len(projection_ids or []),
            "obsp": len(neighbor_ids or []),
            "memory_preflight": preflight.as_payload(),
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="export_anndata",
        artifact_id=export_id,
    )
    return result
