"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/scalars/builders/sidecar_table.py

Publish declared scalar sidecar parquet files as first-class scalar tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...io.json_io import read_json
from ...io.parquet_io import read_table, write_table
from ...workspaces.loader import WorkspaceContext
from ..common import BuiltScalarArtifact, ScalarInputRef, _optional_param, _require_param

_SCALAR_SIDECAR_TABLE_KIND = "scalar_sidecar_table"


def _source_scalar_paths(context: WorkspaceContext, source_scalar: str) -> tuple[Path, Path]:
    source_dir = context.output_root / "scalars" / source_scalar
    manifest_path = source_dir / "manifest.json"
    table_path = source_dir / "table.parquet"
    if not manifest_path.is_file():
        raise MissingArtifactError(f"{_SCALAR_SIDECAR_TABLE_KIND} is missing source manifest: {source_scalar}")
    if not table_path.is_file():
        raise MissingArtifactError(f"{_SCALAR_SIDECAR_TABLE_KIND} is missing source table: {source_scalar}")
    return table_path, manifest_path


def _safe_sidecar_name(value: object) -> str:
    sidecar = str(value)
    path = Path(sidecar)
    if not sidecar or path.name != sidecar or path.is_absolute():
        raise ContractViolationError(
            f"{_SCALAR_SIDECAR_TABLE_KIND} sidecar must be a file name under the source scalar directory"
        )
    if path.suffix != ".parquet":
        raise ContractViolationError(f"{_SCALAR_SIDECAR_TABLE_KIND} sidecar must be a parquet file")
    return sidecar


def _manifest_declares_sidecar(manifest: dict[str, object], sidecar: str) -> bool:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        return False
    for output in outputs:
        if not isinstance(output, dict):
            continue
        if str(output.get("path") or "") == sidecar:
            return True
    return False


def _validate_required_columns(table: pa.Table, required_columns: list[str]) -> None:
    missing_columns = sorted(set(required_columns) - set(table.column_names))
    if missing_columns:
        raise ContractViolationError(
            f"{_SCALAR_SIDECAR_TABLE_KIND} sidecar table is missing required columns: {missing_columns}"
        )


def build_scalar_sidecar_table_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, object],
) -> BuiltScalarArtifact:
    """Expose a declared scalar sidecar parquet as a normal scalar table."""

    source_scalar = str(_require_param(params, "source_scalar"))
    sidecar = _safe_sidecar_name(_require_param(params, "sidecar"))
    required_columns = [str(column) for column in _optional_param(params, "required_columns", default=[])]
    require_manifest_output = bool(_optional_param(params, "require_manifest_output", default=True))

    _, manifest_path = _source_scalar_paths(context, source_scalar)
    manifest = read_json(manifest_path)
    if require_manifest_output and not _manifest_declares_sidecar(manifest, sidecar):
        raise ContractViolationError(
            f"{_SCALAR_SIDECAR_TABLE_KIND} source scalar {source_scalar!r} does not declare sidecar {sidecar!r}"
        )

    sidecar_path = context.output_root / "scalars" / source_scalar / sidecar
    if not sidecar_path.is_file():
        raise MissingArtifactError(f"{_SCALAR_SIDECAR_TABLE_KIND} is missing sidecar: {sidecar_path}")
    table = read_table(sidecar_path)
    _validate_required_columns(table, required_columns)
    write_table(table, artifact_dir / "table.parquet")

    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=[
            ScalarInputRef(kind="scalar_manifest", artifact_id=source_scalar, path=manifest_path),
            ScalarInputRef(kind="scalar_sidecar", artifact_id=source_scalar, path=sidecar_path),
        ],
        outputs=[],
        stats={
            "source_scalar": source_scalar,
            "sidecar": sidecar,
            "required_columns": required_columns,
        },
    )
