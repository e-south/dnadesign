"""
LatentDNA source adapter for canonical Infer feature-scalar sidecars.

This adapter exposes `_derived/infer/feature_scalar_aliases.parquet` joined to
`feature_scalars.parquet`, USR sequence views, mutable view semantics, and the
owning dataset rows. It mirrors the feature-vector sidecar adapter, but returns
one scalar `value` column for outputs such as Evo2 log-likelihood totals and
mean-per-token diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa

from . import infer_sidecar_join

_FEATURE_SCALAR_ALIAS_RELATIVE_PATH = "_derived/infer/feature_scalar_aliases.parquet"
_FEATURE_SCALAR_RELATIVE_PATH = "_derived/infer/feature_scalars.parquet"
_SCALAR_COLUMN = "value"
_SCALAR_CREATED_AT_COLUMN = "feature_scalar_created_at"
_ALIAS_CREATED_AT_COLUMN = "feature_scalar_alias_created_at"
_SOURCE_LABEL = "infer feature scalar sidecar"
_BATCH_SIZE = 4096
_ALIAS_SCHEMA = pa.schema(
    [
        pa.field("alias_id", pa.string()),
        pa.field("view_id", pa.string()),
        pa.field("view_name", pa.string()),
        pa.field("sequence_id", pa.string()),
        pa.field("feature_scalar_key", pa.string()),
        pa.field("forward_pass_key", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("model_name", pa.string()),
        pa.field("model_revision", pa.string()),
        pa.field("scalar_kind", pa.string()),
        pa.field("orientation", pa.string()),
        pa.field("source_dataset_id", pa.string()),
        pa.field("feature_request_digest", pa.string()),
        pa.field("created_at", pa.string()),
    ]
)
_SCALAR_SCHEMA = pa.schema(
    [
        pa.field("feature_scalar_key", pa.string()),
        pa.field(_SCALAR_COLUMN, pa.float64()),
        pa.field("created_at", pa.string()),
    ]
)
_SCALAR_VALUE_FIELD_TYPES = {
    _SCALAR_COLUMN: pa.float64(),
    _SCALAR_CREATED_AT_COLUMN: pa.string(),
}
_CONTRACT = infer_sidecar_join.InferSidecarJoinContract(
    source_label=_SOURCE_LABEL,
    alias_role="infer_feature_scalar_aliases",
    payload_role="infer_feature_scalars",
    missing_payload_label="feature scalars",
    alias_relative_path=_FEATURE_SCALAR_ALIAS_RELATIVE_PATH,
    payload_relative_path=_FEATURE_SCALAR_RELATIVE_PATH,
    payload_key_column="feature_scalar_key",
    payload_value_column=_SCALAR_COLUMN,
    payload_created_at_column=_SCALAR_CREATED_AT_COLUMN,
    alias_created_at_column=_ALIAS_CREATED_AT_COLUMN,
    alias_schema=_ALIAS_SCHEMA,
    payload_schema=_SCALAR_SCHEMA,
    payload_value_field_types=_SCALAR_VALUE_FIELD_TYPES,
    vector_columns=[],
)


def feature_scalar_aliases_path(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return infer_sidecar_join.alias_path(_CONTRACT, root, dataset, workspace_dir=workspace_dir)


def _read_alias_table(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
) -> pa.Table:
    return infer_sidecar_join.read_alias_table(_CONTRACT, root, dataset, workspace_dir=workspace_dir, where=where)


def inspect_schema(
    root: str, dataset: str, *, workspace_dir: Path, where: Mapping[str, object] | None
) -> dict[str, Any]:
    aliases = _read_alias_table(root, dataset, workspace_dir=workspace_dir, where=where)
    return infer_sidecar_join.inspect_schema(
        _CONTRACT,
        root,
        dataset,
        workspace_dir=workspace_dir,
        where=where,
        aliases=aliases,
    )


def iter_batches(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
    batch_size: int = _BATCH_SIZE,
):
    yield from infer_sidecar_join.iter_batches(
        _CONTRACT,
        root,
        dataset,
        workspace_dir=workspace_dir,
        where=where,
        columns=columns,
        batch_size=batch_size,
    )


def read_table(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    where: Mapping[str, object] | None,
    columns: list[str] | None,
) -> pa.Table:
    return infer_sidecar_join.read_table(
        _CONTRACT,
        root,
        dataset,
        workspace_dir=workspace_dir,
        where=where,
        columns=columns,
        batch_size=_BATCH_SIZE,
    )


def source_provenance(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    columns: list[str] | None,
) -> list[dict[str, object]]:
    return infer_sidecar_join.source_provenance(_CONTRACT, root, dataset, workspace_dir=workspace_dir, columns=columns)
