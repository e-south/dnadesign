"""
LatentDNA source adapter for canonical Infer feature-vector sidecars.

This adapter exposes `_derived/infer/feature_aliases.parquet` joined to
`feature_vectors.parquet`, USR sequence views, mutable view semantics, and the
owning dataset rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa

from . import infer_sidecar_join

_FEATURE_ALIAS_RELATIVE_PATH = "_derived/infer/feature_aliases.parquet"
_FEATURE_VECTOR_RELATIVE_PATH = "_derived/infer/feature_vectors.parquet"
_VECTOR_COLUMN = "value"
_VECTOR_CREATED_AT_COLUMN = "feature_vector_created_at"
_ALIAS_CREATED_AT_COLUMN = "feature_alias_created_at"
_SOURCE_LABEL = "infer feature sidecar"
_BATCH_SIZE = 2048
_ALIAS_SCHEMA = pa.schema(
    [
        pa.field("alias_id", pa.string()),
        pa.field("view_id", pa.string()),
        pa.field("view_name", pa.string()),
        pa.field("sequence_id", pa.string()),
        pa.field("feature_vector_key", pa.string()),
        pa.field("forward_pass_key", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("model_name", pa.string()),
        pa.field("model_revision", pa.string()),
        pa.field("layer_name", pa.string()),
        pa.field("representation_kind", pa.string()),
        pa.field("pooling_operation", pa.string()),
        pa.field("pooling_start_0", pa.int64()),
        pa.field("pooling_end_0", pa.int64()),
        pa.field("orientation", pa.string()),
        pa.field("source_dataset_id", pa.string()),
        pa.field("feature_request_digest", pa.string()),
        pa.field("runtime_fingerprint_key", pa.string()),
        pa.field("sequence_case_policy", pa.string()),
        pa.field("created_at", pa.string()),
    ]
)
_VECTOR_SCHEMA = pa.schema(
    [
        pa.field("feature_vector_key", pa.string()),
        pa.field(_VECTOR_COLUMN, pa.list_(pa.float64())),
        pa.field("created_at", pa.string()),
    ]
)
_VECTOR_VALUE_FIELD_TYPES = {
    _VECTOR_COLUMN: pa.list_(pa.float64()),
    _VECTOR_CREATED_AT_COLUMN: pa.string(),
}
_CONTRACT = infer_sidecar_join.InferSidecarJoinContract(
    source_label=_SOURCE_LABEL,
    alias_role="infer_feature_aliases",
    payload_role="infer_feature_vectors",
    missing_payload_label="feature vectors",
    alias_relative_path=_FEATURE_ALIAS_RELATIVE_PATH,
    payload_relative_path=_FEATURE_VECTOR_RELATIVE_PATH,
    payload_key_column="feature_vector_key",
    payload_value_column=_VECTOR_COLUMN,
    payload_created_at_column=_VECTOR_CREATED_AT_COLUMN,
    alias_created_at_column=_ALIAS_CREATED_AT_COLUMN,
    alias_schema=_ALIAS_SCHEMA,
    payload_schema=_VECTOR_SCHEMA,
    payload_value_field_types=_VECTOR_VALUE_FIELD_TYPES,
    vector_columns=[_VECTOR_COLUMN],
)


def feature_aliases_path(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return infer_sidecar_join.alias_path(_CONTRACT, root, dataset, workspace_dir=workspace_dir)


def feature_vectors_path(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    return infer_sidecar_join.payload_path(_CONTRACT, root, dataset, workspace_dir=workspace_dir)


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


def iter_grouped_batches(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    requests: list[infer_sidecar_join.SidecarBatchRequest],
    batch_size: int = _BATCH_SIZE,
):
    yield from infer_sidecar_join.iter_grouped_batches(
        _CONTRACT,
        root,
        dataset,
        workspace_dir=workspace_dir,
        requests=requests,
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
