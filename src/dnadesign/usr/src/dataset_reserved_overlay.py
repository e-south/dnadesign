"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_reserved_overlay.py

Reserved overlay write helpers for USR dataset state and tombstone operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import NamespaceError, SchemaError
from .overlays import overlay_dir_path, overlay_metadata, overlay_path, with_overlay_metadata
from .storage.parquet import PARQUET_COMPRESSION, now_utc


class DatasetReservedOverlayHost(Protocol):
    dir: Path

    def _auto_freeze_registry(self, *, record_auto_event: bool = True) -> tuple[Path, str, bool]: ...

    def _validate_registry_schema(self, *, namespace: str, schema: pa.Schema, key: str) -> None: ...

    def _registry_hash(self, *, required: bool) -> str | None: ...


def write_reserved_overlay(
    dataset: DatasetReservedOverlayHost,
    namespace: str,
    key: str,
    overlay_df: pd.DataFrame,
    *,
    validate_registry: bool = False,
    schema_types: dict[str, pa.DataType] | None = None,
    namespace_pattern: re.Pattern[str],
) -> int:
    dataset._auto_freeze_registry()
    if not namespace_pattern.match(namespace):
        raise NamespaceError(
            "Invalid namespace. Use lowercase letters, digits, and underscores, starting with a letter."
        )
    if overlay_df[key].duplicated().any():
        raise SchemaError(f"Overlay has duplicate keys for '{key}'.")
    out_path = overlay_path(dataset.dir, namespace)
    dir_path = overlay_dir_path(dataset.dir, namespace)
    if dir_path.exists():
        raise SchemaError(f"Overlay parts already exist for namespace '{namespace}'. Remove them before writing.")
    if out_path.exists():
        meta = overlay_metadata(out_path)
        if meta.get("key") != key:
            raise SchemaError(f"Overlay key mismatch for namespace '{namespace}': existing={meta.get('key')} new={key}")
        existing_df = pq.read_table(out_path).to_pandas()
        if existing_df[key].duplicated().any():
            raise SchemaError(f"Existing overlay has duplicate keys for '{key}'.")
        existing_df = existing_df.set_index(key, drop=False)
        new_df = overlay_df.set_index(key, drop=False)
        combined = existing_df
        for col in new_df.columns:
            if col == key:
                continue
            if col in combined.columns:
                combined.loc[new_df.index, col] = new_df[col]
            else:
                combined = combined.join(new_df[[col]], how="outer")
        combined[key] = combined.index
        overlay_df = combined.reset_index(drop=True)

    schema = None
    if schema_types:
        fields = []
        for col in overlay_df.columns:
            if col in schema_types:
                fields.append(pa.field(col, schema_types[col]))
        if fields:
            schema = pa.schema(fields)

    table = pa.Table.from_pandas(overlay_df, preserve_index=False, schema=schema)
    if validate_registry:
        dataset._validate_registry_schema(namespace=namespace, schema=table.schema, key=key)
    reg_hash = dataset._registry_hash(required=False)
    table = with_overlay_metadata(
        table,
        namespace=namespace,
        key=key,
        created_at=now_utc(),
        registry_hash=reg_hash,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp, compression=PARQUET_COMPRESSION)
    os.replace(tmp, out_path)
    return int(overlay_df.shape[0])
