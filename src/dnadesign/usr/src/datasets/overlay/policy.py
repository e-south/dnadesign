"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/overlay/policy.py

Overlay policy and registry-coercion helpers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import pyarrow as pa

from ...errors import NamespaceError, SchemaError
from ...overlays import overlay_dir_path
from ...registry import registry_entry
from ...schema import REQUIRED_COLUMNS

SUPPORTED_OVERLAY_KEYS = {"id", "sequence", "sequence_norm", "sequence_ci"}


def validate_overlay_join_key(key: str, *, context_label: str) -> str:
    resolved_key = str(key).strip()
    if resolved_key not in SUPPORTED_OVERLAY_KEYS:
        raise SchemaError(f"Unsupported {context_label} '{resolved_key}'.")
    return resolved_key


def validate_overlay_target(
    *,
    dataset: Any,
    namespace: str,
    key: str,
    namespace_pattern: Any,
    reserved_namespaces: set[str],
) -> str:
    dataset._require_exists()
    if not namespace_pattern.match(namespace):
        raise NamespaceError(
            "Invalid namespace. Use lowercase letters, digits, and underscores, starting with a letter."
        )
    if namespace in reserved_namespaces:
        raise NamespaceError(f"Namespace '{namespace}' is reserved.")
    resolved_key = validate_overlay_join_key(key, context_label="join key")
    part_dir = overlay_dir_path(dataset.dir, namespace)
    if part_dir.exists():
        raise SchemaError(
            f"Overlay parts already exist for namespace '{namespace}'. "
            "Use write_overlay_part or compact the parts first."
        )
    return resolved_key


def normalize_overlay_targets(namespace: str, columns: Iterable[str]) -> list[str]:
    targets: list[str] = []
    for column_name in columns:
        if column_name.startswith(namespace + "__"):
            targets.append(column_name)
            continue
        if "__" in column_name:
            raise NamespaceError(f"Column '{column_name}' does not belong to namespace '{namespace}'.")
        targets.append(f"{namespace}__{column_name}")
    return targets


def ensure_overlay_columns_allowed(columns: Iterable[str]) -> None:
    essential = {column_name for column_name, _ in REQUIRED_COLUMNS}
    for column_name in columns:
        if column_name in essential:
            raise NamespaceError(f"Refusing to write essential column: {column_name}")
        if "__" not in column_name:
            raise NamespaceError(f"Derived columns must be namespaced (got '{column_name}').")


def _overlay_table_from_registry(overlay_df: pd.DataFrame, *, entry: Any, key: str) -> pa.Table:
    fields = [pa.field(key, pa.string())]
    allowed = {column.name: column.type for column in entry.columns}
    for name in overlay_df.columns:
        if name == key:
            continue
        if name not in allowed:
            raise SchemaError(f"Overlay column '{name}' not registered under namespace '{entry.namespace}'.")
        fields.append(pa.field(name, _registry_type_to_arrow(allowed[name])))
    schema = pa.schema(fields)
    try:
        arrays: dict[str, pa.Array] = {}
        for field in schema:
            arrays[field.name] = _overlay_arrow_array(overlay_df[field.name], field=field)
        return pa.table(arrays, schema=schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
        raise SchemaError(f"Overlay type mismatch under namespace '{entry.namespace}': {error}") from error


def _overlay_arrow_array(series: pd.Series, *, field: pa.Field) -> pa.Array:
    if field.name == "id" or pa.types.is_string(field.type) or pa.types.is_struct(field.type):
        values = [_normalize_arrow_value(value) for value in series.tolist()]
        return pa.array(values, type=field.type)
    if pa.types.is_list(field.type) or pa.types.is_large_list(field.type) or pa.types.is_fixed_size_list(field.type):
        return pa.array(series.tolist(), type=field.type)
    return pa.Array.from_pandas(series, type=field.type)


def _normalize_arrow_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        value = converted
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _registry_type_to_arrow(type_str: str) -> pa.DataType:
    primitive = {
        "string": pa.string(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
        "float16": pa.float16(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
    }
    if type_str in primitive:
        return primitive[type_str]
    if type_str.startswith("list<") and type_str.endswith(">"):
        inner = type_str[len("list<") : -1].strip()
        return pa.list_(_registry_type_to_arrow(inner))
    if type_str.startswith("fixed_size_list<") and type_str.endswith("]"):
        inner_and_size = type_str[len("fixed_size_list<") :]
        inner, size_text = inner_and_size.split(">[", 1)
        return pa.list_(_registry_type_to_arrow(inner.strip()), int(size_text[:-1]))
    if type_str.startswith("timestamp[") and type_str.endswith("]"):
        inner = type_str[len("timestamp[") : -1]
        parts = [part.strip() for part in inner.split(",")]
        if len(parts) == 1:
            return pa.timestamp(parts[0])
        if len(parts) == 2:
            return pa.timestamp(parts[0], tz=parts[1])
    if type_str.startswith("struct<") and type_str.endswith(">"):
        fields = []
        for item in _split_top_level_registry_fields(type_str[len("struct<") : -1]):
            name, inner = item.split(":", 1)
            fields.append(pa.field(name.strip(), _registry_type_to_arrow(inner.strip())))
        return pa.struct(fields)
    raise SchemaError(f"Unsupported registry type '{type_str}'.")


def _split_top_level_registry_fields(text: str) -> list[str]:
    depth = 0
    current: list[str] = []
    parts: list[str] = []
    for char in text:
        if char in "<[":
            depth += 1
        elif char in ">]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def coerce_null_overlay_columns_to_registry_schema(
    *,
    dataset: Any,
    namespace: str,
    tbl: pa.Table,
    key: str,
) -> pa.Table:
    if not any(pa.types.is_null(field.type) for field in tbl.schema if field.name != key):
        return tbl
    registry = dataset._registry(required=True)
    entry = registry_entry(registry, namespace)
    target_types = {column.name: _registry_type_to_arrow(column.type) for column in entry.columns}
    arrays: list[pa.Array | pa.ChunkedArray] = []
    names: list[str] = []
    changed = False
    for field in tbl.schema:
        names.append(field.name)
        if field.name == key or not pa.types.is_null(field.type):
            arrays.append(tbl[field.name])
            continue
        target_type = target_types.get(field.name)
        if target_type is None:
            arrays.append(tbl[field.name])
            continue
        arrays.append(pa.nulls(tbl.num_rows, type=target_type))
        changed = True
    if not changed:
        return tbl
    return pa.Table.from_arrays(arrays, names=names)
