"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/registry/validation.py

Registry validation helpers for USR namespaces and overlay schemas.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa

from ..contracts import SchemaError
from .models import USR_STATE_COLUMNS, USR_STATE_NAMESPACE, RegistryColumn, RegistryEntry
from .typespec import arrow_type_str, parse_type_str


def registry_entry(entries: dict[str, RegistryEntry], namespace: str) -> RegistryEntry:
    if namespace not in entries:
        raise SchemaError(f"Namespace '{namespace}' is not registered. Register it with `usr namespace register ...`.")
    return entries[namespace]


def validate_overlay_schema(
    namespace: str,
    schema: pa.Schema,
    *,
    registry: dict[str, RegistryEntry],
    key: str,
) -> None:
    entry = registry_entry(registry, namespace)
    allowed = {column.name: column.type for column in entry.columns}
    for field in schema:
        if field.name == key:
            continue
        if field.name not in allowed:
            raise SchemaError(f"Overlay column '{field.name}' not registered under namespace '{namespace}'.")
        expected = allowed[field.name]
        actual = arrow_type_str(field.type)
        if actual != expected:
            raise SchemaError(f"Overlay column '{field.name}' type mismatch: expected {expected}, got {actual}.")


def parse_columns_spec(spec: str, *, namespace: str) -> list[RegistryColumn]:
    cols: list[RegistryColumn] = []
    if not spec:
        return cols
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise SchemaError(f"Invalid column spec '{raw}'. Use name:type.")
        name, type_str = raw.split(":", 1)
        name = name.strip()
        type_str = type_str.strip()
        if not name or not type_str:
            raise SchemaError(f"Invalid column spec '{raw}'. Use name:type.")
        cols.append(RegistryColumn(name=name, type=type_str))
    _validate_columns(namespace, cols)
    return cols


def _parse_entry(namespace: str, entry: object) -> RegistryEntry:
    if not isinstance(entry, dict):
        raise SchemaError(f"Registry entry for '{namespace}' must be a mapping.")
    owner = entry.get("owner")
    description = entry.get("description")
    cols_raw = entry.get("columns") or []
    if not isinstance(cols_raw, list):
        raise SchemaError(f"Registry entry for '{namespace}' must define 'columns' as a list.")
    cols: list[RegistryColumn] = []
    for record in cols_raw:
        if not isinstance(record, dict):
            raise SchemaError(f"Registry column for '{namespace}' must be a mapping.")
        name = record.get("name")
        type_str = record.get("type")
        if not name or not type_str:
            raise SchemaError(f"Registry column for '{namespace}' requires name and type.")
        cols.append(RegistryColumn(name=str(name), type=str(type_str)))
    _validate_columns(namespace, cols)
    return RegistryEntry(namespace=namespace, owner=owner, description=description, columns=cols)


def _validate_columns(namespace: str, cols: list[RegistryColumn]) -> None:
    names = [column.name for column in cols]
    if len(names) != len(set(names)):
        raise SchemaError(f"Registry namespace '{namespace}' has duplicate column names.")
    prefix = f"{namespace}__"
    for column in cols:
        if "__" not in column.name:
            raise SchemaError(f"Registry column '{column.name}' must be namespaced.")
        if not column.name.startswith(prefix):
            raise SchemaError(f"Registry column '{column.name}' must start with '{prefix}'.")
        _ = parse_type_str(column.type)


def _ensure_usr_state_entry(entries: dict[str, RegistryEntry]) -> None:
    if USR_STATE_NAMESPACE not in entries:
        raise SchemaError(
            "Registry must include reserved namespace 'usr_state'. Add usr_state columns to registry.yaml."
        )
    expected = {column.name: column.type for column in USR_STATE_COLUMNS}
    actual = {column.name: column.type for column in entries[USR_STATE_NAMESPACE].columns}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatched = []
        for name in sorted(set(expected) & set(actual)):
            if expected[name] != actual[name]:
                mismatched.append(f"{name} (expected {expected[name]}, got {actual[name]})")
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if mismatched:
            details.append(f"mismatched={mismatched}")
        raise SchemaError("Registry entry for 'usr_state' must match the reserved schema. " + " ".join(details))
