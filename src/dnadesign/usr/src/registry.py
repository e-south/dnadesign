"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/registry.py

Namespace registry loading and validation for USR overlays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pyarrow as pa
import yaml

from .errors import SchemaError

REGISTRY_FILENAME = "registry.yaml"


@dataclass(frozen=True)
class RegistryColumn:
    name: str
    type: str


@dataclass(frozen=True)
class RegistryEntry:
    namespace: str
    owner: Optional[str]
    description: Optional[str]
    columns: List[RegistryColumn]


def registry_path(root: Path) -> Path:
    return Path(root) / REGISTRY_FILENAME


def load_registry(root: Path, *, required: bool) -> Dict[str, RegistryEntry]:
    path = registry_path(root)
    if not path.exists():
        if required:
            raise SchemaError(f"Registry required but not found: {path}. Create it with `usr namespace register ...`.")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    namespaces = data.get("namespaces") or {}
    if not isinstance(namespaces, dict):
        raise SchemaError("Registry file must contain a 'namespaces' mapping.")
    out: Dict[str, RegistryEntry] = {}
    for ns, entry in namespaces.items():
        out[str(ns)] = _parse_entry(str(ns), entry)
    return out


def save_registry(root: Path, entries: Dict[str, RegistryEntry]) -> Path:
    path = registry_path(root)
    payload = {
        "namespaces": {
            ns: {
                "owner": entry.owner,
                "description": entry.description,
                "columns": [{"name": c.name, "type": c.type} for c in entry.columns],
            }
            for ns, entry in sorted(entries.items())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
    return path


def register_namespace(
    root: Path,
    *,
    namespace: str,
    columns: Iterable[RegistryColumn],
    owner: Optional[str] = None,
    description: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    entries = load_registry(root, required=False)
    if namespace in entries and not overwrite:
        raise SchemaError(f"Namespace '{namespace}' already registered. Use --overwrite to replace.")
    cols = list(columns)
    if not cols:
        raise SchemaError("Registry entry must include at least one column.")
    _validate_columns(namespace, cols)
    entries[namespace] = RegistryEntry(
        namespace=namespace,
        owner=owner,
        description=description,
        columns=cols,
    )
    return save_registry(root, entries)


def registry_entry(entries: Dict[str, RegistryEntry], namespace: str) -> RegistryEntry:
    if namespace not in entries:
        raise SchemaError(f"Namespace '{namespace}' is not registered. Register it with `usr namespace register ...`.")
    return entries[namespace]


def validate_overlay_schema(
    namespace: str,
    schema: pa.Schema,
    *,
    registry: Dict[str, RegistryEntry],
    key: str,
) -> None:
    entry = registry_entry(registry, namespace)
    allowed = {c.name: c.type for c in entry.columns}
    for field in schema:
        if field.name == key:
            continue
        if field.name not in allowed:
            raise SchemaError(f"Overlay column '{field.name}' not registered under namespace '{namespace}'.")
        expected = allowed[field.name]
        actual = arrow_type_str(field.type)
        if actual != expected:
            raise SchemaError(f"Overlay column '{field.name}' type mismatch: expected {expected}, got {actual}.")


def parse_columns_spec(spec: str, *, namespace: str) -> List[RegistryColumn]:
    cols: List[RegistryColumn] = []
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
    cols: List[RegistryColumn] = []
    for rec in cols_raw:
        if not isinstance(rec, dict):
            raise SchemaError(f"Registry column for '{namespace}' must be a mapping.")
        name = rec.get("name")
        type_str = rec.get("type")
        if not name or not type_str:
            raise SchemaError(f"Registry column for '{namespace}' requires name and type.")
        cols.append(RegistryColumn(name=str(name), type=str(type_str)))
    _validate_columns(namespace, cols)
    return RegistryEntry(namespace=namespace, owner=owner, description=description, columns=cols)


def _validate_columns(namespace: str, cols: List[RegistryColumn]) -> None:
    names = [c.name for c in cols]
    if len(names) != len(set(names)):
        raise SchemaError(f"Registry namespace '{namespace}' has duplicate column names.")
    prefix = f"{namespace}__"
    for c in cols:
        if "__" not in c.name:
            raise SchemaError(f"Registry column '{c.name}' must be namespaced.")
        if not c.name.startswith(prefix):
            raise SchemaError(f"Registry column '{c.name}' must start with '{prefix}'.")
        _ = parse_type_str(c.type)


def parse_type_str(type_str: str) -> str:
    type_str = type_str.strip()
    if type_str in {
        "string",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "bool",
    }:
        return type_str
    if type_str.startswith("list<") and type_str.endswith(">"):
        inner = type_str[len("list<") : -1].strip()
        return f"list<{parse_type_str(inner)}>"
    if type_str.startswith("timestamp[") and type_str.endswith("]"):
        return type_str
    raise SchemaError(f"Unsupported registry type '{type_str}'.")


def arrow_type_str(dtype: pa.DataType) -> str:
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"
    if pa.types.is_int8(dtype):
        return "int8"
    if pa.types.is_int16(dtype):
        return "int16"
    if pa.types.is_int32(dtype):
        return "int32"
    if pa.types.is_int64(dtype):
        return "int64"
    if pa.types.is_uint8(dtype):
        return "uint8"
    if pa.types.is_uint16(dtype):
        return "uint16"
    if pa.types.is_uint32(dtype):
        return "uint32"
    if pa.types.is_uint64(dtype):
        return "uint64"
    if pa.types.is_float16(dtype):
        return "float16"
    if pa.types.is_float32(dtype):
        return "float32"
    if pa.types.is_float64(dtype):
        return "float64"
    if pa.types.is_boolean(dtype):
        return "bool"
    if pa.types.is_timestamp(dtype):
        tz = dtype.tz
        unit = dtype.unit
        if tz:
            return f"timestamp[{unit}, {tz}]"
        return f"timestamp[{unit}]"
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return f"list<{arrow_type_str(dtype.value_type)}>"
    raise SchemaError(f"Unsupported Arrow type '{dtype}'.")
