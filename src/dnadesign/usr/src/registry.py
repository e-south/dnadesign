"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/registry.py

Namespace registry loading and validation for USR overlays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pyarrow as pa
import yaml

from .errors import SchemaError

REGISTRY_FILENAME = "registry.yaml"
USR_STATE_NAMESPACE = "usr_state"


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


USR_STATE_COLUMNS: list[RegistryColumn] = [
    RegistryColumn("usr_state__masked", "bool"),
    RegistryColumn("usr_state__qc_status", "string"),
    RegistryColumn("usr_state__split", "string"),
    RegistryColumn("usr_state__supersedes", "string"),
    RegistryColumn("usr_state__lineage", "list<string>"),
]


def registry_path(root: Path) -> Path:
    return Path(root) / REGISTRY_FILENAME


def load_registry(root: Path, *, required: bool) -> Dict[str, RegistryEntry]:
    path = registry_path(root)
    if not path.exists():
        if required:
            raise SchemaError(f"Registry required but not found: {path}. Create it with `usr namespace register ...`.")
        return {}
    return _load_registry_file(path)


def load_registry_file(path: Path) -> Dict[str, RegistryEntry]:
    return _load_registry_file(Path(path))


def _load_registry_file(path: Path) -> Dict[str, RegistryEntry]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    namespaces = data.get("namespaces") or {}
    if not isinstance(namespaces, dict):
        raise SchemaError("Registry file must contain a 'namespaces' mapping.")
    out: Dict[str, RegistryEntry] = {}
    for ns, entry in namespaces.items():
        out[str(ns)] = _parse_entry(str(ns), entry)
    _ensure_usr_state_entry(out)
    return out


def usr_state_entry() -> RegistryEntry:
    return RegistryEntry(
        namespace=USR_STATE_NAMESPACE,
        owner="usr",
        description="Reserved record-state overlay (masked/qc/split/lineage).",
        columns=list(USR_STATE_COLUMNS),
    )


def _ensure_usr_state_entry(entries: Dict[str, RegistryEntry]) -> None:
    if USR_STATE_NAMESPACE not in entries:
        raise SchemaError(
            "Registry must include reserved namespace 'usr_state'. Add usr_state columns to registry.yaml."
        )
    expected = {c.name: c.type for c in USR_STATE_COLUMNS}
    actual = {c.name: c.type for c in entries[USR_STATE_NAMESPACE].columns}
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


def save_registry(root: Path, entries: Dict[str, RegistryEntry]) -> Path:
    path = registry_path(root)
    payload = _registry_payload(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
    return path


def registry_hash(root: Path, *, required: bool) -> Optional[str]:
    path = registry_path(root)
    if not path.exists():
        if required:
            raise SchemaError(f"Registry required but not found: {path}.")
        return None
    data = registry_bytes(root)
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def registry_bytes(root: Path) -> bytes:
    entries = load_registry(root, required=True)
    payload = _registry_payload(entries)
    text = yaml.safe_dump(payload, sort_keys=True)
    return text.encode("utf-8")


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
    if USR_STATE_NAMESPACE not in entries and namespace != USR_STATE_NAMESPACE:
        entries[USR_STATE_NAMESPACE] = usr_state_entry()
    if namespace in entries and not overwrite:
        raise SchemaError(f"Namespace '{namespace}' already registered. Use --overwrite to replace.")
    cols = list(columns)
    if not cols:
        raise SchemaError("Registry entry must include at least one column.")
    _validate_columns(namespace, cols)
    if namespace == USR_STATE_NAMESPACE:
        expected = {c.name: c.type for c in USR_STATE_COLUMNS}
        actual = {c.name: c.type for c in cols}
        if expected != actual:
            raise SchemaError("Reserved namespace 'usr_state' must match the standard schema.")
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
    if type_str.startswith("fixed_size_list<"):
        inner, size = _parse_fixed_size_list(type_str)
        return f"fixed_size_list<{parse_type_str(inner)}>[{size}]"
    if type_str.startswith("list<") and type_str.endswith(">"):
        inner = type_str[len("list<") : -1].strip()
        return f"list<{parse_type_str(inner)}>"
    if type_str.startswith("struct<") and type_str.endswith(">"):
        inner = type_str[len("struct<") : -1].strip()
        fields = _split_top_level(inner)
        if not fields:
            raise SchemaError("Struct type must include at least one field.")
        parsed_fields = []
        for field in fields:
            if ":" not in field:
                raise SchemaError(f"Struct field '{field}' must be name:type.")
            name, inner_type = field.split(":", 1)
            name = name.strip()
            inner_type = inner_type.strip()
            if not name or not inner_type:
                raise SchemaError(f"Struct field '{field}' must be name:type.")
            parsed_fields.append(f"{name}:{parse_type_str(inner_type)}")
        return f"struct<{','.join(parsed_fields)}>"
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
    if pa.types.is_fixed_size_list(dtype):
        return f"fixed_size_list<{arrow_type_str(dtype.value_type)}>[{dtype.list_size}]"
    if pa.types.is_struct(dtype):
        fields = ",".join(f"{field.name}:{arrow_type_str(field.type)}" for field in dtype)
        return f"struct<{fields}>"
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return f"list<{arrow_type_str(dtype.value_type)}>"
    raise SchemaError(f"Unsupported Arrow type '{dtype}'.")


def _registry_payload(entries: Dict[str, RegistryEntry]) -> dict:
    return {
        "namespaces": {
            ns: {
                "owner": entry.owner,
                "description": entry.description,
                "columns": [{"name": c.name, "type": c.type} for c in entry.columns],
            }
            for ns, entry in sorted(entries.items())
        }
    }


def _split_top_level(spec: str) -> List[str]:
    parts: List[str] = []
    if not spec:
        return parts
    depth = 0
    start = 0
    for i, ch in enumerate(spec):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            part = spec[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1
    tail = spec[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_fixed_size_list(type_str: str) -> tuple[str, int]:
    prefix = "fixed_size_list<"
    if not type_str.startswith(prefix):
        raise SchemaError(f"Invalid fixed_size_list type '{type_str}'.")
    inner_spec = type_str[len(prefix) :]
    depth = 0
    inner_end = None
    for i, ch in enumerate(inner_spec):
        if ch == "<":
            depth += 1
        elif ch == ">":
            if depth == 0:
                inner_end = i
                break
            depth -= 1
    if inner_end is None:
        raise SchemaError(f"Invalid fixed_size_list type '{type_str}'.")
    inner = inner_spec[:inner_end].strip()
    rest = inner_spec[inner_end + 1 :].strip()
    if not rest.startswith("[") or not rest.endswith("]"):
        raise SchemaError(f"Invalid fixed_size_list size in '{type_str}'.")
    size_str = rest[1:-1].strip()
    if not size_str.isdigit():
        raise SchemaError(f"Invalid fixed_size_list size in '{type_str}'.")
    size = int(size_str)
    if size <= 0:
        raise SchemaError(f"fixed_size_list size must be positive in '{type_str}'.")
    return inner, size
