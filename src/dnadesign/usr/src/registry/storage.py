"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/registry/storage.py

USR registry persistence, hashing, and cache helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..contracts import SchemaError
from .models import (
    REGISTRY_FILENAME,
    USR_STATE_COLUMNS,
    USR_STATE_NAMESPACE,
    RegistryColumn,
    RegistryEntry,
    _clone_registry_entries,
    derived_entry,
    seq_annot_entry,
    usr_label_entry,
    usr_state_entry,
)
from .validation import _ensure_usr_state_entry, _parse_entry, _validate_columns, registry_entry


@dataclass
class _RegistryCacheEntry:
    mtime_ns: int
    size: int
    entries: dict[str, RegistryEntry]
    canonical_bytes: bytes | None = None
    canonical_hash: str | None = None


_REGISTRY_CACHE: dict[str, _RegistryCacheEntry] = {}
_REGISTRY_CACHE_MAX = 4_096


def registry_path(root: Path) -> Path:
    return Path(root) / REGISTRY_FILENAME


def load_registry(root: Path, *, required: bool) -> dict[str, RegistryEntry]:
    path = registry_path(root)
    if not path.exists():
        if required:
            raise SchemaError(f"Registry required but not found: {path}. Create it with `usr namespace register ...`.")
        return {}
    return _load_registry_file(path)


def load_registry_file(path: Path) -> dict[str, RegistryEntry]:
    return _load_registry_file(Path(path))


def _registry_cache_entry(path: Path) -> _RegistryCacheEntry:
    resolved_path = Path(path).resolve()
    cache_key = str(resolved_path)
    try:
        stat = resolved_path.stat()
    except FileNotFoundError as exc:
        raise SchemaError(f"Registry required but not found: {resolved_path}.") from exc
    stat_key = (int(stat.st_mtime_ns), int(stat.st_size))
    cached = _REGISTRY_CACHE.get(cache_key)
    if cached is not None and (cached.mtime_ns, cached.size) == stat_key:
        return cached

    with resolved_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    namespaces = data.get("namespaces") or {}
    if not isinstance(namespaces, dict):
        raise SchemaError("Registry file must contain a 'namespaces' mapping.")
    out: dict[str, RegistryEntry] = {}
    for namespace, entry in namespaces.items():
        out[str(namespace)] = _parse_entry(str(namespace), entry)
    _ensure_usr_state_entry(out)
    cache_entry = _RegistryCacheEntry(
        mtime_ns=stat_key[0],
        size=stat_key[1],
        entries=_clone_registry_entries(out),
    )
    _REGISTRY_CACHE[cache_key] = cache_entry
    if len(_REGISTRY_CACHE) > _REGISTRY_CACHE_MAX:
        _REGISTRY_CACHE.clear()
    return cache_entry


def _load_registry_file(path: Path) -> dict[str, RegistryEntry]:
    cache_entry = _registry_cache_entry(path)
    return _clone_registry_entries(cache_entry.entries)


def _registry_canonical_bytes(path: Path) -> bytes:
    cache_entry = _registry_cache_entry(path)
    if cache_entry.canonical_bytes is None:
        payload = _registry_payload(cache_entry.entries)
        cache_entry.canonical_bytes = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
        cache_entry.canonical_hash = hashlib.sha256(cache_entry.canonical_bytes).hexdigest()
    return cache_entry.canonical_bytes


def _registry_canonical_hash(path: Path) -> str:
    cache_entry = _registry_cache_entry(path)
    if cache_entry.canonical_hash is None:
        _registry_canonical_bytes(path)
    assert cache_entry.canonical_hash is not None
    return cache_entry.canonical_hash


def save_registry(root: Path, entries: dict[str, RegistryEntry]) -> Path:
    path = registry_path(root)
    payload = _registry_payload(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
    _REGISTRY_CACHE.pop(str(path.resolve()), None)
    return path


def registry_hash(root: Path, *, required: bool) -> str | None:
    path = registry_path(root)
    if not path.exists():
        if required:
            raise SchemaError(f"Registry required but not found: {path}.")
        return None
    return _registry_canonical_hash(path)


def registry_bytes(root: Path) -> bytes:
    return _registry_canonical_bytes(registry_path(root))


def registry_bytes_for_entries(entries: dict[str, RegistryEntry]) -> bytes:
    payload = _registry_payload(entries)
    text = yaml.safe_dump(payload, sort_keys=True)
    return text.encode("utf-8")


def registry_hash_for_entries(entries: dict[str, RegistryEntry]) -> str:
    h = hashlib.sha256()
    h.update(registry_bytes_for_entries(entries))
    return h.hexdigest()


def namespace_contract_hash(root: Path, namespace: str, *, required: bool) -> str | None:
    path = registry_path(root)
    if not path.exists():
        if required:
            raise SchemaError(f"Registry required but not found: {path}.")
        return None
    entries = load_registry(root, required=True)
    return namespace_contract_hash_for_entries(entries, namespace)


def namespace_contract_hash_for_entries(entries: dict[str, RegistryEntry], namespace: str) -> str:
    entry = registry_entry(entries, namespace)
    payload = {
        "namespace": entry.namespace,
        "columns": [{"name": column.name, "type": column.type} for column in entry.columns],
    }
    text = yaml.safe_dump(payload, sort_keys=True)
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def register_namespace(
    root: Path,
    *,
    namespace: str,
    columns: list[RegistryColumn] | tuple[RegistryColumn, ...],
    owner: str | None = None,
    description: str | None = None,
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
        expected = {column.name: column.type for column in USR_STATE_COLUMNS}
        actual = {column.name: column.type for column in cols}
        if expected != actual:
            raise SchemaError("Reserved namespace 'usr_state' must match the standard schema.")
    entries[namespace] = RegistryEntry(
        namespace=namespace,
        owner=owner,
        description=description,
        columns=cols,
    )
    return save_registry(root, entries)


def ensure_registry_entries(root: Path, entries: list[RegistryEntry] | tuple[RegistryEntry, ...]) -> Path:
    current_entries = load_registry(root, required=False)
    if USR_STATE_NAMESPACE not in current_entries:
        current_entries[USR_STATE_NAMESPACE] = usr_state_entry()
    changed = False
    for required_entry in entries:
        existing = current_entries.get(required_entry.namespace)
        if existing is None:
            current_entries[required_entry.namespace] = RegistryEntry(
                namespace=required_entry.namespace,
                owner=required_entry.owner,
                description=required_entry.description,
                columns=list(required_entry.columns),
            )
            changed = True
            continue
        expected_columns = {column.name: column.type for column in required_entry.columns}
        current_columns = {column.name: column.type for column in existing.columns}
        for name, expected_type in expected_columns.items():
            observed_type = current_columns.get(name)
            if observed_type is None:
                existing.columns.append(RegistryColumn(name=name, type=expected_type))
                changed = True
                continue
            if observed_type != expected_type:
                raise SchemaError(
                    f"Registry namespace '{required_entry.namespace}' column '{name}' has type "
                    f"'{observed_type}', expected '{expected_type}'."
                )
        if existing.owner is None and required_entry.owner is not None:
            existing.owner = required_entry.owner
            changed = True
        if existing.description is None and required_entry.description is not None:
            existing.description = required_entry.description
            changed = True
        _validate_columns(existing.namespace, existing.columns)
    if not changed:
        return registry_path(root)
    return save_registry(root, current_entries)


def ensure_sequence_contract_namespaces(root: Path) -> Path:
    return ensure_registry_entries(
        root,
        entries=(
            usr_label_entry(),
            seq_annot_entry(),
            derived_entry(),
        ),
    )


def _registry_payload(entries: dict[str, RegistryEntry]) -> dict:
    return {
        "namespaces": {
            namespace: {
                "owner": entry.owner,
                "description": entry.description,
                "columns": [{"name": column.name, "type": column.type} for column in entry.columns],
            }
            for namespace, entry in sorted(entries.items())
        }
    }
