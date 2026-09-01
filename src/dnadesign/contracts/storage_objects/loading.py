"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/storage_objects/loading.py

Strict JSON loading for storage-object manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from .models import (
    SCHEMA_ID,
    ObjectKind,
    ResourceRole,
    RetentionPolicy,
    StorageClass,
    StorageObjectError,
    StorageObjectManifest,
    StoredResource,
)

_MANIFEST_FIELDS = {
    "schema",
    "storage_id",
    "owner_repository",
    "owner_tool",
    "object_kind",
    "content_schema",
    "content_schema_version",
    "producer_revision",
    "storage_class",
    "retention_policy",
    "demo",
    "resources",
}
_OPTIONAL_MANIFEST_FIELDS = {"original_execution_path"}
_RESOURCE_FIELDS = {"path", "digest", "role"}
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_RETENTION_BY_STORAGE_CLASS = {
    StorageClass.AUTHORITATIVE: {RetentionPolicy.RETAIN, RetentionPolicy.REVIEW_BEFORE_DELETE},
    StorageClass.REPRODUCIBLE: {
        RetentionPolicy.RETAIN,
        RetentionPolicy.REBUILDABLE,
        RetentionPolicy.REVIEW_BEFORE_DELETE,
    },
    StorageClass.CACHE: {RetentionPolicy.REBUILDABLE},
    StorageClass.COLD: {RetentionPolicy.COLD, RetentionPolicy.RETAIN},
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StorageObjectError(f"storage object JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StorageObjectError(f"{label} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    if missing:
        raise StorageObjectError(f"{label} is missing required fields: {', '.join(missing)}")
    unsupported = sorted(set(value) - required - optional)
    if unsupported:
        raise StorageObjectError(f"{label} has unsupported fields: {', '.join(unsupported)}")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StorageObjectError(f"{label} must be a non-empty string")
    return value.strip()


def _identifier(value: object, *, label: str) -> str:
    token = _text(value, label=label)
    if not _IDENTIFIER.fullmatch(token):
        raise StorageObjectError(f"{label} must be a lowercase identifier")
    return token


def normalize_relative_path(value: object, *, label: str) -> str:
    """Return one confined portable file path."""

    token = _text(value, label=label)
    if "\x00" in token:
        raise StorageObjectError(f"{label} must not contain NUL bytes")
    candidate = PurePosixPath(token)
    unsafe_parts = any(part in {".", ".."} for part in candidate.parts)
    if "\\" in token or candidate.is_absolute() or not candidate.parts or unsafe_parts:
        raise StorageObjectError(f"{label} must be a confined relative path")
    return candidate.as_posix()


def _digest(value: object, *, label: str) -> str:
    token = _text(value, label=label)
    if not _DIGEST.fullmatch(token):
        raise StorageObjectError(f"{label} must be a lowercase sha256 digest")
    return token


def _resources(value: object) -> tuple[StoredResource, ...]:
    if not isinstance(value, list):
        raise StorageObjectError("resources must be an array")
    resources: list[StoredResource] = []
    resource_paths: set[str] = set()
    for index, raw_item in enumerate(value):
        label = f"resources[{index}]"
        item = _mapping(raw_item, label=label)
        _exact_fields(item, required=_RESOURCE_FIELDS, label=label)
        try:
            role = ResourceRole(_text(item["role"], label=f"{label}.role"))
        except ValueError as exc:
            raise StorageObjectError(f"unsupported resource role {item['role']!r}") from exc
        relative_path = normalize_relative_path(item["path"], label=f"{label}.path")
        if relative_path in resource_paths:
            raise StorageObjectError(f"resource path is declared more than once: {relative_path}")
        resource_paths.add(relative_path)
        resources.append(
            StoredResource(
                relative_path=relative_path,
                digest=_digest(item["digest"], label=f"{label}.digest"),
                role=role,
            )
        )
    return tuple(resources)


def load_storage_object_manifest_bytes(
    manifest_bytes: bytes,
    *,
    source_label: str,
) -> StorageObjectManifest:
    """Parse one already-read manifest buffer without reopening its source."""

    try:
        raw = json.loads(manifest_bytes.decode("utf-8"), object_pairs_hook=_strict_object)
    except StorageObjectError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StorageObjectError(f"cannot parse storage object manifest {source_label}: {exc}") from exc

    payload = _mapping(raw, label="storage object manifest")
    _exact_fields(
        payload,
        required=_MANIFEST_FIELDS,
        optional=_OPTIONAL_MANIFEST_FIELDS,
        label="storage object manifest",
    )
    if payload["schema"] != SCHEMA_ID:
        raise StorageObjectError(f"unsupported storage object schema {payload['schema']!r}; expected {SCHEMA_ID!r}")
    if type(payload["demo"]) is not bool:
        raise StorageObjectError("storage object manifest.demo must be a boolean")
    try:
        object_kind = ObjectKind(_text(payload["object_kind"], label="object_kind"))
    except ValueError as exc:
        raise StorageObjectError(f"unsupported object_kind {payload['object_kind']!r}") from exc
    try:
        storage_class = StorageClass(_text(payload["storage_class"], label="storage_class"))
    except ValueError as exc:
        raise StorageObjectError(f"unsupported storage_class {payload['storage_class']!r}") from exc
    try:
        retention_policy = RetentionPolicy(_text(payload["retention_policy"], label="retention_policy"))
    except ValueError as exc:
        raise StorageObjectError(f"unsupported retention_policy {payload['retention_policy']!r}") from exc
    if retention_policy not in _RETENTION_BY_STORAGE_CLASS[storage_class]:
        raise StorageObjectError(
            f"retention_policy {retention_policy.value!r} is incompatible with storage_class {storage_class.value!r}"
        )
    if object_kind is ObjectKind.TOOL_CACHE and storage_class is not StorageClass.CACHE:
        raise StorageObjectError("tool-cache objects require storage_class 'cache'")
    resources = _resources(payload["resources"])
    if object_kind is ObjectKind.TOOL_CACHE and any(resource.role is not ResourceRole.CACHE for resource in resources):
        raise StorageObjectError("tool-cache objects require every resource role to be 'cache'")

    original_path = payload.get("original_execution_path")
    return StorageObjectManifest(
        storage_id=_identifier(payload["storage_id"], label="storage_id"),
        owner_repository=_identifier(payload["owner_repository"], label="owner_repository"),
        owner_tool=_identifier(payload["owner_tool"], label="owner_tool"),
        object_kind=object_kind,
        content_schema=_identifier(payload["content_schema"], label="content_schema"),
        content_schema_version=_text(payload["content_schema_version"], label="content_schema_version"),
        producer_revision=_text(payload["producer_revision"], label="producer_revision"),
        storage_class=storage_class,
        retention_policy=retention_policy,
        demo=payload["demo"],
        resources=resources,
        original_execution_path=(
            None if original_path is None else _text(original_path, label="original_execution_path")
        ),
    )


def load_storage_object_manifest(manifest_path: Path) -> StorageObjectManifest:
    """Read and parse one exact storage-object manifest without inferring defaults."""

    source = Path(manifest_path)
    if source.is_symlink():
        raise StorageObjectError(f"storage object manifest must not be a symlink: {source}")
    try:
        manifest_bytes = source.read_bytes()
    except OSError as exc:
        raise StorageObjectError(f"cannot read storage object manifest {source}: {exc}") from exc
    return load_storage_object_manifest_bytes(manifest_bytes, source_label=str(source))
