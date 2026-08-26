"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/workspace_storage/loading.py

Strict JSON loading for workspace-storage manifests.

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
    RetentionPolicy,
    StorageClass,
    StoredResource,
    WorkspaceStorageError,
    WorkspaceStorageManifest,
)

_MANIFEST_FIELDS = {
    "schema",
    "workspace_id",
    "owner_repository",
    "owner_tool",
    "workspace_schema",
    "workspace_schema_version",
    "producer_revision",
    "storage_class",
    "retention_policy",
    "demo",
    "inputs",
    "artifacts",
}
_OPTIONAL_MANIFEST_FIELDS = {"original_execution_path"}
_RESOURCE_FIELDS = {"path", "digest"}
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
            raise WorkspaceStorageError(f"workspace storage JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WorkspaceStorageError(f"{label} must be an object")
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
        raise WorkspaceStorageError(f"{label} is missing required fields: {', '.join(missing)}")
    unsupported = sorted(set(value) - required - optional)
    if unsupported:
        raise WorkspaceStorageError(f"{label} has unsupported fields: {', '.join(unsupported)}")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceStorageError(f"{label} must be a non-empty string")
    return value.strip()


def _identifier(value: object, *, label: str) -> str:
    token = _text(value, label=label)
    if not _IDENTIFIER.fullmatch(token):
        raise WorkspaceStorageError(f"{label} must be a lowercase identifier")
    return token


def _digest(value: object, *, label: str) -> str:
    token = _text(value, label=label)
    if not _DIGEST.fullmatch(token):
        raise WorkspaceStorageError(f"{label} must be a lowercase sha256 digest")
    return token


def _relative_path(value: object, *, label: str) -> str:
    token = _text(value, label=label)
    candidate = PurePosixPath(token)
    unsafe_parts = any(part in {".", ".."} for part in candidate.parts)
    if "\\" in token or candidate.is_absolute() or not candidate.parts or unsafe_parts:
        raise WorkspaceStorageError(f"{label} must be a confined relative path")
    return candidate.as_posix()


def _resources(value: object, *, label: str) -> tuple[StoredResource, ...]:
    if not isinstance(value, list):
        raise WorkspaceStorageError(f"{label} must be an array")
    resources: list[StoredResource] = []
    for index, raw_item in enumerate(value):
        item_label = f"{label}[{index}]"
        item = _mapping(raw_item, label=item_label)
        _exact_fields(item, required=_RESOURCE_FIELDS, label=item_label)
        resources.append(
            StoredResource(
                relative_path=_relative_path(item["path"], label=f"{item_label}.path"),
                digest=_digest(item["digest"], label=f"{item_label}.digest"),
            )
        )
    return tuple(resources)


def load_workspace_storage_manifest(manifest_path: Path) -> WorkspaceStorageManifest:
    """Parse one exact manifest without inferring paths or defaults."""

    source = Path(manifest_path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except WorkspaceStorageError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceStorageError(f"cannot read workspace storage manifest {source}: {exc}") from exc

    payload = _mapping(raw, label="workspace storage manifest")
    _exact_fields(
        payload,
        required=_MANIFEST_FIELDS,
        optional=_OPTIONAL_MANIFEST_FIELDS,
        label="workspace storage manifest",
    )
    if payload["schema"] != SCHEMA_ID:
        raise WorkspaceStorageError(
            f"unsupported workspace storage schema {payload['schema']!r}; expected {SCHEMA_ID!r}"
        )
    if type(payload["demo"]) is not bool:
        raise WorkspaceStorageError("workspace storage manifest.demo must be a boolean")

    try:
        storage_class = StorageClass(_text(payload["storage_class"], label="storage_class"))
    except ValueError as exc:
        raise WorkspaceStorageError(f"unsupported storage_class {payload['storage_class']!r}") from exc
    try:
        retention_policy = RetentionPolicy(_text(payload["retention_policy"], label="retention_policy"))
    except ValueError as exc:
        raise WorkspaceStorageError(f"unsupported retention_policy {payload['retention_policy']!r}") from exc
    if retention_policy not in _RETENTION_BY_STORAGE_CLASS[storage_class]:
        raise WorkspaceStorageError(
            f"retention_policy {retention_policy.value!r} is incompatible with storage_class {storage_class.value!r}"
        )

    original_execution_path_raw = payload.get("original_execution_path")
    original_execution_path = (
        None
        if original_execution_path_raw is None
        else _text(original_execution_path_raw, label="original_execution_path")
    )
    return WorkspaceStorageManifest(
        workspace_id=_identifier(payload["workspace_id"], label="workspace_id"),
        owner_repository=_identifier(payload["owner_repository"], label="owner_repository"),
        owner_tool=_identifier(payload["owner_tool"], label="owner_tool"),
        workspace_schema=_identifier(payload["workspace_schema"], label="workspace_schema"),
        workspace_schema_version=_text(payload["workspace_schema_version"], label="workspace_schema_version"),
        producer_revision=_text(payload["producer_revision"], label="producer_revision"),
        storage_class=storage_class,
        retention_policy=retention_policy,
        demo=payload["demo"],
        inputs=_resources(payload["inputs"], label="inputs"),
        artifacts=_resources(payload["artifacts"], label="artifacts"),
        original_execution_path=original_execution_path,
    )
