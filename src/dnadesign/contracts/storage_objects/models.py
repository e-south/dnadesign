"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/storage_objects/models.py

Typed values for external storage objects and storage-root verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SCHEMA_ID = "dnadesign.storage-object/v1"
ROOT_SCHEMA_ID = "dnadesign.storage-root/v1"
MANIFEST_NAME = "storage.object.json"
LOCK_NAME = ".storage-object.lock"
MAX_DEMO_BYTES = 2_000_000


class StorageObjectError(ValueError):
    """Raised when a storage object or root violates its explicit contract."""


class StorageObjectPublicationUnsupported(StorageObjectError):
    """Raised when the filesystem lacks a required atomic publication primitive."""


class StorageObjectPublicationUncertain(StorageObjectError):
    """Raised when an interrupted atomic operation cannot prove the winning receipt."""


class ObjectKind(StrEnum):
    """Physical storage shelf without importing tool-owned content meaning."""

    WORKSPACE = "workspace"
    STORE = "store"
    TOOL_CACHE = "tool-cache"


class ResourceRole(StrEnum):
    """Storage-level role for one exact file."""

    INPUT = "input"
    ARTIFACT = "artifact"
    METADATA = "metadata"
    CACHE = "cache"


class StorageClass(StrEnum):
    AUTHORITATIVE = "authoritative"
    REPRODUCIBLE = "reproducible"
    CACHE = "cache"
    COLD = "cold"


class RetentionPolicy(StrEnum):
    RETAIN = "retain"
    REBUILDABLE = "rebuildable"
    REVIEW_BEFORE_DELETE = "review-before-delete"
    COLD = "cold"


@dataclass(frozen=True, slots=True)
class StoredResource:
    """One exact file identity relative to the storage-object root."""

    relative_path: str
    digest: str
    role: ResourceRole


@dataclass(frozen=True, slots=True)
class StorageObjectManifest:
    """Parsed storage metadata independent of filesystem verification."""

    storage_id: str
    owner_repository: str
    owner_tool: str
    object_kind: ObjectKind
    content_schema: str
    content_schema_version: str
    producer_revision: str
    storage_class: StorageClass
    retention_policy: RetentionPolicy
    demo: bool
    resources: tuple[StoredResource, ...]
    original_execution_path: str | None
    schema: str = SCHEMA_ID


@dataclass(frozen=True, slots=True)
class VerifiedStoredResource:
    """A confined regular file whose bytes match the declared digest."""

    relative_path: str
    path: Path
    digest: str
    role: ResourceRole
    size_bytes: int
    device_id: int
    inode: int


@dataclass(frozen=True, slots=True)
class VerifiedStorageObject:
    """A file-closed storage object ready for a tool-owned acceptance check."""

    root: Path
    manifest_path: Path
    manifest_digest: str
    manifest_device_id: int
    manifest_inode: int
    manifest: StorageObjectManifest
    resources: tuple[VerifiedStoredResource, ...]

    def summary(self) -> dict[str, object]:
        role_counts: dict[str, int] = {}
        for resource in self.resources:
            role_counts[resource.role.value] = role_counts.get(resource.role.value, 0) + 1
        return {
            "manifest_digest": self.manifest_digest,
            "object_kind": self.manifest.object_kind.value,
            "owner_repository": self.manifest.owner_repository,
            "owner_tool": self.manifest.owner_tool,
            "resource_count": len(self.resources),
            "resources_by_role": dict(sorted(role_counts.items())),
            "schema": self.manifest.schema,
            "status": "verified",
            "storage_class": self.manifest.storage_class.value,
            "storage_id": self.manifest.storage_id,
            "total_bytes": sum(resource.size_bytes for resource in self.resources),
        }


@dataclass(frozen=True, slots=True)
class VerifiedStorageRoot:
    """All convention-routed storage objects under one external storage root."""

    root: Path
    objects: tuple[VerifiedStorageObject, ...]

    def summary(self) -> dict[str, object]:
        kind_counts: dict[str, int] = {}
        owners: set[tuple[str, str]] = set()
        for storage_object in self.objects:
            kind = storage_object.manifest.object_kind.value
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            owners.add(
                (
                    storage_object.manifest.owner_repository,
                    storage_object.manifest.owner_tool,
                )
            )
        return {
            "object_count": len(self.objects),
            "objects_by_kind": dict(sorted(kind_counts.items())),
            "owner_count": len(owners),
            "schema": ROOT_SCHEMA_ID,
            "status": "verified",
        }
