"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/workspace_storage/models.py

Typed values for the neutral workspace-storage contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SCHEMA_ID = "dnadesign.workspace-storage/v1"
MANIFEST_NAME = "workspace.storage.json"


class WorkspaceStorageError(ValueError):
    """Raised when a workspace-storage manifest or its bytes violate the contract."""


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
    """One manifest-declared file identity relative to the workspace root."""

    relative_path: str
    digest: str


@dataclass(frozen=True, slots=True)
class WorkspaceStorageManifest:
    """Parsed storage metadata independent of filesystem verification."""

    workspace_id: str
    owner_repository: str
    owner_tool: str
    workspace_schema: str
    workspace_schema_version: str
    producer_revision: str
    storage_class: StorageClass
    retention_policy: RetentionPolicy
    demo: bool
    inputs: tuple[StoredResource, ...]
    artifacts: tuple[StoredResource, ...]
    original_execution_path: str | None
    schema: str = SCHEMA_ID


@dataclass(frozen=True, slots=True)
class VerifiedStoredResource:
    """A confined file whose bytes match its manifest digest."""

    relative_path: str
    path: Path
    digest: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class VerifiedWorkspaceStorage:
    """A source-closed workspace-storage manifest and all declared files."""

    root: Path
    manifest_path: Path
    manifest: WorkspaceStorageManifest
    inputs: tuple[VerifiedStoredResource, ...]
    artifacts: tuple[VerifiedStoredResource, ...]

    def summary(self) -> dict[str, object]:
        return {
            "artifact_count": len(self.artifacts),
            "input_count": len(self.inputs),
            "owner_repository": self.manifest.owner_repository,
            "owner_tool": self.manifest.owner_tool,
            "schema": self.manifest.schema,
            "status": "verified",
            "storage_class": self.manifest.storage_class.value,
            "workspace_id": self.manifest.workspace_id,
        }
