"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/workspace_storage/__init__.py

Publishes the neutral, fail-fast storage envelope for external tool workspaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .loading import load_workspace_storage_manifest
from .models import (
    MANIFEST_NAME,
    SCHEMA_ID,
    RetentionPolicy,
    StorageClass,
    StoredResource,
    VerifiedStoredResource,
    VerifiedWorkspaceStorage,
    WorkspaceStorageError,
    WorkspaceStorageManifest,
)
from .validation import verify_workspace_storage

__all__ = [
    "MANIFEST_NAME",
    "SCHEMA_ID",
    "RetentionPolicy",
    "StorageClass",
    "StoredResource",
    "VerifiedStoredResource",
    "VerifiedWorkspaceStorage",
    "WorkspaceStorageError",
    "WorkspaceStorageManifest",
    "load_workspace_storage_manifest",
    "verify_workspace_storage",
]
