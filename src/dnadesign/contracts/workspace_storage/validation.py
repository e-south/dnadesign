"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/workspace_storage/validation.py

Filesystem closure for workspace-storage manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .loading import load_workspace_storage_manifest
from .models import (
    MANIFEST_NAME,
    StoredResource,
    VerifiedStoredResource,
    VerifiedWorkspaceStorage,
    WorkspaceStorageError,
)


def _git_checkout_ancestor(root: Path) -> Path | None:
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _verify_resource(root: Path, resource: StoredResource, *, kind: str) -> VerifiedStoredResource:
    source_path = root / resource.relative_path
    try:
        resolved = source_path.resolve(strict=True)
    except OSError as exc:
        raise WorkspaceStorageError(f"declared {kind} does not resolve: {resource.relative_path}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkspaceStorageError(f"declared {kind} escapes workspace root: {resource.relative_path}") from exc
    if not resolved.is_file():
        raise WorkspaceStorageError(f"declared {kind} is not a file: {resource.relative_path}")
    observed_digest = _sha256(resolved)
    if observed_digest != resource.digest:
        raise WorkspaceStorageError(
            f"declared {kind} digest mismatch for {resource.relative_path}: "
            f"expected {resource.digest}, observed {observed_digest}"
        )
    return VerifiedStoredResource(
        relative_path=resource.relative_path,
        path=resolved,
        digest=observed_digest,
        size_bytes=resolved.stat().st_size,
    )


def verify_workspace_storage(workspace_root: Path) -> VerifiedWorkspaceStorage:
    """Verify one explicit workspace root, its manifest, and all declared bytes."""

    root = Path(workspace_root).expanduser().resolve()
    if not root.is_dir():
        raise WorkspaceStorageError(f"workspace root is not a directory: {root}")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise WorkspaceStorageError(f"workspace root is missing {MANIFEST_NAME}: {root}")
    manifest = load_workspace_storage_manifest(manifest_path)
    checkout = _git_checkout_ancestor(root)
    if checkout is not None and not manifest.demo:
        raise WorkspaceStorageError(
            f"non-demo workspace cannot live inside a Git checkout: workspace={root}, checkout={checkout}"
        )

    declared_paths: set[str] = set()
    for resource in (*manifest.inputs, *manifest.artifacts):
        if resource.relative_path in declared_paths:
            raise WorkspaceStorageError(f"resource path is declared more than once: {resource.relative_path}")
        declared_paths.add(resource.relative_path)

    inputs = tuple(_verify_resource(root, item, kind="input") for item in manifest.inputs)
    artifacts = tuple(_verify_resource(root, item, kind="artifact") for item in manifest.artifacts)
    return VerifiedWorkspaceStorage(
        root=root,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        inputs=inputs,
        artifacts=artifacts,
    )
