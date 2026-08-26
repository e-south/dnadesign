"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/storage_objects/validation.py

Exact filesystem closure for storage objects and their routed root.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from .loading import load_storage_object_manifest
from .models import (
    MANIFEST_NAME,
    MAX_DEMO_BYTES,
    ObjectKind,
    StorageObjectError,
    StoredResource,
    VerifiedStorageObject,
    VerifiedStorageRoot,
    VerifiedStoredResource,
)

_SHELF_KINDS = {
    "workspaces": ObjectKind.WORKSPACE,
    "stores": ObjectKind.STORE,
    "tool-cache": ObjectKind.TOOL_CACHE,
}
_ALLOWED_ROOT_FILES = {"AGENTS.md"}


def _git_checkout_ancestor(root: Path, *, include_root: bool) -> Path | None:
    candidates = (root, *root.parents) if include_root else root.parents
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def storage_file_paths(root: Path) -> tuple[Path, ...]:
    """Return every regular file while rejecting symlinks anywhere below root."""

    files: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory.is_symlink():
                relative = directory.relative_to(root).as_posix()
                raise StorageObjectError(f"symlink is not allowed: {relative}")
        for file_name in file_names:
            path = current_path / file_name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise StorageObjectError(f"symlink is not allowed: {relative}")
            if path.is_file():
                files.append(path)
    return tuple(files)


def _verify_resource(root: Path, resource: StoredResource) -> VerifiedStoredResource:
    source_path = root / resource.relative_path
    if source_path.is_symlink():
        raise StorageObjectError(f"symlink is not allowed: {resource.relative_path}")
    try:
        resolved = source_path.resolve(strict=True)
    except OSError as exc:
        raise StorageObjectError(f"declared resource does not resolve: {resource.relative_path}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StorageObjectError(f"declared resource escapes storage root: {resource.relative_path}") from exc
    if not resolved.is_file():
        raise StorageObjectError(f"declared resource is not a file: {resource.relative_path}")
    observed_digest = _sha256(resolved)
    if observed_digest != resource.digest:
        raise StorageObjectError(
            f"declared resource digest mismatch for {resource.relative_path}: "
            f"expected {resource.digest}, observed {observed_digest}"
        )
    return VerifiedStoredResource(
        relative_path=resource.relative_path,
        path=resolved,
        digest=observed_digest,
        role=resource.role,
        size_bytes=resolved.stat().st_size,
    )


def _verify_demo(checkout: Path, verified: VerifiedStorageObject) -> None:
    total_bytes = verified.manifest_path.stat().st_size + sum(resource.size_bytes for resource in verified.resources)
    if total_bytes > MAX_DEMO_BYTES:
        raise StorageObjectError(f"demo exceeds {MAX_DEMO_BYTES} bytes: {total_bytes}")
    for path in (verified.manifest_path, *(resource.path for resource in verified.resources)):
        relative = path.relative_to(checkout).as_posix()
        completed = subprocess.run(
            ["git", "-C", str(checkout), "ls-files", "--error-unmatch", "--", relative],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise StorageObjectError(f"demo file is not tracked: {relative}")


def verify_storage_object(storage_root: Path) -> VerifiedStorageObject:
    """Verify one explicit storage object and require exact file closure."""

    requested_root = Path(storage_root).expanduser()
    if requested_root.is_symlink():
        raise StorageObjectError(f"storage object root must not be a symlink: {requested_root}")
    root = requested_root.resolve()
    if not root.is_dir():
        raise StorageObjectError(f"storage object root is not a directory: {root}")
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise StorageObjectError(f"storage object manifest must not be a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise StorageObjectError(f"storage object root is missing {MANIFEST_NAME}: {root}")
    manifest = load_storage_object_manifest(manifest_path)

    declared_paths: set[str] = set()
    for resource in manifest.resources:
        if resource.relative_path == MANIFEST_NAME:
            raise StorageObjectError(f"manifest cannot declare itself: {MANIFEST_NAME}")
        if resource.relative_path in declared_paths:
            raise StorageObjectError(f"resource path is declared more than once: {resource.relative_path}")
        declared_paths.add(resource.relative_path)

    resources = tuple(_verify_resource(root, resource) for resource in manifest.resources)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in storage_file_paths(root)
        if path.name != MANIFEST_NAME or path.parent != root
    }
    undeclared = sorted(actual_paths - declared_paths)
    if undeclared:
        raise StorageObjectError(f"undeclared files: {', '.join(undeclared)}")
    missing = sorted(declared_paths - actual_paths)
    if missing:
        raise StorageObjectError(f"declared files are missing: {', '.join(missing)}")

    verified = VerifiedStorageObject(
        root=root,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        resources=resources,
    )
    checkout = _git_checkout_ancestor(
        root,
        include_root=manifest.object_kind is not ObjectKind.TOOL_CACHE,
    )
    if checkout is not None:
        if not manifest.demo:
            raise StorageObjectError(
                f"non-demo storage object cannot live inside a Git checkout: object={root}, checkout={checkout}"
            )
        _verify_demo(checkout, verified)
    return verified


def verify_storage_root(storage_root: Path) -> VerifiedStorageRoot:
    """Verify routed storage shelves and every contained object."""

    requested_root = Path(storage_root).expanduser()
    if requested_root.is_symlink():
        raise StorageObjectError(f"storage root must not be a symlink: {requested_root}")
    root = requested_root.resolve()
    if not root.is_dir():
        raise StorageObjectError(f"storage root is not a directory: {root}")
    allowed_shelves = set(_SHELF_KINDS) | _ALLOWED_ROOT_FILES
    unexpected_root_paths = sorted(path.name for path in root.iterdir() if path.name not in allowed_shelves)
    if unexpected_root_paths:
        raise StorageObjectError(f"unexpected path in storage root: {', '.join(unexpected_root_paths)}")
    routing_file = root / "AGENTS.md"
    if routing_file.is_symlink():
        raise StorageObjectError(f"storage root routing file must not be a symlink: {routing_file}")
    if routing_file.exists() and not routing_file.is_file():
        raise StorageObjectError(f"storage root routing file must be a regular file: {routing_file}")
    objects: list[VerifiedStorageObject] = []
    identities: set[tuple[str, str, str]] = set()
    for shelf_name, expected_kind in _SHELF_KINDS.items():
        shelf = root / shelf_name
        if shelf.is_symlink():
            raise StorageObjectError(f"storage shelf must not be a symlink: {shelf}")
        if not shelf.is_dir():
            raise StorageObjectError(f"storage root is missing shelf {shelf_name!r}")
        unexpected_shelf_paths = sorted(path.name for path in shelf.iterdir() if not path.is_dir())
        if unexpected_shelf_paths:
            raise StorageObjectError(
                f"unexpected path in storage shelf {shelf_name!r}: {', '.join(unexpected_shelf_paths)}"
            )
        for owner_directory in sorted(path for path in shelf.iterdir() if path.is_dir()):
            if owner_directory.is_symlink():
                raise StorageObjectError(f"storage shelf owner must not be a symlink: {owner_directory}")
            unexpected_owner_paths = sorted(path.name for path in owner_directory.iterdir() if not path.is_dir())
            if unexpected_owner_paths:
                raise StorageObjectError(
                    f"unexpected path in storage owner directory {owner_directory.name!r}: "
                    f"{', '.join(unexpected_owner_paths)}"
                )
            for object_directory in sorted(path for path in owner_directory.iterdir() if path.is_dir()):
                if object_directory.is_symlink():
                    raise StorageObjectError(f"storage object directory must not be a symlink: {object_directory}")
                verified = verify_storage_object(object_directory)
                manifest = verified.manifest
                if manifest.object_kind is not expected_kind:
                    raise StorageObjectError(
                        f"object_kind {manifest.object_kind.value!r} does not match shelf {shelf_name!r}"
                    )
                if manifest.owner_tool != owner_directory.name:
                    raise StorageObjectError(
                        f"owner_tool {manifest.owner_tool!r} does not match directory {owner_directory.name!r}"
                    )
                if manifest.storage_id != object_directory.name:
                    raise StorageObjectError(
                        f"storage_id {manifest.storage_id!r} does not match directory {object_directory.name!r}"
                    )
                identity = (
                    manifest.owner_repository,
                    manifest.owner_tool,
                    manifest.storage_id,
                )
                if identity in identities:
                    raise StorageObjectError(f"storage identity is duplicated: {identity}")
                identities.add(identity)
                objects.append(verified)
    return VerifiedStorageRoot(root=root, objects=tuple(objects))
