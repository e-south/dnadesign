"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/storage_objects/inventory.py

Deterministic manifest generation for one pre-existing storage object.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from .loading import load_storage_object_manifest, normalize_relative_path
from .models import (
    LOCK_NAME,
    MANIFEST_NAME,
    SCHEMA_ID,
    ObjectKind,
    ResourceRole,
    RetentionPolicy,
    StorageClass,
    StorageObjectError,
)
from .validation import resolve_storage_path, storage_file_paths, verify_storage_object


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StorageObjectError(f"cannot read storage resource {path}: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def _write_manifest(
    manifest_path: Path,
    payload: dict[str, object],
    *,
    previous_bytes: bytes | None = None,
    allow_untracked_demo_manifest: bool = False,
) -> dict[str, object]:
    manifest_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = -1
    temporary: Path | None = None
    temporary_created = False
    previous_mode: int | None = None
    try:
        if previous_bytes is not None:
            previous_mode = stat.S_IMODE(manifest_path.stat(follow_symlinks=False).st_mode)
        else:
            root_mode = stat.S_IMODE(manifest_path.parent.stat(follow_symlinks=False).st_mode)
            previous_mode = 0o664 if root_mode & stat.S_IWGRP else 0o644
        descriptor, temporary_name = tempfile.mkstemp(
            dir=manifest_path.parent,
            prefix=f".{MANIFEST_NAME}.tmp-",
        )
        temporary = Path(temporary_name)
        temporary_created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(manifest_text)
        if previous_mode is not None:
            temporary.chmod(previous_mode, follow_symlinks=False)
        os.replace(temporary, manifest_path)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_created and temporary is not None:
            temporary.unlink(missing_ok=True)
        raise StorageObjectError(f"cannot write storage object manifest: {exc}") from exc
    try:
        summary = verify_storage_object(
            manifest_path.parent,
            _allow_untracked_demo_manifest=allow_untracked_demo_manifest,
        ).summary()
        if allow_untracked_demo_manifest:
            summary["status"] = "created-pending-git-add"
            object_root = shlex.quote(str(manifest_path.parent))
            summary["next_step"] = (
                f"git -C {object_root} add -- {MANIFEST_NAME} && dnadesign-storage validate {object_root}"
            )
        return summary
    except Exception as validation_error:
        if previous_bytes is None:
            manifest_path.unlink(missing_ok=True)
        else:
            restore_path: Path | None = None
            try:
                descriptor, restore_name = tempfile.mkstemp(
                    dir=manifest_path.parent,
                    prefix=f".{MANIFEST_NAME}.restore-",
                )
                restore_path = Path(restore_name)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(previous_bytes)
                if previous_mode is not None:
                    restore_path.chmod(previous_mode, follow_symlinks=False)
                os.replace(restore_path, manifest_path)
            except OSError as restore_error:
                if restore_path is not None:
                    restore_path.unlink(missing_ok=True)
                raise StorageObjectError(
                    f"storage object validation failed and the prior manifest could not be restored: {restore_error}"
                ) from validation_error
        raise


@contextmanager
def _manifest_lock(root: Path) -> Iterator[None]:
    lock_path = root / LOCK_NAME
    try:
        root_stat = root.stat(follow_symlinks=False)
        root_mode = stat.S_IMODE(root_stat.st_mode)
        if root_mode & stat.S_IWGRP and not root_mode & stat.S_ISGID:
            raise StorageObjectError(
                "group-writable storage object roots must set the setgid bit "
                "so coordination files inherit the shared group"
            )
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise StorageObjectError(f"storage object lock must be a regular file: {lock_path}")
        if lock_path.exists() and lock_path.stat(follow_symlinks=False).st_size != 0:
            raise StorageObjectError(f"storage object lock must be an empty coordination file: {lock_path}")
        if (
            root_mode & stat.S_IWGRP
            and lock_path.exists()
            and lock_path.stat(follow_symlinks=False).st_gid != root_stat.st_gid
        ):
            raise StorageObjectError(f"storage object lock does not inherit the shared object group: {lock_path}")
        if (
            root_mode & stat.S_IWGRP
            and lock_path.exists()
            and not lock_path.stat(follow_symlinks=False).st_mode & stat.S_IWGRP
        ):
            raise StorageObjectError(f"storage object lock must be group-writable in a shared object root: {lock_path}")
    except OSError as exc:
        raise StorageObjectError(f"cannot inspect storage object lock {lock_path}: {exc}") from exc
    lock_mode = 0o664 if root_mode & stat.S_IWGRP else 0o644
    lock = FileLock(lock_path, timeout=30, mode=lock_mode)
    try:
        lock.acquire()
    except Timeout as exc:
        raise StorageObjectError(f"timed out waiting for storage object manifest lock: {root}") from exc
    except OSError as exc:
        raise StorageObjectError(f"cannot acquire storage object manifest lock {lock_path}: {exc}") from exc
    try:
        lock_stat = lock_path.stat(follow_symlinks=False)
        if root_mode & stat.S_IWGRP and lock_stat.st_gid != root_stat.st_gid:
            raise StorageObjectError(f"storage object lock does not inherit the shared object group: {lock_path}")
        if root_mode & stat.S_IWGRP and not lock_stat.st_mode & stat.S_IWGRP:
            raise StorageObjectError(f"storage object lock must be group-writable in a shared object root: {lock_path}")
    except (OSError, StorageObjectError) as inspection_error:
        try:
            lock.release()
        except OSError as release_error:
            raise StorageObjectError(
                f"cannot inspect storage object lock after acquisition and lock {lock_path} could not be released: "
                f"{release_error}"
            ) from inspection_error
        if isinstance(inspection_error, StorageObjectError):
            raise
        raise StorageObjectError(
            f"cannot inspect storage object lock after acquisition {lock_path}: {inspection_error}"
        ) from inspection_error
    try:
        yield
    except BaseException as body_error:
        try:
            lock.release()
        except OSError as release_error:
            raise StorageObjectError(
                f"storage operation failed and manifest lock {lock_path} could not be released: {release_error}"
            ) from body_error
        raise
    else:
        try:
            lock.release()
        except OSError as exc:
            raise StorageObjectError(f"cannot release storage object manifest lock {lock_path}: {exc}") from exc


def _assert_no_ambiguous_manifest_staging(root: Path) -> None:
    """Fail closed rather than deleting a staging-shaped user file."""

    candidates = {root / f".{MANIFEST_NAME}.tmp"}
    candidates.update(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    candidates.update(root.glob(f".{MANIFEST_NAME}.restore-*"))
    present = sorted(path.name for path in candidates if path.exists() or path.is_symlink())
    if present:
        raise StorageObjectError(
            "storage object contains ambiguous manifest staging state; inspect and remove it explicitly: "
            + ", ".join(present)
        )


def inventory_storage_object(
    storage_root: Path,
    *,
    storage_id: str,
    owner_repository: str,
    owner_tool: str,
    object_kind: str,
    content_schema: str,
    content_schema_version: str,
    producer_revision: str,
    storage_class: str,
    retention_policy: str,
    input_paths: tuple[str, ...] = (),
    metadata_paths: tuple[str, ...] = (),
    cache_paths: tuple[str, ...] = (),
    original_execution_path: str | None = None,
    demo: bool = False,
) -> dict[str, object]:
    """Write one no-overwrite manifest, then verify the resulting object."""

    requested_root = Path(storage_root).expanduser()
    if requested_root.is_symlink():
        raise StorageObjectError(f"storage object root must not be a symlink: {requested_root}")
    root = resolve_storage_path(requested_root, label="storage object root")
    if not root.is_dir():
        raise StorageObjectError(f"storage object root is not a directory: {root}")
    try:
        parsed_kind = ObjectKind(object_kind)
        parsed_class = StorageClass(storage_class)
        parsed_retention = RetentionPolicy(retention_policy)
    except ValueError as exc:
        raise StorageObjectError(f"invalid inventory enum value: {exc}") from exc

    normalized_inputs = {normalize_relative_path(path, label="input path") for path in input_paths}
    normalized_metadata = {normalize_relative_path(path, label="metadata path") for path in metadata_paths}
    normalized_caches = {normalize_relative_path(path, label="cache path") for path in cache_paths}
    duplicate_roles = sorted(
        (normalized_inputs & normalized_metadata)
        | (normalized_inputs & normalized_caches)
        | (normalized_metadata & normalized_caches)
    )
    if duplicate_roles:
        raise StorageObjectError(f"inventory paths have multiple roles: {', '.join(duplicate_roles)}")
    with _manifest_lock(root):
        _assert_no_ambiguous_manifest_staging(root)
        manifest_path = root / MANIFEST_NAME
        if manifest_path.exists() or manifest_path.is_symlink():
            raise StorageObjectError(f"storage object manifest already exists: {manifest_path}")
        files = tuple(
            path
            for path in storage_file_paths(root)
            if path.name not in {MANIFEST_NAME, LOCK_NAME} or path.parent != root
        )
        relative_files = {path.relative_to(root).as_posix() for path in files}
        missing_declared = sorted((normalized_inputs | normalized_metadata | normalized_caches) - relative_files)
        if missing_declared:
            raise StorageObjectError(f"declared inventory paths are missing: {', '.join(missing_declared)}")
        resources = [
            {
                "digest": _sha256(path),
                "path": path.relative_to(root).as_posix(),
                "role": (
                    ResourceRole.INPUT.value
                    if path.relative_to(root).as_posix() in normalized_inputs
                    else (
                        ResourceRole.METADATA.value
                        if path.relative_to(root).as_posix() in normalized_metadata
                        else (
                            ResourceRole.CACHE.value
                            if parsed_kind is ObjectKind.TOOL_CACHE
                            or path.relative_to(root).as_posix() in normalized_caches
                            else ResourceRole.ARTIFACT.value
                        )
                    )
                ),
            }
            for path in files
        ]
        payload: dict[str, object] = {
            "content_schema": content_schema,
            "content_schema_version": content_schema_version,
            "demo": demo,
            "object_kind": parsed_kind.value,
            "owner_repository": owner_repository,
            "owner_tool": owner_tool,
            "producer_revision": producer_revision,
            "resources": resources,
            "retention_policy": parsed_retention.value,
            "schema": SCHEMA_ID,
            "storage_class": parsed_class.value,
            "storage_id": storage_id,
        }
        if original_execution_path is not None:
            payload["original_execution_path"] = original_execution_path
        return _write_manifest(
            manifest_path,
            payload,
            allow_untracked_demo_manifest=demo,
        )


def refresh_storage_object(
    storage_root: Path,
    *,
    expected_manifest_digest: str,
    producer_revision: str,
    cache_paths: tuple[str, ...] = (),
) -> dict[str, object]:
    """Refresh a changed object while preserving identity and protected roles."""

    requested_root = Path(storage_root).expanduser()
    if requested_root.is_symlink():
        raise StorageObjectError(f"storage object root must not be a symlink: {requested_root}")
    root = resolve_storage_path(requested_root, label="storage object root")
    if not root.is_dir():
        raise StorageObjectError(f"storage object root is not a directory: {root}")
    with _manifest_lock(root):
        _assert_no_ambiguous_manifest_staging(root)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise StorageObjectError(f"storage object root is missing a regular {MANIFEST_NAME}: {root}")
        observed_manifest_digest = _sha256(manifest_path)
        if observed_manifest_digest != expected_manifest_digest:
            raise StorageObjectError(
                "storage object manifest changed before refresh: "
                f"expected {expected_manifest_digest}, observed {observed_manifest_digest}"
            )
        try:
            previous_bytes = manifest_path.read_bytes()
        except OSError as exc:
            raise StorageObjectError(f"cannot read storage object manifest {manifest_path}: {exc}") from exc
        manifest = load_storage_object_manifest(manifest_path)
        if manifest.object_kind not in {ObjectKind.WORKSPACE, ObjectKind.STORE}:
            raise StorageObjectError(
                "storage receipt refresh is limited to active workspaces and stores; "
                f"found object_kind={manifest.object_kind.value}"
            )
        prior_resources = {resource.relative_path: resource for resource in manifest.resources}
        prior_roles = {path: resource.role for path, resource in prior_resources.items()}
        normalized_caches = {normalize_relative_path(path, label="cache path") for path in cache_paths}
        files = tuple(
            path
            for path in storage_file_paths(root)
            if path.name not in {MANIFEST_NAME, LOCK_NAME} or path.parent != root
        )
        relative_files = {path.relative_to(root).as_posix() for path in files}
        protected_paths = {
            path for path, role in prior_roles.items() if role in {ResourceRole.INPUT, ResourceRole.METADATA}
        }
        missing_protected = sorted(protected_paths - relative_files)
        if missing_protected:
            raise StorageObjectError(
                f"cannot refresh after removing input or metadata files: {', '.join(missing_protected)}"
            )
        changed_protected = sorted(
            path for path in protected_paths if _sha256(root / path) != prior_resources[path].digest
        )
        if changed_protected:
            raise StorageObjectError(
                f"cannot refresh after changing input or metadata files: {', '.join(changed_protected)}"
            )
        missing_caches = sorted(normalized_caches - relative_files)
        if missing_caches:
            raise StorageObjectError(f"declared refresh cache paths are missing: {', '.join(missing_caches)}")
        role_changes = sorted(
            path for path in normalized_caches if path in prior_roles and prior_roles[path] is not ResourceRole.CACHE
        )
        if role_changes:
            raise StorageObjectError(f"refresh cannot change existing resource roles: {', '.join(role_changes)}")
        resources = []
        for path in files:
            relative_path = path.relative_to(root).as_posix()
            role = prior_roles.get(relative_path)
            if role is None:
                role = ResourceRole.CACHE if relative_path in normalized_caches else ResourceRole.ARTIFACT
            resources.append(
                {
                    "digest": _sha256(path),
                    "path": relative_path,
                    "role": role.value,
                }
            )
        payload: dict[str, object] = {
            "content_schema": manifest.content_schema,
            "content_schema_version": manifest.content_schema_version,
            "demo": manifest.demo,
            "object_kind": manifest.object_kind.value,
            "owner_repository": manifest.owner_repository,
            "owner_tool": manifest.owner_tool,
            "producer_revision": producer_revision,
            "resources": resources,
            "retention_policy": manifest.retention_policy.value,
            "schema": manifest.schema,
            "storage_class": manifest.storage_class.value,
            "storage_id": manifest.storage_id,
        }
        if manifest.original_execution_path is not None:
            payload["original_execution_path"] = manifest.original_execution_path
        return _write_manifest(manifest_path, payload, previous_bytes=previous_bytes)
