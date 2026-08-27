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
import stat
import subprocess
from pathlib import Path

from .loading import load_storage_object_manifest_bytes
from .models import (
    LOCK_NAME,
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


def resolve_storage_path(path: Path, *, label: str, strict: bool = False) -> Path:
    """Resolve one contract path while normalizing filesystem-loop failures."""

    try:
        return path.resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        raise StorageObjectError(f"{label} does not resolve: {path}: {exc}") from exc


def _directory_entries(directory: Path, *, label: str) -> tuple[Path, ...]:
    try:
        return tuple(directory.iterdir())
    except OSError as exc:
        raise StorageObjectError(f"cannot enumerate {label} {directory}: {exc}") from exc


def _git_checkout_ancestor(root: Path, *, include_root: bool) -> Path | None:
    candidates = (root, *root.parents) if include_root else root.parents
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StorageObjectError(f"cannot read storage resource {path}: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _recheck_verified_manifest(verified: VerifiedStorageObject) -> None:
    """Reject a root snapshot whose verified receipt has since changed."""

    manifest_path = verified.root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StorageObjectError(
            "storage object manifest changed during root validation; retry while producers are quiescent"
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise StorageObjectError(
            "storage object manifest changed during root validation; retry while producers are quiescent"
        ) from exc
    if _sha256_bytes(manifest_bytes) != verified.manifest_digest:
        raise StorageObjectError(
            "storage object manifest changed during root validation; retry while producers are quiescent"
        )


def _storage_tree_paths(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return regular files and directories while rejecting unsafe entries."""

    def _raise_walk_error(error: OSError) -> None:
        raise StorageObjectError(
            f"cannot traverse storage object: {error.filename or root}: {error.strerror or error}"
        ) from error

    files: list[Path] = []
    directories: list[Path] = []
    for current, directory_names, file_names in os.walk(
        root,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory.is_symlink():
                relative = directory.relative_to(root).as_posix()
                raise StorageObjectError(f"symlink is not allowed: {relative}")
            directories.append(directory)
        for file_name in file_names:
            path = current_path / file_name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise StorageObjectError(f"symlink is not allowed: {relative}")
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise StorageObjectError(f"cannot inspect storage entry: {relative}: {exc}") from exc
            if not stat.S_ISREG(mode):
                raise StorageObjectError(f"non-regular storage entry is not allowed: {relative}")
            files.append(path)
    return tuple(files), tuple(directories)


def storage_file_paths(root: Path) -> tuple[Path, ...]:
    """Return every regular file while rejecting symlinks and special entries."""

    files, _directories = _storage_tree_paths(root)
    return files


def _verify_resource(root: Path, resource: StoredResource) -> VerifiedStoredResource:
    source_path = root / resource.relative_path
    if source_path.is_symlink():
        raise StorageObjectError(f"symlink is not allowed: {resource.relative_path}")
    resolved = resolve_storage_path(
        source_path,
        label=f"declared resource {resource.relative_path}",
        strict=True,
    )
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
    try:
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        raise StorageObjectError(f"cannot inspect storage resource {resource.relative_path}: {exc}") from exc
    return VerifiedStoredResource(
        relative_path=resource.relative_path,
        path=resolved,
        digest=observed_digest,
        role=resource.role,
        size_bytes=size_bytes,
    )


def _verify_shared_resource_access(
    root: Path,
    resources: tuple[VerifiedStoredResource, ...],
    directories: tuple[Path, ...],
    *,
    shared_group: int,
) -> None:
    """Require shared-object readers to reach and hash every declared resource."""

    for resource in resources:
        try:
            resource_stat = resource.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise StorageObjectError(f"cannot inspect shared resource {resource.relative_path}: {exc}") from exc
        if resource_stat.st_gid != shared_group:
            raise StorageObjectError(
                f"shared resource does not inherit the storage object group: {resource.relative_path}"
            )
        if not resource_stat.st_mode & stat.S_IRGRP:
            raise StorageObjectError(f"shared resource must be group-readable: {resource.relative_path}")
    for directory in sorted(directories):
        relative = directory.relative_to(root).as_posix()
        try:
            directory_stat = directory.stat(follow_symlinks=False)
        except OSError as exc:
            raise StorageObjectError(f"cannot inspect shared resource directory {relative}: {exc}") from exc
        if directory_stat.st_gid != shared_group:
            raise StorageObjectError(f"shared resource directory does not inherit the storage object group: {relative}")
        required = stat.S_IRGRP | stat.S_IXGRP
        if directory_stat.st_mode & required != required:
            raise StorageObjectError(f"shared resource directory must be group-readable and traversable: {relative}")


def _verify_coordination_posture(
    root: Path,
    manifest_path: Path,
    lock_path: Path,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int, int]]:
    """Validate and fingerprint root, manifest, and lock coordination state."""

    try:
        root_stat = root.stat(follow_symlinks=False)
        root_mode = stat.S_IMODE(root_stat.st_mode)
        if root_mode & stat.S_IWOTH:
            raise StorageObjectError(
                "storage object roots must not be other-writable because unrelated accounts "
                "cannot share a trusted coordination boundary"
            )
        if root_mode & stat.S_IWGRP and not root_mode & stat.S_ISGID:
            raise StorageObjectError(
                "group-writable storage object roots must set the setgid bit "
                "so coordination files inherit the shared group"
            )
        if root_mode & stat.S_IWGRP and root_mode & stat.S_ISVTX:
            raise StorageObjectError(
                "group-writable storage object roots must not set the sticky bit "
                "because collaborators must be able to replace contract-owned receipts"
            )
        if root_mode & stat.S_IWGRP and not root_mode & stat.S_IXGRP:
            raise StorageObjectError(
                "group-writable storage object roots must be group-traversable "
                "so collaborators can reach coordination files"
            )
        if root_mode & stat.S_IWGRP and not root_mode & stat.S_IRGRP:
            raise StorageObjectError(
                "group-writable storage object roots must be group-readable "
                "so collaborators can enumerate declared content"
            )
        manifest_stat = manifest_path.stat(follow_symlinks=False)
        manifest_mode = stat.S_IMODE(manifest_stat.st_mode)
        if root_mode & stat.S_IWGRP and manifest_stat.st_gid != root_stat.st_gid:
            raise StorageObjectError(
                f"storage object manifest does not inherit the shared object group: {manifest_path}"
            )
        if root_mode & stat.S_IWGRP and not manifest_mode & stat.S_IRGRP:
            raise StorageObjectError(
                f"storage object manifest must be group-readable in a shared object root: {manifest_path}"
            )
        if lock_path.is_symlink():
            raise StorageObjectError(f"storage object lock must be a regular file: {lock_path}")
        try:
            lock_stat = lock_path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise StorageObjectError(f"storage object lock is missing: {lock_path}") from exc
        if not stat.S_ISREG(lock_stat.st_mode):
            raise StorageObjectError(f"storage object lock must be a regular file: {lock_path}")
        lock_mode = stat.S_IMODE(lock_stat.st_mode)
        if lock_stat.st_size != 0:
            raise StorageObjectError(f"storage object lock must be an empty coordination file: {lock_path}")
        owner_required = stat.S_IRUSR | stat.S_IWUSR
        if lock_mode & owner_required != owner_required:
            raise StorageObjectError(f"storage object lock must be owner-readable and owner-writable: {lock_path}")
        if root_mode & stat.S_IWGRP and lock_stat.st_gid != root_stat.st_gid:
            raise StorageObjectError(f"storage object lock does not inherit the shared object group: {lock_path}")
        if root_mode & stat.S_IWGRP and not lock_mode & stat.S_IWGRP:
            raise StorageObjectError(f"storage object lock must be group-writable in a shared object root: {lock_path}")
        if root_mode & stat.S_IWGRP and not lock_mode & stat.S_IRGRP:
            raise StorageObjectError(f"storage object lock must be group-readable in a shared object root: {lock_path}")
        lock_state = (
            lock_mode,
            lock_stat.st_gid,
            lock_stat.st_size,
            lock_stat.st_dev,
            lock_stat.st_ino,
        )
    except OSError as exc:
        raise StorageObjectError(f"cannot inspect storage object lock {lock_path}: {exc}") from exc
    return (
        (root_mode, root_stat.st_gid),
        (manifest_mode, manifest_stat.st_gid),
        lock_state,
    )


def _verify_demo_git_index_entry(checkout: Path, path: Path, expected_digest: str) -> None:
    """Bind one verified demo file snapshot to its stage-0 Git index blob."""

    relative = path.relative_to(checkout).as_posix()
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "ls-files",
                "--error-unmatch",
                "--",
                f":(literal){relative}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise StorageObjectError(f"cannot verify demo Git tracking for {relative}: {exc}") from exc
    if completed.returncode != 0:
        raise StorageObjectError(f"demo file is not tracked: {relative}")
    try:
        indexed = subprocess.run(
            ["git", "-C", str(checkout), "cat-file", "blob", f":{relative}"],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise StorageObjectError(f"cannot verify demo Git index bytes for {relative}: {exc}") from exc
    if indexed.returncode != 0:
        detail = indexed.stderr.decode(errors="replace").strip()
        raise StorageObjectError(
            f"cannot verify demo Git index bytes for {relative}: {detail or 'git cat-file failed'}"
        )
    if _sha256_bytes(indexed.stdout) != expected_digest:
        raise StorageObjectError(f"demo file differs from Git index: {relative}")


def verify_manifest_index_if_git_resident(root: Path, manifest_path: Path, expected_digest: str) -> bool:
    """Bind a Git-resident prior receipt to its index before refresh."""

    checkout = _git_checkout_ancestor(root, include_root=True)
    if checkout is None:
        return False
    _verify_demo_git_index_entry(checkout, manifest_path, expected_digest)
    return True


def _verify_demo(
    checkout: Path,
    verified: VerifiedStorageObject,
    *,
    allow_pending_manifest: bool,
) -> None:
    total_bytes = verified.manifest_path.stat().st_size + sum(resource.size_bytes for resource in verified.resources)
    if total_bytes > MAX_DEMO_BYTES:
        raise StorageObjectError(f"demo exceeds {MAX_DEMO_BYTES} bytes: {total_bytes}")
    for path, expected_digest in (
        (verified.manifest_path, verified.manifest_digest),
        *((resource.path, resource.digest) for resource in verified.resources),
    ):
        if allow_pending_manifest and path == verified.manifest_path:
            continue
        _verify_demo_git_index_entry(checkout, path, expected_digest)


def verify_storage_object(
    storage_root: Path,
    *,
    _allow_pending_demo_manifest: bool = False,
) -> VerifiedStorageObject:
    """Verify one explicit storage object and require exact file closure."""

    requested_root = Path(storage_root).expanduser()
    if requested_root.is_symlink():
        raise StorageObjectError(f"storage object root must not be a symlink: {requested_root}")
    root = resolve_storage_path(requested_root, label="storage object root")
    if not root.is_dir():
        raise StorageObjectError(f"storage object root is not a directory: {root}")
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise StorageObjectError(f"storage object manifest must not be a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise StorageObjectError(f"storage object root is missing {MANIFEST_NAME}: {root}")
    lock_path = root / LOCK_NAME
    coordination_state = _verify_coordination_posture(root, manifest_path, lock_path)
    (root_mode, shared_group), _manifest_state, _lock_state = coordination_state
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise StorageObjectError(f"cannot read storage object manifest {manifest_path}: {exc}") from exc
    manifest = load_storage_object_manifest_bytes(manifest_bytes, source_label=str(manifest_path))

    declared_paths: set[str] = set()
    for resource in manifest.resources:
        if resource.relative_path == MANIFEST_NAME:
            raise StorageObjectError(f"manifest cannot declare itself: {MANIFEST_NAME}")
        if resource.relative_path in declared_paths:
            raise StorageObjectError(f"resource path is declared more than once: {resource.relative_path}")
        declared_paths.add(resource.relative_path)

    resources = tuple(_verify_resource(root, resource) for resource in manifest.resources)
    first_file_paths, first_directory_paths = _storage_tree_paths(root)
    if root_mode & stat.S_IWGRP:
        _verify_shared_resource_access(
            root,
            resources,
            first_directory_paths,
            shared_group=shared_group,
        )
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in first_file_paths
        if path.name not in {MANIFEST_NAME, LOCK_NAME} or path.parent != root
    }
    undeclared = sorted(actual_paths - declared_paths)
    if undeclared:
        raise StorageObjectError(f"undeclared files: {', '.join(undeclared)}")
    missing = sorted(declared_paths - actual_paths)
    if missing:
        raise StorageObjectError(f"declared files are missing: {', '.join(missing)}")
    second_resources = tuple(_verify_resource(root, resource) for resource in manifest.resources)
    second_file_paths, second_directory_paths = _storage_tree_paths(root)
    second_actual_paths = {
        path.relative_to(root).as_posix()
        for path in second_file_paths
        if path.name not in {MANIFEST_NAME, LOCK_NAME} or path.parent != root
    }
    first_state = tuple((item.relative_path, item.digest, item.size_bytes) for item in resources)
    second_state = tuple((item.relative_path, item.digest, item.size_bytes) for item in second_resources)
    try:
        second_manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise StorageObjectError(f"cannot reread storage object manifest {manifest_path}: {exc}") from exc
    second_coordination_state = _verify_coordination_posture(root, manifest_path, lock_path)
    if root_mode & stat.S_IWGRP:
        _verify_shared_resource_access(
            root,
            second_resources,
            second_directory_paths,
            shared_group=shared_group,
        )
    first_directories = tuple(path.relative_to(root).as_posix() for path in first_directory_paths)
    second_directories = tuple(path.relative_to(root).as_posix() for path in second_directory_paths)
    if (
        manifest_bytes != second_manifest_bytes
        or coordination_state != second_coordination_state
        or first_state != second_state
        or actual_paths != second_actual_paths
        or first_directories != second_directories
    ):
        raise StorageObjectError("storage object changed during validation; retry while the producer is quiescent")

    verified = VerifiedStorageObject(
        root=root,
        manifest_path=resolve_storage_path(manifest_path, label="storage object manifest", strict=True),
        manifest_digest=_sha256_bytes(second_manifest_bytes),
        manifest=manifest,
        resources=second_resources,
    )
    checkout = _git_checkout_ancestor(
        root,
        include_root=(manifest.demo or manifest.object_kind is not ObjectKind.TOOL_CACHE),
    )
    if manifest.demo:
        if checkout is None:
            raise StorageObjectError(f"demo storage object must live inside a Git checkout: {root}")
        _verify_demo(
            checkout,
            verified,
            allow_pending_manifest=_allow_pending_demo_manifest,
        )
    elif checkout is not None:
        raise StorageObjectError(
            f"non-demo storage object cannot live inside a Git checkout: object={root}, checkout={checkout}"
        )
    return verified


def _routed_object_directories(root: Path) -> tuple[tuple[Path, ObjectKind, str], ...]:
    """Enumerate one exact routed-root snapshot without verifying object bytes."""

    allowed_shelves = set(_SHELF_KINDS) | _ALLOWED_ROOT_FILES
    root_entries = _directory_entries(root, label="storage root")
    unexpected_root_paths = sorted(path.name for path in root_entries if path.name not in allowed_shelves)
    if unexpected_root_paths:
        raise StorageObjectError(f"unexpected path in storage root: {', '.join(unexpected_root_paths)}")
    routing_file = root / "AGENTS.md"
    if routing_file.is_symlink():
        raise StorageObjectError(f"storage root routing file must not be a symlink: {routing_file}")
    if routing_file.exists() and not routing_file.is_file():
        raise StorageObjectError(f"storage root routing file must be a regular file: {routing_file}")
    routes: list[tuple[Path, ObjectKind, str]] = []
    for shelf_name, expected_kind in _SHELF_KINDS.items():
        shelf = root / shelf_name
        if shelf.is_symlink():
            raise StorageObjectError(f"storage shelf must not be a symlink: {shelf}")
        if not shelf.is_dir():
            raise StorageObjectError(f"storage root is missing shelf {shelf_name!r}")
        shelf_entries = _directory_entries(shelf, label="storage shelf")
        unexpected_shelf_paths = sorted(path.name for path in shelf_entries if not path.is_dir())
        if unexpected_shelf_paths:
            raise StorageObjectError(
                f"unexpected path in storage shelf {shelf_name!r}: {', '.join(unexpected_shelf_paths)}"
            )
        for owner_directory in sorted(path for path in shelf_entries if path.is_dir()):
            if owner_directory.is_symlink():
                raise StorageObjectError(f"storage shelf owner must not be a symlink: {owner_directory}")
            owner_entries = _directory_entries(owner_directory, label="storage owner directory")
            unexpected_owner_paths = sorted(path.name for path in owner_entries if not path.is_dir())
            if unexpected_owner_paths:
                raise StorageObjectError(
                    f"unexpected path in storage owner directory {owner_directory.name!r}: "
                    f"{', '.join(unexpected_owner_paths)}"
                )
            for object_directory in sorted(path for path in owner_entries if path.is_dir()):
                if object_directory.is_symlink():
                    raise StorageObjectError(f"storage object directory must not be a symlink: {object_directory}")
                routes.append((object_directory, expected_kind, owner_directory.name))
    return tuple(routes)


def verify_storage_root(storage_root: Path) -> VerifiedStorageRoot:
    """Verify routed storage shelves and every contained object."""

    requested_root = Path(storage_root).expanduser()
    if requested_root.is_symlink():
        raise StorageObjectError(f"storage root must not be a symlink: {requested_root}")
    root = resolve_storage_path(requested_root, label="storage root")
    if not root.is_dir():
        raise StorageObjectError(f"storage root is not a directory: {root}")
    routes = _routed_object_directories(root)
    objects: list[VerifiedStorageObject] = []
    identities: set[tuple[str, str, str]] = set()
    for object_directory, expected_kind, owner_name in routes:
        verified = verify_storage_object(object_directory)
        manifest = verified.manifest
        if manifest.object_kind is not expected_kind:
            shelf_name = object_directory.relative_to(root).parts[0]
            raise StorageObjectError(f"object_kind {manifest.object_kind.value!r} does not match shelf {shelf_name!r}")
        if manifest.owner_tool != owner_name:
            raise StorageObjectError(f"owner_tool {manifest.owner_tool!r} does not match directory {owner_name!r}")
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
    for verified in objects:
        _recheck_verified_manifest(verified)
    if routes != _routed_object_directories(root):
        raise StorageObjectError("storage root changed during validation; retry while object routing is quiescent")
    return VerifiedStorageRoot(root=root, objects=tuple(objects))
