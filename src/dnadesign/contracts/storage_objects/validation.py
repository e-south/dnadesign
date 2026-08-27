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
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .loading import load_storage_object_manifest_bytes
from .locking import acquire_existing_lock, release_lock
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
_GitIndexEntry = tuple[bytes, bytes, bytes]
_GitIndexSnapshot = tuple[bytes, dict[str, tuple[_GitIndexEntry, ...]]]
_DemoGitAuthority = tuple[Path, tuple[Path, ...], dict[str, str], bytes]
_StorageTreeEntryState = tuple[str, int, int, int, int, int, int, int, int, int]
_CoordinationState = tuple[
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, int],
]
_RoutedObject = tuple[Path, ObjectKind, str, tuple[_StorageTreeEntryState, ...]]


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


def _rebind_verified_storage_object(verified: VerifiedStorageObject) -> VerifiedStorageObject:
    """Reapply the full object contract before returning root-level evidence."""

    observed = verify_storage_object(verified.root)
    if observed.manifest_digest != verified.manifest_digest:
        raise StorageObjectError(
            "storage object manifest changed during root validation; retry while producers are quiescent"
        )
    expected_state = tuple(
        (
            resource.relative_path,
            resource.digest,
            resource.size_bytes,
            resource.device_id,
            resource.inode,
        )
        for resource in verified.resources
    )
    rebound_state = tuple(
        (
            resource.relative_path,
            resource.digest,
            resource.size_bytes,
            resource.device_id,
            resource.inode,
        )
        for resource in observed.resources
    )
    if rebound_state != expected_state:
        raise StorageObjectError(
            "storage object resources changed during root validation; retry while producers are quiescent"
        )
    return observed


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


def _storage_tree_state(
    root: Path,
    files: tuple[Path, ...],
    directories: tuple[Path, ...],
) -> tuple[_StorageTreeEntryState, ...]:
    """Fingerprint the identity, type, and ownership posture of one exact tree."""

    state: list[_StorageTreeEntryState] = []
    for path in (root, *directories, *files):
        relative = "." if path == root else path.relative_to(root).as_posix()
        try:
            entry_stat = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise StorageObjectError(f"cannot inspect storage entry: {relative}: {exc}") from exc
        is_coordination_lock = path.parent == root and path.name == LOCK_NAME
        state.append(
            (
                relative,
                entry_stat.st_dev,
                entry_stat.st_ino,
                stat.S_IFMT(entry_stat.st_mode),
                stat.S_IMODE(entry_stat.st_mode),
                entry_stat.st_uid,
                entry_stat.st_gid,
                entry_stat.st_size,
                0 if is_coordination_lock else entry_stat.st_mtime_ns,
                0 if is_coordination_lock else entry_stat.st_ctime_ns,
            )
        )
    return tuple(state)


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
    try:
        initial_stat = resolved.stat(follow_symlinks=False)
    except OSError as exc:
        raise StorageObjectError(f"cannot inspect storage resource {resource.relative_path}: {exc}") from exc
    if not stat.S_ISREG(initial_stat.st_mode):
        raise StorageObjectError(f"declared resource is not a file: {resource.relative_path}")
    observed_digest = _sha256(resolved)
    if observed_digest != resource.digest:
        raise StorageObjectError(
            f"declared resource digest mismatch for {resource.relative_path}: "
            f"expected {resource.digest}, observed {observed_digest}"
        )
    try:
        final_stat = resolved.stat(follow_symlinks=False)
    except OSError as exc:
        raise StorageObjectError(f"cannot inspect storage resource {resource.relative_path}: {exc}") from exc
    initial_identity = (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_size,
        initial_stat.st_mode,
    )
    final_identity = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mode,
    )
    if initial_identity != final_identity:
        raise StorageObjectError(
            f"declared resource changed during validation: {resource.relative_path}; "
            "retry while the producer is quiescent"
        )
    return VerifiedStoredResource(
        relative_path=resource.relative_path,
        path=resolved,
        digest=observed_digest,
        role=resource.role,
        size_bytes=final_stat.st_size,
        device_id=final_stat.st_dev,
        inode=final_stat.st_ino,
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
) -> _CoordinationState:
    """Validate and fingerprint root, manifest, and lock coordination state."""

    try:
        root_stat = root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise StorageObjectError(f"storage object root must remain a directory: {root}")
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
        if lock_mode & stat.S_IWOTH:
            raise StorageObjectError(f"storage object lock must not be other-writable: {lock_path}")
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
        (
            stat.S_IFMT(root_stat.st_mode),
            root_mode,
            root_stat.st_uid,
            root_stat.st_gid,
            root_stat.st_dev,
            root_stat.st_ino,
        ),
        (manifest_mode, manifest_stat.st_gid, manifest_stat.st_dev, manifest_stat.st_ino),
        lock_state,
    )


@contextmanager
def _validation_manifest_lock(root: Path) -> Iterator[None]:
    """Hold one existing writer lock while proving its pathname stays bound."""

    manifest_path = root / MANIFEST_NAME
    lock_path = root / LOCK_NAME
    root_state, _manifest_state, lock_state = _verify_coordination_posture(root, manifest_path, lock_path)
    root_identity = root_state[-2:]
    lock_mode, lock_gid, lock_size, lock_device, lock_inode = lock_state
    lock_identity = (lock_device, lock_inode)
    lock_descriptor = _acquire_existing_validation_lock(
        lock_path,
        expected_identity=lock_identity,
        expected_mode=lock_mode,
        expected_gid=lock_gid,
        expected_size=lock_size,
    )
    try:
        acquired_root = root.stat(follow_symlinks=False)
        acquired_named = lock_path.stat(follow_symlinks=False)
        acquired_held = os.fstat(lock_descriptor)
        if (acquired_root.st_dev, acquired_root.st_ino) != root_identity:
            raise StorageObjectError(f"storage object root changed before lock acquisition completed: {root}")
        if (acquired_named.st_dev, acquired_named.st_ino) != lock_identity or (
            acquired_held.st_dev,
            acquired_held.st_ino,
        ) != lock_identity:
            raise StorageObjectError(f"storage object lock changed before acquisition completed: {lock_path}")
        if (
            not stat.S_ISREG(acquired_named.st_mode)
            or not stat.S_ISREG(acquired_held.st_mode)
            or acquired_named.st_size != lock_size
            or acquired_held.st_size != lock_size
            or stat.S_IMODE(acquired_named.st_mode) != lock_mode
            or stat.S_IMODE(acquired_held.st_mode) != lock_mode
            or acquired_named.st_gid != lock_gid
            or acquired_held.st_gid != lock_gid
        ):
            raise StorageObjectError(f"storage object lock posture changed before acquisition completed: {lock_path}")
    except (OSError, StorageObjectError) as inspection_error:
        try:
            _release_validation_lock(lock_descriptor)
        except OSError as release_error:
            raise StorageObjectError(
                f"cannot inspect acquired storage object lock and release failed {lock_path}: {release_error}"
            ) from inspection_error
        if isinstance(inspection_error, StorageObjectError):
            raise
        raise StorageObjectError(f"cannot inspect acquired storage object lock {lock_path}: {inspection_error}") from (
            inspection_error
        )

    body_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        body_error = exc
    completion_error: StorageObjectError | None = None
    try:
        final_root = root.stat(follow_symlinks=False)
        before = lock_path.stat(follow_symlinks=False)
        held = os.fstat(lock_descriptor)
        after = lock_path.stat(follow_symlinks=False)
        if (final_root.st_dev, final_root.st_ino) != root_identity:
            raise StorageObjectError(f"storage object root changed while holding its manifest lock: {root}")
        if (
            (before.st_dev, before.st_ino) != lock_identity
            or (held.st_dev, held.st_ino) != lock_identity
            or (after.st_dev, after.st_ino) != lock_identity
        ):
            raise StorageObjectError(f"storage object lock changed before validation completion: {lock_path}")
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(held.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or before.st_size != lock_size
            or held.st_size != lock_size
            or after.st_size != lock_size
            or stat.S_IMODE(before.st_mode) != lock_mode
            or stat.S_IMODE(held.st_mode) != lock_mode
            or stat.S_IMODE(after.st_mode) != lock_mode
            or before.st_gid != lock_gid
            or held.st_gid != lock_gid
            or after.st_gid != lock_gid
        ):
            raise StorageObjectError(f"storage object lock posture changed before validation completion: {lock_path}")
    except OSError as exc:
        completion_error = StorageObjectError(f"cannot inspect storage object lock at completion {lock_path}: {exc}")
    except StorageObjectError as exc:
        completion_error = exc
    try:
        _release_validation_lock(lock_descriptor)
    except OSError as release_error:
        raise StorageObjectError(f"cannot release storage object validation lock {lock_path}: {release_error}") from (
            body_error or completion_error
        )
    if body_error is not None:
        raise body_error
    if completion_error is not None:
        raise completion_error


def _acquire_existing_validation_lock(
    lock_path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_mode: int,
    expected_gid: int,
    expected_size: int,
    timeout_seconds: float = 30.0,
) -> int:
    """Lock an existing no-follow descriptor without creating or truncating its path."""
    return acquire_existing_lock(
        lock_path,
        expected_identity=expected_identity,
        expected_mode=expected_mode,
        expected_gid=expected_gid,
        expected_size=expected_size,
        timeout_seconds=timeout_seconds,
    )


def _release_validation_lock(descriptor: int) -> None:
    """Release and close one validation-owned advisory-lock descriptor."""
    release_lock(descriptor)


def _git_authority_environment() -> dict[str, str]:
    """Remove caller-provided repository selection from demo Git reads."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--local-env-vars"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise StorageObjectError(
            f"cannot verify demo Git tracking because repository-local environment cannot be enumerated: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise StorageObjectError(
            "cannot verify demo Git tracking because repository-local environment cannot be enumerated: "
            f"{detail or 'git rev-parse failed'}"
        )
    environment = os.environ.copy()
    for variable in completed.stdout.splitlines():
        environment.pop(variable, None)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _read_demo_git_index_snapshot(
    checkout: Path,
    *,
    git_environment: dict[str, str],
) -> _GitIndexSnapshot:
    """Read one exact stage-entry snapshot from the repository's real index."""

    try:
        completed = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(checkout),
                "ls-files",
                "--stage",
                "-z",
            ],
            check=False,
            capture_output=True,
            env=git_environment,
        )
    except OSError as exc:
        raise StorageObjectError(f"cannot read demo Git index snapshot: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise StorageObjectError(f"cannot read demo Git index snapshot: {detail or 'git ls-files failed'}")
    parsed: dict[str, list[_GitIndexEntry]] = {}
    records = completed.stdout.removesuffix(b"\0").split(b"\0") if completed.stdout else []
    for record in records:
        header, separator, indexed_path = record.partition(b"\t")
        fields = header.split()
        if not separator or len(fields) != 3:
            raise StorageObjectError("cannot parse demo Git index snapshot")
        relative = indexed_path.decode(errors="surrogateescape")
        parsed.setdefault(relative, []).append((fields[0], fields[1], fields[2]))
    return completed.stdout, {relative: tuple(entries) for relative, entries in parsed.items()}


def _verify_demo_git_index_entry(
    checkout: Path,
    relative: str,
    entries: tuple[_GitIndexEntry, ...],
    expected_digest: str | None,
    *,
    git_environment: dict[str, str],
) -> None:
    """Bind one verified demo file snapshot to one captured stage-0 blob."""

    if len(entries) != 1 or entries[0][2] != b"0":
        raise StorageObjectError(f"demo Git index entry must have exactly one stage-0 record: {relative}")
    index_mode, object_id, _index_stage = entries[0]
    if index_mode not in {b"100644", b"100755"}:
        mode_label = index_mode.decode(errors="replace")
        raise StorageObjectError(
            f"demo Git index entry must be a regular file (mode 100644 or 100755); found mode {mode_label}: {relative}"
        )
    try:
        indexed = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(checkout), "cat-file", "blob", object_id],
            check=False,
            capture_output=True,
            env=git_environment,
        )
    except OSError as exc:
        raise StorageObjectError(f"cannot verify demo Git index bytes for {relative}: {exc}") from exc
    if indexed.returncode != 0:
        detail = indexed.stderr.decode(errors="replace").strip()
        raise StorageObjectError(
            f"cannot verify demo Git index bytes for {relative}: {detail or 'git cat-file failed'}"
        )
    if expected_digest is not None and _sha256_bytes(indexed.stdout) != expected_digest:
        raise StorageObjectError(f"demo file differs from Git index: {relative}")


def _assert_demo_git_index_stable(
    checkout: Path,
    expected_snapshot: bytes,
    *,
    git_environment: dict[str, str],
    validation_scope: str = "validation",
) -> None:
    """Reject authority assembled across more than one Git index state."""

    observed_snapshot, _entries = _read_demo_git_index_snapshot(
        checkout,
        git_environment=git_environment,
    )
    if observed_snapshot != expected_snapshot:
        raise StorageObjectError(
            f"demo Git index changed during {validation_scope}; retry while repository staging is quiescent"
        )


def verify_manifest_index_if_git_resident(root: Path, manifest_path: Path, expected_digest: str) -> bool:
    """Bind a Git-resident prior receipt to its index before refresh."""

    checkout = _git_checkout_ancestor(root, include_root=True)
    if checkout is None:
        return False
    git_environment = _git_authority_environment()
    snapshot, index_entries = _read_demo_git_index_snapshot(
        checkout,
        git_environment=git_environment,
    )
    relative = manifest_path.relative_to(checkout).as_posix()
    entries = index_entries.get(relative)
    if entries is None:
        raise StorageObjectError(f"demo file is not tracked: {relative}")
    _verify_demo_git_index_entry(
        checkout,
        relative,
        entries,
        expected_digest,
        git_environment=git_environment,
    )
    _assert_demo_git_index_stable(
        checkout,
        snapshot,
        git_environment=git_environment,
    )
    return True


def _verify_demo(
    checkout: Path,
    verified: VerifiedStorageObject,
    *,
    allow_pending_manifest: bool,
    allow_pending_lock: bool,
) -> tuple[dict[str, str], bytes]:
    git_environment = _git_authority_environment()
    snapshot, index_entries = _read_demo_git_index_snapshot(
        checkout,
        git_environment=git_environment,
    )
    try:
        manifest_size = verified.manifest_path.stat(follow_symlinks=False).st_size
    except OSError as exc:
        raise StorageObjectError(
            f"cannot inspect demo manifest after validation {verified.manifest_path}: {exc}"
        ) from exc
    total_bytes = manifest_size + sum(resource.size_bytes for resource in verified.resources)
    if total_bytes > MAX_DEMO_BYTES:
        raise StorageObjectError(f"demo exceeds {MAX_DEMO_BYTES} bytes: {total_bytes}")
    expected_files = (
        (verified.manifest_path, verified.manifest_digest),
        (verified.root / LOCK_NAME, _sha256_bytes(b"")),
        *((resource.path, resource.digest) for resource in verified.resources),
    )
    expected_by_relative = {
        path.relative_to(checkout).as_posix(): (path, expected_digest) for path, expected_digest in expected_files
    }
    root_relative = verified.root.relative_to(checkout).as_posix()
    indexed_root_paths = (
        set(index_entries)
        if root_relative == "."
        else {relative for relative in index_entries if relative.startswith(f"{root_relative}/")}
    )
    extra = sorted(indexed_root_paths - set(expected_by_relative))
    if extra:
        raise StorageObjectError(f"demo Git index has undeclared entries: {', '.join(extra)}")

    for relative, (path, expected_digest) in expected_by_relative.items():
        pending = (allow_pending_manifest and path == verified.manifest_path) or (
            allow_pending_lock and path == verified.root / LOCK_NAME
        )
        entries = index_entries.get(relative)
        if entries is None:
            if pending:
                continue
            raise StorageObjectError(f"demo file is not tracked: {relative}")
        _verify_demo_git_index_entry(
            checkout,
            relative,
            entries,
            None if pending else expected_digest,
            git_environment=git_environment,
        )
    return git_environment, snapshot


def _recheck_verified_demo_snapshot(
    verified: VerifiedStorageObject,
    expected_coordination_state: _CoordinationState,
    expected_tree_state: tuple[_StorageTreeEntryState, ...],
) -> None:
    """Rebind verified demo bytes and coordination after Git authority reads."""

    try:
        manifest_bytes = verified.manifest_path.read_bytes()
    except OSError as exc:
        raise StorageObjectError(
            f"cannot reread demo manifest after Git index validation {verified.manifest_path}: {exc}"
        ) from exc
    if _sha256_bytes(manifest_bytes) != verified.manifest_digest:
        raise StorageObjectError(
            "demo manifest changed during Git index validation; retry while the producer is quiescent"
        )
    resources = tuple(_verify_resource(verified.root, resource) for resource in verified.manifest.resources)
    expected_resources = tuple(
        (
            resource.relative_path,
            resource.digest,
            resource.size_bytes,
            resource.device_id,
            resource.inode,
        )
        for resource in verified.resources
    )
    observed_resources = tuple(
        (
            resource.relative_path,
            resource.digest,
            resource.size_bytes,
            resource.device_id,
            resource.inode,
        )
        for resource in resources
    )
    coordination_state = _verify_coordination_posture(
        verified.root,
        verified.manifest_path,
        verified.root / LOCK_NAME,
    )
    final_file_paths, final_directory_paths = _storage_tree_paths(verified.root)
    observed_tree_state = _storage_tree_state(verified.root, final_file_paths, final_directory_paths)
    if (
        observed_resources != expected_resources
        or coordination_state != expected_coordination_state
        or observed_tree_state != expected_tree_state
    ):
        raise StorageObjectError(
            "demo storage object changed during Git index validation; retry while the producer is quiescent"
        )


def verify_storage_object(
    storage_root: Path,
    *,
    _allow_pending_demo_manifest: bool = False,
    _allow_pending_demo_lock: bool = False,
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
    (_root_type, root_mode, _root_owner, shared_group, _root_device, _root_inode), _manifest_state, _lock_state = (
        coordination_state
    )
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
    second_tree_state = _storage_tree_state(root, second_file_paths, second_directory_paths)
    second_actual_paths = {
        path.relative_to(root).as_posix()
        for path in second_file_paths
        if path.name not in {MANIFEST_NAME, LOCK_NAME} or path.parent != root
    }
    first_state = tuple(
        (item.relative_path, item.digest, item.size_bytes, item.device_id, item.inode) for item in resources
    )
    second_state = tuple(
        (item.relative_path, item.digest, item.size_bytes, item.device_id, item.inode) for item in second_resources
    )
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
        git_environment, git_index_snapshot = _verify_demo(
            checkout,
            verified,
            allow_pending_manifest=_allow_pending_demo_manifest,
            allow_pending_lock=_allow_pending_demo_lock,
        )
        _recheck_verified_demo_snapshot(
            verified,
            second_coordination_state,
            second_tree_state,
        )
        _assert_demo_git_index_stable(
            checkout,
            git_index_snapshot,
            git_environment=git_environment,
        )
    elif checkout is not None:
        raise StorageObjectError(
            f"non-demo storage object cannot live inside a Git checkout: object={root}, checkout={checkout}"
        )
    return verified


def _routed_object_directories(
    root: Path,
) -> tuple[_RoutedObject, ...]:
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
    routes: list[_RoutedObject] = []
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
                object_files, object_directories = _storage_tree_paths(object_directory)
                routes.append(
                    (
                        object_directory,
                        expected_kind,
                        owner_directory.name,
                        _storage_tree_state(object_directory, object_files, object_directories),
                    )
                )
    return tuple(routes)


def _route_coordination_identity(routes: tuple[_RoutedObject, ...]) -> tuple[tuple[object, ...], ...]:
    """Compare discovered route ownership without lock-acquisition timestamp noise."""

    return tuple(
        (
            object_root,
            object_kind,
            owner_name,
            tree_state[0][:7],
            next((entry[:8] for entry in tree_state if entry[0] == LOCK_NAME), None),
        )
        for object_root, object_kind, owner_name, tree_state in routes
    )


def _demo_checkout_groups(
    objects: tuple[VerifiedStorageObject, ...] | list[VerifiedStorageObject],
) -> tuple[tuple[Path, tuple[Path, ...]], ...]:
    """Group demo roots by their independent Git index authority."""

    grouped: dict[Path, list[Path]] = {}
    for verified in objects:
        if not verified.manifest.demo:
            continue
        checkout = _git_checkout_ancestor(verified.root, include_root=True)
        if checkout is None:
            raise StorageObjectError(f"demo storage object must live inside a Git checkout: {verified.root}")
        grouped.setdefault(checkout, []).append(verified.root)
    return tuple(
        (checkout, tuple(sorted(grouped[checkout], key=lambda path: path.as_posix())))
        for checkout in sorted(grouped, key=lambda path: path.as_posix())
    )


def _capture_demo_git_authorities(
    objects: tuple[VerifiedStorageObject, ...] | list[VerifiedStorageObject],
) -> tuple[_DemoGitAuthority, ...]:
    """Capture each checkout index once for the full locked root-validation pass."""

    checkout_groups = _demo_checkout_groups(objects)
    if len(checkout_groups) > 1:
        raise StorageObjectError(
            "routed root demos must share one Git checkout because separate indexes "
            "cannot provide one coherent root authority snapshot"
        )
    authorities: list[_DemoGitAuthority] = []
    for checkout, member_roots in checkout_groups:
        git_environment = _git_authority_environment()
        snapshot, _entries = _read_demo_git_index_snapshot(
            checkout,
            git_environment=git_environment,
        )
        authorities.append((checkout, member_roots, git_environment, snapshot))
    return tuple(authorities)


def _assert_demo_git_authorities_stable(
    objects: tuple[VerifiedStorageObject, ...],
    authorities: tuple[_DemoGitAuthority, ...],
) -> None:
    """Recheck every independent checkout after all object rebinds finish."""

    expected_groups = tuple((checkout, member_roots) for checkout, member_roots, _environment, _snapshot in authorities)
    if _demo_checkout_groups(objects) != expected_groups:
        raise StorageObjectError(
            "demo Git checkout grouping changed during root validation; retry while repository routing is quiescent"
        )
    for checkout, _member_roots, git_environment, snapshot in authorities:
        _assert_demo_git_index_stable(
            checkout,
            snapshot,
            git_environment=git_environment,
            validation_scope="root validation",
        )


def _verify_locked_storage_root(root: Path, routes: tuple[_RoutedObject, ...]) -> VerifiedStorageRoot:
    """Verify all routes while every discovered object writer lock is held."""

    objects: list[VerifiedStorageObject] = []
    identities: set[tuple[str, str, str]] = set()
    for object_directory, expected_kind, owner_name, _directory_state in routes:
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
    demo_git_authorities = _capture_demo_git_authorities(objects)
    revalidated_objects: list[VerifiedStorageObject] = []
    for verified in objects:
        _recheck_verified_manifest(verified)
        revalidated = verify_storage_object(verified.root)
        if revalidated.manifest_digest != verified.manifest_digest:
            raise StorageObjectError(
                "storage object manifest changed during root validation; retry while producers are quiescent"
            )
        revalidated_objects.append(revalidated)
    if routes != _routed_object_directories(root):
        raise StorageObjectError("storage root changed during validation; retry while object routing is quiescent")
    final_objects = tuple(_rebind_verified_storage_object(verified) for verified in revalidated_objects)
    if routes != _routed_object_directories(root):
        raise StorageObjectError("storage root changed during validation; retry while object routing is quiescent")
    _assert_demo_git_authorities_stable(final_objects, demo_git_authorities)
    return VerifiedStorageRoot(root=root, objects=final_objects)


def verify_storage_root(storage_root: Path) -> VerifiedStorageRoot:
    """Verify routed storage shelves and every contained object."""

    requested_root = Path(storage_root).expanduser()
    if requested_root.is_symlink():
        raise StorageObjectError(f"storage root must not be a symlink: {requested_root}")
    root = resolve_storage_path(requested_root, label="storage root")
    if not root.is_dir():
        raise StorageObjectError(f"storage root is not a directory: {root}")
    discovered_routes = _routed_object_directories(root)
    ordered_roots = sorted((route[0] for route in discovered_routes), key=lambda path: path.as_posix())
    with ExitStack() as locks:
        for object_root in ordered_roots:
            locks.enter_context(_validation_manifest_lock(object_root))
        locked_routes = _routed_object_directories(root)
        if _route_coordination_identity(locked_routes) != _route_coordination_identity(discovered_routes):
            raise StorageObjectError(
                "storage root changed during validation while acquiring object locks; "
                "retry while object routing is quiescent"
            )
        return _verify_locked_storage_root(root, locked_routes)
