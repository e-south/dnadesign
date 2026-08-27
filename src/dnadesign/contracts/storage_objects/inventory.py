"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/storage_objects/inventory.py

Deterministic manifest generation for one pre-existing storage object.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import shlex
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

from filelock import FileLock, Timeout

from .loading import load_storage_object_manifest_bytes, normalize_relative_path
from .models import (
    LOCK_NAME,
    MANIFEST_NAME,
    SCHEMA_ID,
    ObjectKind,
    ResourceRole,
    RetentionPolicy,
    StorageClass,
    StorageObjectError,
    StorageObjectPublicationUncertain,
    StorageObjectPublicationUnsupported,
)
from .validation import (
    resolve_storage_path,
    storage_file_paths,
    verify_manifest_index_if_git_resident,
    verify_storage_object,
)

_LINUX_RENAME_NOREPLACE = 0x00000001
_RENAME_EXCHANGE = 0x00000002
_DARWIN_RENAME_EXCL = 0x00000004


def _require_posix_publication_capabilities() -> None:
    """Fail before mutation when safe manifest publication is unavailable."""

    unavailable: list[str] = []
    required_functions = (
        "geteuid",
        "fchmod",
        "fstat",
        "fsync",
        "link",
        "open",
        "stat",
        "unlink",
    )
    for name in required_functions:
        if not callable(getattr(os, name, None)):
            unavailable.append(name)
    for name in ("O_DIRECTORY", "O_NOFOLLOW"):
        if not hasattr(os, name):
            unavailable.append(name)

    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    for name in ("stat", "unlink"):
        if not any(getattr(function, "__name__", None) == name for function in supports_dir_fd):
            unavailable.append(f"{name}_dir_fd")
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", ())
    for name in ("link", "stat"):
        if not any(getattr(function, "__name__", None) == name for function in supports_follow_symlinks):
            unavailable.append(f"{name}_follow_symlinks")

    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except (OSError, TypeError):
        libc = None
    native_rename_available = bool(
        libc is not None
        and (
            (sys.platform == "darwin" and hasattr(libc, "renameatx_np"))
            or (sys.platform.startswith("linux") and hasattr(libc, "renameat2"))
        )
    )
    if not native_rename_available:
        unavailable.append("atomic rename (renameatx_np or renameat2)")

    if unavailable:
        raise StorageObjectPublicationUnsupported(
            "storage manifest publication requires POSIX ownership, directory-descriptor, "
            "symlink-safe, and atomic rename capabilities before mutation; unavailable: "
            + ", ".join(unavailable)
            + "; use a supported macOS or Linux filesystem"
        )


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


def _rollback_manifest(
    manifest_path: Path,
    *,
    published_bytes: bytes,
    previous_bytes: bytes | None,
    previous_mode: int,
    operation_error: BaseException,
    published_identity: tuple[int, int] | None = None,
) -> None:
    """Undo only the manifest snapshot published by the failed operation."""

    restore_path: Path | None = None
    restore_identity: tuple[int, int] | None = None
    try:
        if manifest_path.is_symlink():
            raise StorageObjectError(f"cannot roll back a symlinked storage object manifest: {manifest_path}")
        if not manifest_path.exists():
            if previous_bytes is None:
                return
            raise StorageObjectError("storage object manifest disappeared before the prior receipt could be restored")
        current_bytes = manifest_path.read_bytes()
        if previous_bytes is not None and current_bytes == previous_bytes:
            return
        if current_bytes != published_bytes:
            raise StorageObjectError(
                "storage object manifest changed after publication; refusing to overwrite unrelated receipt bytes"
            )
        if previous_bytes is None:
            if published_identity is None:
                raise StorageObjectPublicationUncertain(
                    "cannot identify the receipt created by the failed operation; rollback is unsafe"
                )
            _rollback_create_only_manifest(
                manifest_path,
                published_bytes=published_bytes,
                published_identity=published_identity,
                operation_error=operation_error,
            )
            return
        if published_identity is None:
            raise StorageObjectPublicationUncertain(
                "cannot identify the refresh receipt published by the failed operation; rollback is unsafe"
            )
        descriptor, restore_name = tempfile.mkstemp(
            dir=manifest_path.parent,
            prefix=f".{MANIFEST_NAME}.restore-",
        )
        restore_path = Path(restore_name)
        restore_identity = _entry_identity(restore_path)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(previous_bytes)
            handle.flush()
            os.fchmod(handle.fileno(), previous_mode)
            os.fsync(handle.fileno())
        try:
            _publish_refresh_manifest(
                restore_path,
                manifest_path,
                previous_bytes=published_bytes,
                expected_previous_identity=published_identity,
                expected_staged_identity=restore_identity,
                expected_staged_bytes=previous_bytes,
            )
        except StorageObjectPublicationUncertain:
            restore_path = None
            raise
        except BaseException as restore_error:
            if restore_identity is not None:
                try:
                    _unlink_owned_entry(
                        restore_path,
                        expected_identity=restore_identity,
                        context="refresh restore staging entry",
                        missing_ok=True,
                    )
                except StorageObjectPublicationUncertain:
                    restore_path = None
                    raise
                except BaseException as cleanup_error:
                    retained_restore_path = restore_path
                    restore_path = None
                    raise StorageObjectPublicationUncertain(
                        "refresh rollback failed and recovery staging cleanup is uncertain; "
                        f"inspect {manifest_path} and {retained_restore_path}"
                    ) from cleanup_error
            restore_path = None
            if isinstance(restore_error, StorageObjectPublicationUnsupported):
                raise
            raise StorageObjectError(
                f"storage object operation failed and conditional manifest rollback failed: {restore_error}"
            ) from operation_error
        restore_path = None
    except OSError as restore_error:
        if restore_path is not None and restore_identity is not None:
            _unlink_owned_entry(
                restore_path,
                expected_identity=restore_identity,
                context="refresh restore staging entry",
                missing_ok=True,
            )
        raise StorageObjectError(
            f"storage object operation failed and manifest rollback failed: {restore_error}"
        ) from operation_error


def _entry_identity(path: Path) -> tuple[int, int]:
    entry_stat = path.lstat()
    return entry_stat.st_dev, entry_stat.st_ino


def _open_owner_cleanup_directory(parent: Path) -> tuple[Path, int]:
    """Open this OS owner's persistent private cleanup namespace."""

    cleanup_directory = parent / f".{MANIFEST_NAME}.cleanup-owner-{os.geteuid()}"
    try:
        cleanup_directory.mkdir(mode=0o750)
    except FileExistsError:
        pass
    except OSError as creation_error:
        raise StorageObjectPublicationUncertain(
            f"cannot create owner-private storage cleanup directory: {cleanup_directory}"
        ) from creation_error
    descriptor: int | None = None
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(cleanup_directory, flags)
        opened = os.fstat(descriptor)
        named = cleanup_directory.lstat()
        parent_stat = parent.stat(follow_symlinks=False)
    except OSError as inspection_error:
        if descriptor is not None:
            os.close(descriptor)
        raise StorageObjectPublicationUncertain(
            f"cannot open owner-private storage cleanup directory: {cleanup_directory}"
        ) from inspection_error
    mode = stat.S_IMODE(opened.st_mode)
    structurally_trusted = (
        stat.S_ISDIR(opened.st_mode)
        and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
        and opened.st_uid == os.geteuid()
        and mode & 0o700 == 0o700
        and mode & 0o022 == 0
    )
    shared_root = bool(stat.S_IMODE(parent_stat.st_mode) & stat.S_IWGRP)
    if not structurally_trusted or (shared_root and opened.st_gid != parent_stat.st_gid):
        os.close(descriptor)
        raise StorageObjectPublicationUncertain(
            f"storage cleanup directory is not an owner-write-private boundary: {cleanup_directory}"
        )
    expected_mode = 0o750 if shared_root else 0o700
    try:
        os.fchmod(descriptor, expected_mode)
        opened = os.fstat(descriptor)
        named = cleanup_directory.lstat()
    except OSError as posture_error:
        os.close(descriptor)
        raise StorageObjectPublicationUncertain(
            f"cannot set storage cleanup directory posture to {expected_mode:04o}: {cleanup_directory}"
        ) from posture_error
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino) or stat.S_IMODE(
        opened.st_mode
    ) & 0o777 != expected_mode:
        os.close(descriptor)
        raise StorageObjectPublicationUncertain(
            f"storage cleanup directory posture changed before use: {cleanup_directory}"
        )
    return cleanup_directory, descriptor


def _preflight_owner_cleanup_directory(parent: Path) -> None:
    _cleanup_directory, descriptor = _open_owner_cleanup_directory(parent)
    os.close(descriptor)


def _directory_entry_identity(directory_descriptor: int, name: str) -> tuple[int, int]:
    entry_stat = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    return entry_stat.st_dev, entry_stat.st_ino


def _directory_entry_present(directory_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _atomic_move_no_replace_into_directory(source: Path, destination_directory: int, destination_name: str) -> None:
    source_directory = os.open(source.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        _atomic_rename_at(
            source_directory,
            source.name,
            destination_directory,
            destination_name,
            darwin_flags=_DARWIN_RENAME_EXCL,
            linux_flags=_LINUX_RENAME_NOREPLACE,
            operation="no-replace cleanup move",
        )
    finally:
        os.close(source_directory)


def _atomic_move_no_replace_from_directory(source_directory: int, source_name: str, destination: Path) -> None:
    destination_directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        _atomic_rename_at(
            source_directory,
            source_name,
            destination_directory,
            destination.name,
            darwin_flags=_DARWIN_RENAME_EXCL,
            linux_flags=_LINUX_RENAME_NOREPLACE,
            operation="no-replace cleanup restore",
        )
    finally:
        os.close(destination_directory)


def _unlink_owned_entry(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    context: str,
    missing_ok: bool = False,
) -> None:
    """Delete only after atomically displacing an entry into an owner-private directory."""

    cleanup_directory, cleanup_descriptor = _open_owner_cleanup_directory(path.parent)
    try:
        candidate_name: str | None = None
        move_error: BaseException | None = None
        for _attempt in range(16):
            candidate = f"entry-{secrets.token_hex(16)}"
            try:
                _atomic_move_no_replace_into_directory(path, cleanup_descriptor, candidate)
            except BaseException as exc:
                move_error = exc
                source_present = path.exists() or path.is_symlink()
                candidate_present = _directory_entry_present(cleanup_descriptor, candidate)
                if not source_present and candidate_present:
                    candidate_name = candidate
                    break
                if isinstance(exc, FileExistsError) and source_present:
                    continue
                if isinstance(exc, FileNotFoundError) and not source_present and not candidate_present:
                    if missing_ok:
                        return
                    raise StorageObjectPublicationUncertain(
                        f"{context} disappeared before cleanup; outcome is uncertain"
                    )
                if isinstance(exc, StorageObjectPublicationUnsupported):
                    raise
                raise StorageObjectPublicationUncertain(
                    f"cannot atomically quarantine {context} before cleanup; outcome is uncertain"
                ) from exc
            else:
                candidate_name = candidate
                break
        if candidate_name is None:
            raise StorageObjectPublicationUncertain(
                f"cannot reserve a collision-free quarantine for {context}; outcome is uncertain"
            ) from move_error

        try:
            observed_identity = _directory_entry_identity(cleanup_descriptor, candidate_name)
        except OSError as inspection_error:
            raise StorageObjectPublicationUncertain(
                f"cannot identify quarantined {context}; retained in {cleanup_directory}"
            ) from inspection_error
        if observed_identity != expected_identity:
            try:
                _atomic_move_no_replace_from_directory(cleanup_descriptor, candidate_name, path)
            except BaseException:
                restored = _entry_has_identity(path, observed_identity) and not _directory_entry_present(
                    cleanup_descriptor, candidate_name
                )
            else:
                restored = _entry_has_identity(path, observed_identity)
            disposition = (
                "restored to its original path" if restored else f"retained at {cleanup_directory / candidate_name}"
            )
            raise StorageObjectPublicationUncertain(
                f"{context} changed at the cleanup boundary; foreign entry {disposition}"
            )
        try:
            os.unlink(candidate_name, dir_fd=cleanup_descriptor)
        except OSError as cleanup_error:
            raise StorageObjectPublicationUncertain(
                f"cannot remove quarantined owned {context}; retained in {cleanup_directory}"
            ) from cleanup_error
    finally:
        os.close(cleanup_descriptor)


def _entry_matches_regular_bytes(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_bytes: bytes,
) -> bool:
    """Match one pathname to an exact regular-file inode and byte snapshot."""

    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != expected_identity:
            return False
        content = path.read_bytes()
        after = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(after.st_mode) and (after.st_dev, after.st_ino) == expected_identity and content == expected_bytes
    )


def _entry_has_identity(path: Path, expected_identity: tuple[int, int]) -> bool:
    try:
        return _entry_identity(path) == expected_identity
    except OSError:
        return False


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_exchange(source: Path, destination: Path) -> None:
    """Atomically exchange two same-directory entries on supported POSIX kernels."""

    _atomic_rename(
        source,
        destination,
        darwin_flags=_RENAME_EXCHANGE,
        linux_flags=_RENAME_EXCHANGE,
        operation="exchange",
    )


def _atomic_move_no_replace(source: Path, destination: Path) -> None:
    """Atomically move one same-directory entry only when the destination is absent."""

    _atomic_rename(
        source,
        destination,
        darwin_flags=_DARWIN_RENAME_EXCL,
        linux_flags=_LINUX_RENAME_NOREPLACE,
        operation="no-replace move",
    )


def _atomic_rename(
    source: Path,
    destination: Path,
    *,
    darwin_flags: int,
    linux_flags: int,
    operation: str,
) -> None:
    """Invoke one supported same-directory native rename transaction."""

    if source.parent != destination.parent:
        raise StorageObjectError(f"atomic manifest {operation} requires same-directory entries")
    parent_descriptor = os.open(source.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        _atomic_rename_at(
            parent_descriptor,
            source.name,
            parent_descriptor,
            destination.name,
            darwin_flags=darwin_flags,
            linux_flags=linux_flags,
            operation=operation,
        )
    finally:
        os.close(parent_descriptor)


def _atomic_rename_at(
    source_directory: int,
    source_name: str,
    destination_directory: int,
    destination_name: str,
    *,
    darwin_flags: int,
    linux_flags: int,
    operation: str,
) -> None:
    """Invoke one supported native rename transaction between open directories."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        flags = darwin_flags
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        flags = linux_flags
    else:
        raise StorageObjectPublicationUnsupported(f"this platform does not support atomic storage manifest {operation}")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    result = rename(
        source_directory,
        os.fsencode(source_name),
        destination_directory,
        os.fsencode(destination_name),
        flags,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL}:
        raise StorageObjectPublicationUnsupported(
            f"this filesystem does not support atomic storage manifest {operation}"
        )
    raise OSError(error, os.strerror(error), destination_name)


def _rollback_create_only_manifest(
    manifest_path: Path,
    *,
    published_bytes: bytes,
    published_identity: tuple[int, int],
    operation_error: BaseException,
) -> None:
    """Remove only the create-only receipt owned by the failed operation."""

    descriptor, quarantine_name = tempfile.mkstemp(
        dir=manifest_path.parent,
        prefix=f".{MANIFEST_NAME}.rollback-",
    )
    os.close(descriptor)
    quarantine = Path(quarantine_name)
    quarantine_placeholder_identity = _entry_identity(quarantine)
    _unlink_owned_entry(
        quarantine,
        expected_identity=quarantine_placeholder_identity,
        context="create-only rollback quarantine placeholder",
    )
    try:
        _atomic_move_no_replace(manifest_path, quarantine)
    except FileNotFoundError:
        return
    except BaseException:
        try:
            _entry_identity(manifest_path)
        except FileNotFoundError:
            try:
                owns_quarantine = (
                    _entry_identity(quarantine) == published_identity and quarantine.read_bytes() == published_bytes
                )
            except BaseException:
                owns_quarantine = False
        else:
            owns_quarantine = False
        if owns_quarantine:
            _unlink_owned_entry(
                quarantine,
                expected_identity=published_identity,
                context="create-only rollback quarantine entry",
            )
            _fsync_directory(manifest_path.parent)
        raise

    try:
        quarantine_stat = quarantine.lstat()
        if (quarantine_stat.st_dev, quarantine_stat.st_ino) != published_identity:
            raise StorageObjectPublicationUncertain(
                f"cannot identify the receipt moved during create-only rollback; retained at {quarantine}"
            )
        owns_receipt = stat.S_ISREG(quarantine_stat.st_mode) and quarantine.read_bytes() == published_bytes
    except StorageObjectPublicationUncertain:
        raise
    except BaseException as inspection_error:
        raise StorageObjectPublicationUncertain(
            f"cannot identify the receipt moved during create-only rollback; retained at {quarantine}"
        ) from inspection_error
    if owns_receipt:
        try:
            _unlink_owned_entry(
                quarantine,
                expected_identity=published_identity,
                context="create-only rollback quarantine entry",
            )
            _fsync_directory(manifest_path.parent)
        except BaseException as cleanup_error:
            raise StorageObjectPublicationUncertain(
                f"create-only rollback cleanup is uncertain; inspect {quarantine} and {manifest_path}"
            ) from cleanup_error
        return

    try:
        _atomic_move_no_replace(quarantine, manifest_path)
        _fsync_directory(manifest_path.parent)
    except BaseException as restore_error:
        raise StorageObjectPublicationUncertain(
            f"create-only rollback moved an unrelated receipt; inspect {quarantine} and {manifest_path}"
        ) from restore_error
    raise StorageObjectError(
        "storage object manifest changed after publication; refusing to remove unrelated receipt bytes"
    ) from operation_error


def _preflight_create_only_rollback(directory: Path) -> None:
    """Prove no-replace rollback support before publishing a new receipt."""

    descriptor, source_name = tempfile.mkstemp(
        dir=directory,
        prefix=f".{MANIFEST_NAME}.tmp-preflight-",
    )
    os.close(descriptor)
    source = Path(source_name)
    destination = source.with_name(f"{source.name}.destination")
    source_identity = _entry_identity(source)
    try:
        _atomic_move_no_replace(source, destination)
        if source.exists() or _entry_identity(destination) != source_identity:
            raise StorageObjectPublicationUncertain(
                "cannot prove atomic create-only manifest rollback support on this filesystem"
            )
    finally:
        cleanup_errors: list[BaseException] = []
        for candidate in (source, destination):
            try:
                _unlink_owned_entry(
                    candidate,
                    expected_identity=source_identity,
                    context="create-only rollback preflight staging entry",
                    missing_ok=True,
                )
            except FileNotFoundError:
                continue
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            raise StorageObjectPublicationUncertain(
                "create-only rollback preflight cleanup is uncertain; inspect manifest staging state"
            ) from cleanup_errors[0]


def _publish_create_only_manifest(
    temporary: Path,
    manifest_path: Path,
    *,
    manifest_bytes: bytes,
    manifest_mode: int,
    expected_staged_identity: tuple[int, int],
) -> tuple[int, int]:
    """Publish one staged regular file without any replacement window."""

    published_identity = expected_staged_identity
    _preflight_create_only_rollback(manifest_path.parent)
    if not _entry_matches_regular_bytes(
        temporary,
        expected_identity=published_identity,
        expected_bytes=manifest_bytes,
    ):
        raise StorageObjectPublicationUncertain(
            "manifest staging entry changed before create-only publication; retained for explicit recovery"
        )
    try:
        os.link(temporary, manifest_path, follow_symlinks=False)
    except FileExistsError as exc:
        raise StorageObjectError(
            "storage object manifest appeared before publication; refusing to overwrite it"
        ) from exc
    except NotImplementedError as publication_error:
        _rollback_manifest(
            manifest_path,
            published_bytes=manifest_bytes,
            previous_bytes=None,
            previous_mode=manifest_mode,
            operation_error=publication_error,
            published_identity=published_identity,
        )
        raise StorageObjectPublicationUnsupported(
            "this platform does not support atomic create-only manifest publication"
        ) from publication_error
    except OSError as publication_error:
        _rollback_manifest(
            manifest_path,
            published_bytes=manifest_bytes,
            previous_bytes=None,
            previous_mode=manifest_mode,
            operation_error=publication_error,
            published_identity=published_identity,
        )
        if publication_error.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EPERM}:
            raise StorageObjectPublicationUnsupported(
                "this filesystem does not support atomic create-only manifest publication"
            ) from publication_error
        raise
    except BaseException as publication_error:
        _rollback_manifest(
            manifest_path,
            published_bytes=manifest_bytes,
            previous_bytes=None,
            previous_mode=manifest_mode,
            operation_error=publication_error,
            published_identity=published_identity,
        )
        raise
    try:
        if _entry_identity(manifest_path) != published_identity:
            raise StorageObjectPublicationUncertain(
                "storage object manifest changed after create-only publication; outcome is uncertain"
            )
        _fsync_directory(manifest_path.parent)
        if not _entry_matches_regular_bytes(
            temporary,
            expected_identity=published_identity,
            expected_bytes=manifest_bytes,
        ):
            raise StorageObjectPublicationUncertain(
                "create-only manifest staging entry changed before cleanup; retained for explicit recovery"
            )
        _unlink_owned_entry(
            temporary,
            expected_identity=published_identity,
            context="create-only manifest staging entry",
        )
        _fsync_directory(manifest_path.parent)
    except BaseException as publication_error:
        _rollback_manifest(
            manifest_path,
            published_bytes=manifest_bytes,
            previous_bytes=None,
            previous_mode=manifest_mode,
            operation_error=publication_error,
            published_identity=published_identity,
        )
        raise
    return published_identity


def _recover_candidate_from_unverified_refresh_rollback(
    temporary: Path,
    manifest_path: Path,
    *,
    staged_identity: tuple[int, int],
    staged_bytes: bytes,
    publication_error: BaseException,
    rollback_error: BaseException | None,
) -> NoReturn:
    """Restore the verified candidate after rollback exchanged an unrelated entry."""

    try:
        candidate_retained = _entry_matches_regular_bytes(
            temporary,
            expected_identity=staged_identity,
            expected_bytes=staged_bytes,
        )
        unexpected_identity = _entry_identity(manifest_path)
    except OSError as inspection_error:
        raise StorageObjectPublicationUncertain(
            "atomic refresh rollback promoted an unverified receipt; recovery outcome is uncertain"
        ) from inspection_error
    if not candidate_retained:
        raise StorageObjectPublicationUncertain(
            "atomic refresh rollback promoted an unverified receipt; candidate recovery is uncertain"
        )
    try:
        _atomic_exchange(temporary, manifest_path)
    except BaseException as recovery_error:
        recovered = _entry_matches_regular_bytes(
            manifest_path,
            expected_identity=staged_identity,
            expected_bytes=staged_bytes,
        )
        retained_unexpected = _entry_has_identity(temporary, unexpected_identity)
        if not recovered or not retained_unexpected:
            raise StorageObjectPublicationUncertain(
                "atomic refresh rollback promoted an unverified receipt; candidate recovery failed"
            ) from recovery_error
    if not _entry_matches_regular_bytes(
        manifest_path,
        expected_identity=staged_identity,
        expected_bytes=staged_bytes,
    ) or not _entry_has_identity(temporary, unexpected_identity):
        raise StorageObjectPublicationUncertain(
            "atomic refresh rollback promoted an unverified receipt; candidate recovery is uncertain"
        )
    _fsync_directory(manifest_path.parent)
    raise StorageObjectPublicationUncertain(
        "displaced receipt changed during atomic refresh rollback; retained candidate and recovery entries"
    ) from (rollback_error or publication_error)


def _publish_refresh_manifest(
    temporary: Path,
    manifest_path: Path,
    *,
    previous_bytes: bytes,
    expected_previous_identity: tuple[int, int] | None = None,
    expected_staged_identity: tuple[int, int],
    expected_staged_bytes: bytes,
) -> tuple[int, int]:
    """Exchange the staged receipt, validate the displaced receipt, and commit or swap back."""

    staged_identity = expected_staged_identity
    staged_bytes = expected_staged_bytes
    previous_identity = expected_previous_identity or _entry_identity(manifest_path)
    if not _entry_matches_regular_bytes(
        temporary,
        expected_identity=staged_identity,
        expected_bytes=staged_bytes,
    ):
        raise StorageObjectPublicationUncertain(
            "manifest staging entry changed before refresh publication; retained for explicit recovery"
        )
    exchange_error: BaseException | None = None
    try:
        _atomic_exchange(temporary, manifest_path)
    except BaseException as exc:
        exchange_error = exc
        try:
            exchanged = _entry_identity(manifest_path) == staged_identity
        except OSError as identity_error:
            raise StorageObjectPublicationUncertain(
                "atomic storage manifest exchange failed with an uncertain publication outcome"
            ) from identity_error
        if not exchanged:
            unchanged = _entry_matches_regular_bytes(
                manifest_path,
                expected_identity=previous_identity,
                expected_bytes=previous_bytes,
            ) and _entry_matches_regular_bytes(
                temporary,
                expected_identity=staged_identity,
                expected_bytes=staged_bytes,
            )
            if unchanged:
                raise
            raise StorageObjectPublicationUncertain(
                "initial refresh exchange changed publication entries; retained candidate and recovery state"
            ) from exc

    if not _entry_matches_regular_bytes(
        temporary,
        expected_identity=previous_identity,
        expected_bytes=previous_bytes,
    ):
        raise StorageObjectPublicationUncertain(
            "displaced receipt changed after atomic refresh exchange; retained candidate and recovery entries"
        )
    if not _entry_matches_regular_bytes(
        manifest_path,
        expected_identity=staged_identity,
        expected_bytes=staged_bytes,
    ):
        raise StorageObjectPublicationUncertain(
            "published refresh candidate changed after atomic exchange; retained candidate and recovery entries"
        )

    try:
        if exchange_error is not None:
            raise exchange_error
        _fsync_directory(manifest_path.parent)
        _unlink_owned_entry(
            temporary,
            expected_identity=previous_identity,
            context="refresh manifest staging entry",
        )
        _fsync_directory(manifest_path.parent)
        return staged_identity
    except BaseException as publication_error:
        if isinstance(publication_error, StorageObjectPublicationUncertain) and not _entry_matches_regular_bytes(
            temporary,
            expected_identity=previous_identity,
            expected_bytes=previous_bytes,
        ):
            raise
        if not _entry_matches_regular_bytes(
            temporary,
            expected_identity=previous_identity,
            expected_bytes=previous_bytes,
        ) or not _entry_matches_regular_bytes(
            manifest_path,
            expected_identity=staged_identity,
            expected_bytes=staged_bytes,
        ):
            raise StorageObjectPublicationUncertain(
                "displaced receipt changed before atomic refresh rollback; retained candidate and recovery entries"
            ) from publication_error
        rollback_error: BaseException | None = None
        try:
            _atomic_exchange(temporary, manifest_path)
        except BaseException as exc:
            rollback_error = exc
        rollback_succeeded = _entry_matches_regular_bytes(
            manifest_path,
            expected_identity=previous_identity,
            expected_bytes=previous_bytes,
        ) and _entry_matches_regular_bytes(
            temporary,
            expected_identity=staged_identity,
            expected_bytes=staged_bytes,
        )
        if not rollback_succeeded:
            publication_retained = _entry_matches_regular_bytes(
                manifest_path,
                expected_identity=staged_identity,
                expected_bytes=staged_bytes,
            ) and _entry_matches_regular_bytes(
                temporary,
                expected_identity=previous_identity,
                expected_bytes=previous_bytes,
            )
            if publication_retained:
                raise StorageObjectPublicationUncertain(
                    "atomic storage manifest rollback failed; retained candidate and recovery entries"
                ) from (rollback_error or publication_error)
            _recover_candidate_from_unverified_refresh_rollback(
                temporary,
                manifest_path,
                staged_identity=staged_identity,
                staged_bytes=staged_bytes,
                publication_error=publication_error,
                rollback_error=rollback_error,
            )
        try:
            _fsync_directory(manifest_path.parent)
            _unlink_owned_entry(
                temporary,
                expected_identity=staged_identity,
                context="refresh rollback staging entry",
            )
            _fsync_directory(manifest_path.parent)
        except StorageObjectPublicationUncertain:
            raise
        except BaseException as cleanup_error:
            raise StorageObjectPublicationUncertain(
                "atomic storage manifest rollback completed but cleanup outcome is uncertain"
            ) from cleanup_error
        raise publication_error


def _write_manifest(
    manifest_path: Path,
    payload: dict[str, object],
    *,
    previous_bytes: bytes | None = None,
    allow_pending_demo_manifest: bool = False,
) -> dict[str, object]:
    manifest_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    manifest_bytes = manifest_text.encode("utf-8")
    descriptor = -1
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    temporary_created = False
    preserve_temporary = False
    published_identity: tuple[int, int] | None = None
    previous_mode: int
    try:
        _preflight_owner_cleanup_directory(manifest_path.parent)
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
        temporary_identity = _entry_identity(temporary)
        temporary_created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(manifest_text)
            handle.flush()
            os.fchmod(handle.fileno(), previous_mode)
            os.fsync(handle.fileno())
        if previous_bytes is None:
            try:
                published_identity = _publish_create_only_manifest(
                    temporary,
                    manifest_path,
                    manifest_bytes=manifest_bytes,
                    manifest_mode=previous_mode,
                    expected_staged_identity=temporary_identity,
                )
            except StorageObjectPublicationUncertain:
                preserve_temporary = True
                raise
        else:
            try:
                published_identity = _publish_refresh_manifest(
                    temporary,
                    manifest_path,
                    previous_bytes=previous_bytes,
                    expected_staged_identity=temporary_identity,
                    expected_staged_bytes=manifest_bytes,
                )
            except StorageObjectPublicationUncertain:
                preserve_temporary = True
                raise
    except BaseException as write_error:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_created and temporary is not None and temporary_identity is not None and not preserve_temporary:
            try:
                _unlink_owned_entry(
                    temporary,
                    expected_identity=temporary_identity,
                    context="manifest staging entry during failed-write cleanup",
                    missing_ok=True,
                )
            except StorageObjectPublicationUnsupported:
                raise StorageObjectPublicationUncertain(
                    "publication failed and safe manifest staging cleanup is unsupported; "
                    "retained staging state for explicit recovery"
                ) from write_error
        if isinstance(write_error, OSError):
            raise StorageObjectError(f"cannot write storage object manifest: {write_error}") from write_error
        raise
    try:
        summary = verify_storage_object(
            manifest_path.parent,
            _allow_pending_demo_manifest=allow_pending_demo_manifest,
            _allow_pending_demo_lock=allow_pending_demo_manifest and previous_bytes is None,
        ).summary()
        if allow_pending_demo_manifest:
            summary["status"] = "created-pending-git-add" if previous_bytes is None else "refreshed-pending-git-add"
            object_root = shlex.quote(str(manifest_path.parent))
            python_executable = shlex.quote(sys.executable)
            summary["next_step"] = (
                f"git -C {object_root} add -- {MANIFEST_NAME} {LOCK_NAME} "
                f"&& {python_executable} -m dnadesign.contracts.storage_objects validate {object_root}"
            )
        return summary
    except BaseException as validation_error:
        _rollback_manifest(
            manifest_path,
            published_bytes=manifest_bytes,
            previous_bytes=previous_bytes,
            previous_mode=previous_mode,
            operation_error=validation_error,
            published_identity=published_identity,
        )
        raise


def _assert_held_manifest_lock_binding(
    root: Path,
    lock_path: Path,
    *,
    root_identity: tuple[int, int],
    root_mode: int,
    root_gid: int,
    lock_descriptor: int,
    lock_identity: tuple[int, int],
    lock_mode: int,
) -> None:
    """Prove the held lock inode still owns the canonical coordination pathname."""

    try:
        final_root_stat = root.stat(follow_symlinks=False)
        before = lock_path.stat(follow_symlinks=False)
        held = os.fstat(lock_descriptor)
        after = lock_path.stat(follow_symlinks=False)
    except OSError as inspection_error:
        raise StorageObjectError(
            f"cannot inspect storage object lock at completion {lock_path}: {inspection_error}"
        ) from inspection_error
    if (
        (final_root_stat.st_dev, final_root_stat.st_ino) != root_identity
        or stat.S_IMODE(final_root_stat.st_mode) != root_mode
        or final_root_stat.st_gid != root_gid
    ):
        raise StorageObjectError(f"storage object root posture changed while holding its manifest lock: {root}")
    before_identity = (before.st_dev, before.st_ino)
    held_identity = (held.st_dev, held.st_ino)
    after_identity = (after.st_dev, after.st_ino)
    if before_identity != lock_identity or held_identity != lock_identity or after_identity != lock_identity:
        raise StorageObjectError(f"storage object lock changed before operation completion: {lock_path}")
    if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(held.st_mode) or not stat.S_ISREG(after.st_mode):
        raise StorageObjectError(f"storage object lock must remain a regular file through completion: {lock_path}")
    if before.st_size != 0 or held.st_size != 0 or after.st_size != 0:
        raise StorageObjectError(f"storage object lock must remain empty through completion: {lock_path}")
    if (
        stat.S_IMODE(before.st_mode) != lock_mode
        or stat.S_IMODE(held.st_mode) != lock_mode
        or stat.S_IMODE(after.st_mode) != lock_mode
    ):
        raise StorageObjectError(f"storage object lock mode changed before operation completion: {lock_path}")
    if root_mode & stat.S_IWGRP and (
        before.st_gid != root_gid
        or held.st_gid != root_gid
        or after.st_gid != root_gid
        or not before.st_mode & stat.S_IWGRP
        or not before.st_mode & stat.S_IRGRP
    ):
        raise StorageObjectError(f"storage object lock shared-group posture changed before completion: {lock_path}")


@contextmanager
def _manifest_lock(root: Path, *, allow_missing: bool = False) -> Iterator[None]:
    lock_path = root / LOCK_NAME
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
        if lock_path.is_symlink():
            raise StorageObjectError(f"storage object lock must be a regular file: {lock_path}")
        try:
            inspected_lock_stat = lock_path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            if not allow_missing:
                raise StorageObjectError(f"storage object lock is missing: {lock_path}") from exc
            manifest_path = root / MANIFEST_NAME
            if manifest_path.exists() or manifest_path.is_symlink():
                raise StorageObjectError(
                    f"cannot bootstrap a missing storage object lock beside an existing manifest: {manifest_path}"
                ) from exc
            inspected_lock_identity: tuple[int, int] | None = None
        else:
            if not stat.S_ISREG(inspected_lock_stat.st_mode):
                raise StorageObjectError(f"storage object lock must be a regular file: {lock_path}")
            inspected_lock_identity = (inspected_lock_stat.st_dev, inspected_lock_stat.st_ino)
            if inspected_lock_stat.st_size != 0:
                raise StorageObjectError(f"storage object lock must be an empty coordination file: {lock_path}")
            lock_mode = stat.S_IMODE(inspected_lock_stat.st_mode)
            owner_required = stat.S_IRUSR | stat.S_IWUSR
            if lock_mode & owner_required != owner_required:
                raise StorageObjectError(f"storage object lock must be owner-readable and owner-writable: {lock_path}")
            if root_mode & stat.S_IWGRP and inspected_lock_stat.st_gid != root_stat.st_gid:
                raise StorageObjectError(f"storage object lock does not inherit the shared object group: {lock_path}")
            if root_mode & stat.S_IWGRP and not inspected_lock_stat.st_mode & stat.S_IWGRP:
                raise StorageObjectError(
                    f"storage object lock must be group-writable in a shared object root: {lock_path}"
                )
            if root_mode & stat.S_IWGRP and not inspected_lock_stat.st_mode & stat.S_IRGRP:
                raise StorageObjectError(
                    f"storage object lock must be group-readable in a shared object root: {lock_path}"
                )
    except OSError as exc:
        raise StorageObjectError(f"cannot inspect storage object lock {lock_path}: {exc}") from exc
    lock_mode = 0o664 if root_mode & stat.S_IWGRP else 0o644
    lock = FileLock(lock_path, timeout=30, mode=lock_mode)
    try:
        lock.acquire()
    except Timeout as exc:
        raise StorageObjectError(f"timed out waiting for storage object manifest lock: {root}") from exc
    except (OSError, NotImplementedError) as exc:
        raise StorageObjectError(f"cannot acquire storage object manifest lock {lock_path}: {exc}") from exc
    try:
        lock_stat = lock_path.stat(follow_symlinks=False)
        acquired_lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
        if inspected_lock_identity is not None and acquired_lock_identity != inspected_lock_identity:
            raise StorageObjectError(f"storage object lock changed before acquisition completed: {lock_path}")
        lock_context = getattr(lock, "_context", None)
        lock_descriptor = getattr(lock_context, "lock_file_fd", None)
        if not isinstance(lock_descriptor, int):
            raise StorageObjectError(f"cannot identify the held storage object lock descriptor: {lock_path}")
        held_stat = os.fstat(lock_descriptor)
        if (held_stat.st_dev, held_stat.st_ino) != acquired_lock_identity:
            raise StorageObjectError(f"storage object lock changed before acquisition completed: {lock_path}")
        lock_mode = stat.S_IMODE(lock_stat.st_mode)
        owner_required = stat.S_IRUSR | stat.S_IWUSR
        if lock_mode & owner_required != owner_required:
            raise StorageObjectError(f"storage object lock must be owner-readable and owner-writable: {lock_path}")
        if root_mode & stat.S_IWGRP and lock_stat.st_gid != root_stat.st_gid:
            raise StorageObjectError(f"storage object lock does not inherit the shared object group: {lock_path}")
        if root_mode & stat.S_IWGRP and not lock_stat.st_mode & stat.S_IWGRP:
            raise StorageObjectError(f"storage object lock must be group-writable in a shared object root: {lock_path}")
        if root_mode & stat.S_IWGRP and not lock_stat.st_mode & stat.S_IRGRP:
            raise StorageObjectError(f"storage object lock must be group-readable in a shared object root: {lock_path}")
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
        completion_error: StorageObjectError | None = None
        try:
            _assert_held_manifest_lock_binding(
                root,
                lock_path,
                root_identity=(root_stat.st_dev, root_stat.st_ino),
                root_mode=root_mode,
                root_gid=root_stat.st_gid,
                lock_descriptor=lock_descriptor,
                lock_identity=acquired_lock_identity,
                lock_mode=lock_mode,
            )
        except StorageObjectError as inspection_error:
            completion_error = inspection_error
        committed_manifest_digest = "unavailable"
        if completion_error is None:
            try:
                committed_manifest_digest = _sha256_bytes((root / MANIFEST_NAME).read_bytes())
            except OSError:
                pass
        revalidation_command = (
            f"{shlex.quote(sys.executable)} -m dnadesign.contracts.storage_objects validate {shlex.quote(str(root))}"
        )
        try:
            lock.release()
        except OSError as release_error:
            if completion_error is not None:
                raise StorageObjectPublicationUncertain(
                    "storage operation committed but its coordination lock changed before completion "
                    f"and the held lock could not be released: {lock_path}: {release_error}"
                ) from completion_error
            raise StorageObjectPublicationUncertain(
                "storage operation committed and verified, but its manifest lock release failed; "
                f"winning_manifest_digest={committed_manifest_digest}; do not retry with the prior CAS digest; "
                f"revalidate with `{revalidation_command}`; release_error={release_error}"
            ) from release_error
        if completion_error is not None:
            raise StorageObjectPublicationUncertain(
                "storage operation committed but its coordination lock changed before completion; "
                f"revalidate committed state explicitly: {lock_path}"
            ) from completion_error


def _assert_no_ambiguous_manifest_staging(root: Path) -> None:
    """Fail closed rather than deleting a staging-shaped user file."""

    candidates = {root / f".{MANIFEST_NAME}.tmp"}
    candidates.update(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    candidates.update(root.glob(f".{MANIFEST_NAME}.restore-*"))
    candidates.update(root.glob(f".{MANIFEST_NAME}.rollback-*"))
    candidates.update(
        candidate
        for candidate in root.glob(f".*{MANIFEST_NAME}*.cleanup-*")
        if not candidate.name.startswith(f".{MANIFEST_NAME}.cleanup-owner-")
    )
    for cleanup_directory in root.glob(f".{MANIFEST_NAME}.cleanup-owner-*"):
        if cleanup_directory.is_symlink() or not cleanup_directory.is_dir():
            candidates.add(cleanup_directory)
            continue
        try:
            candidates.update(cleanup_directory.iterdir())
        except OSError as inspection_error:
            raise StorageObjectError(
                f"cannot inspect storage cleanup recovery state: {cleanup_directory}"
            ) from inspection_error
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

    _require_posix_publication_capabilities()
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
    with _manifest_lock(root, allow_missing=True):
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
            allow_pending_demo_manifest=demo,
        )


def refresh_storage_object(
    storage_root: Path,
    *,
    expected_manifest_digest: str,
    producer_revision: str,
    artifact_paths: tuple[str, ...] = (),
    cache_paths: tuple[str, ...] = (),
) -> dict[str, object]:
    """Refresh changed outputs and explicitly demote mistaken metadata roles."""

    _require_posix_publication_capabilities()
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
        try:
            previous_bytes = manifest_path.read_bytes()
        except OSError as exc:
            raise StorageObjectError(f"cannot read storage object manifest {manifest_path}: {exc}") from exc
        observed_manifest_digest = _sha256_bytes(previous_bytes)
        if observed_manifest_digest != expected_manifest_digest:
            raise StorageObjectError(
                "storage object manifest changed before refresh: "
                f"expected {expected_manifest_digest}, observed {observed_manifest_digest}"
            )
        prior_manifest_is_git_resident = verify_manifest_index_if_git_resident(
            root,
            manifest_path,
            observed_manifest_digest,
        )
        manifest = load_storage_object_manifest_bytes(previous_bytes, source_label=str(manifest_path))
        if manifest.demo and not prior_manifest_is_git_resident:
            raise StorageObjectError(f"demo storage object must live inside a Git checkout: {root}")
        if manifest.object_kind not in {ObjectKind.WORKSPACE, ObjectKind.STORE}:
            raise StorageObjectError(
                "storage receipt refresh is limited to active workspaces and stores; "
                f"found object_kind={manifest.object_kind.value}"
            )
        prior_resources = {resource.relative_path: resource for resource in manifest.resources}
        prior_roles = {path: resource.role for path, resource in prior_resources.items()}
        normalized_artifacts = {normalize_relative_path(path, label="artifact path") for path in artifact_paths}
        normalized_caches = {normalize_relative_path(path, label="cache path") for path in cache_paths}
        duplicate_roles = sorted(normalized_artifacts & normalized_caches)
        if duplicate_roles:
            raise StorageObjectError(f"refresh paths have multiple roles: {', '.join(duplicate_roles)}")
        files = tuple(
            path
            for path in storage_file_paths(root)
            if path.name not in {MANIFEST_NAME, LOCK_NAME} or path.parent != root
        )
        relative_files = {path.relative_to(root).as_posix() for path in files}
        unknown_artifacts = sorted(normalized_artifacts - set(prior_roles))
        if unknown_artifacts:
            raise StorageObjectError(
                "artifact reclassification requires an existing receipt resource: " + ", ".join(unknown_artifacts)
            )
        missing_artifacts = sorted(normalized_artifacts - relative_files)
        if missing_artifacts:
            raise StorageObjectError(
                "artifact reclassification target is missing from the storage object: " + ", ".join(missing_artifacts)
            )
        invalid_artifact_roles = sorted(
            path
            for path in normalized_artifacts
            if prior_roles[path] not in {ResourceRole.METADATA, ResourceRole.ARTIFACT}
        )
        if invalid_artifact_roles:
            raise StorageObjectError(
                "only metadata may be reclassified as artifact; input and cache roles remain protected: "
                + ", ".join(invalid_artifact_roles)
            )
        invalid_cache_roles = sorted(
            path
            for path in normalized_caches
            if path in prior_roles and prior_roles[path] not in {ResourceRole.ARTIFACT, ResourceRole.CACHE}
        )
        if invalid_cache_roles:
            raise StorageObjectError(
                "only artifacts may be reclassified as cache; input and metadata roles remain protected: "
                + ", ".join(invalid_cache_roles)
            )
        effective_roles = {
            path: (
                ResourceRole.ARTIFACT
                if path in normalized_artifacts
                else (ResourceRole.CACHE if path in normalized_caches else role)
            )
            for path, role in prior_roles.items()
        }
        protected_paths = {
            path for path, role in effective_roles.items() if role in {ResourceRole.INPUT, ResourceRole.METADATA}
        }
        missing_protected = sorted(protected_paths - relative_files)
        if missing_protected:
            raise StorageObjectError(
                f"cannot refresh after removing input or metadata files: {', '.join(missing_protected)}"
            )
        protected_digests = {path: _sha256(root / path) for path in protected_paths}
        changed_protected = sorted(
            path for path, digest in protected_digests.items() if digest != prior_resources[path].digest
        )
        if changed_protected:
            raise StorageObjectError(
                f"cannot refresh after changing input or metadata files: {', '.join(changed_protected)}"
            )
        missing_caches = sorted(normalized_caches - relative_files)
        if missing_caches:
            raise StorageObjectError(f"declared refresh cache paths are missing: {', '.join(missing_caches)}")
        resources = []
        for path in files:
            relative_path = path.relative_to(root).as_posix()
            role = effective_roles.get(relative_path)
            if role is None:
                role = ResourceRole.CACHE if relative_path in normalized_caches else ResourceRole.ARTIFACT
            digest = _sha256(path)
            if relative_path in protected_digests and digest != protected_digests[relative_path]:
                raise StorageObjectError(f"protected resource changed during refresh: {relative_path}")
            resources.append(
                {
                    "digest": digest,
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
        return _write_manifest(
            manifest_path,
            payload,
            previous_bytes=previous_bytes,
            allow_pending_demo_manifest=manifest.demo,
        )
