"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/storage_objects/locking.py

Descriptor-bound advisory locking for storage-object coordination files.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import errno
import os
import stat
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

from .models import StorageObjectError, StorageObjectPublicationUncertain

PRIVATE_LOCK_MODE = 0o600
SHARED_LOCK_MODE = 0o660


def unavailable_locking_capabilities() -> tuple[str, ...]:
    """Return capabilities required before a writer may mutate coordination state."""

    unavailable: list[str] = []
    if fcntl is None or not callable(getattr(fcntl, "flock", None)):
        unavailable.append("fcntl.flock")
    for name in ("close", "fchmod", "fstat", "fsync", "open", "stat"):
        if not callable(getattr(os, name, None)):
            unavailable.append(name)
    for name in ("O_CLOEXEC", "O_CREAT", "O_DIRECTORY", "O_EXCL", "O_NOFOLLOW", "O_NONBLOCK"):
        if not hasattr(os, name):
            unavailable.append(name)
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    for name in ("open", "stat"):
        if not any(getattr(function, "__name__", None) == name for function in supports_dir_fd):
            unavailable.append(f"{name}_dir_fd")
    return tuple(unavailable)


def _require_locking_capabilities(lock_path: Path) -> None:
    unavailable = unavailable_locking_capabilities()
    if unavailable:
        raise StorageObjectError(
            "storage-object coordination requires POSIX no-follow descriptor locking; "
            f"unavailable: {', '.join(unavailable)}; cannot acquire {lock_path}"
        )


def _acquire_flock(descriptor: int, lock_path: Path, *, timeout_seconds: float) -> None:
    if fcntl is None:  # pragma: no cover - capability guard rejects this platform
        raise StorageObjectError(f"cannot acquire storage object manifest lock {lock_path}: fcntl unavailable")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise StorageObjectError(f"cannot acquire storage object manifest lock {lock_path}: {exc}") from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StorageObjectError(f"timed out waiting for storage object manifest lock: {lock_path.parent}")
            time.sleep(min(0.05, remaining))


def _close_descriptors_after_failure(
    descriptors: tuple[tuple[int | None, str], ...],
    *,
    primary_error: BaseException,
) -> None:
    """Close every owned descriptor without replacing the primary typed outcome."""

    close_errors: list[str] = []
    for descriptor, label in descriptors:
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except OSError as exc:
            close_errors.append(f"{label}: {exc}")
    if not close_errors:
        return
    message = f"{primary_error}; descriptor cleanup also failed: {'; '.join(close_errors)}"
    if isinstance(primary_error, StorageObjectPublicationUncertain):
        raise StorageObjectPublicationUncertain(message) from primary_error
    raise StorageObjectError(message) from primary_error


def acquire_existing_lock(
    lock_path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_mode: int,
    expected_gid: int,
    expected_size: int = 0,
    timeout_seconds: float = 30.0,
) -> int:
    """Lock one pre-inspected regular file without create, follow, or truncation."""

    _require_locking_capabilities(lock_path)
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        raise StorageObjectError(f"cannot open existing storage object lock {lock_path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != expected_identity:
            raise StorageObjectError(f"storage object lock changed before acquisition completed: {lock_path}")
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != expected_mode
            or opened.st_gid != expected_gid
            or opened.st_size != expected_size
        ):
            raise StorageObjectError(f"storage object lock posture changed before acquisition completed: {lock_path}")
        _acquire_flock(descriptor, lock_path, timeout_seconds=timeout_seconds)
        return descriptor
    except BaseException as primary_error:
        _close_descriptors_after_failure(((descriptor, "existing lock"),), primary_error=primary_error)
        raise


def acquire_new_lock(
    lock_path: Path,
    *,
    mode: int,
    expected_gid: int | None,
    expected_parent_identity: tuple[int, int],
    timeout_seconds: float = 30.0,
) -> tuple[int, tuple[int, int]]:
    """Exclusively create, durably bind, and lock one absent coordination file."""

    _require_locking_capabilities(lock_path)
    if mode not in {PRIVATE_LOCK_MODE, SHARED_LOCK_MODE}:
        raise StorageObjectError(f"new storage object lock mode must be 0600 or 0660: {lock_path}")
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        parent_descriptor: int | None = os.open(lock_path.parent, parent_flags)
    except OSError as exc:
        raise StorageObjectError(
            f"cannot open storage object root for lock bootstrap {lock_path.parent}: {exc}"
        ) from exc
    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    try:
        assert parent_descriptor is not None
        parent_stat = os.fstat(parent_descriptor)
        if (parent_stat.st_dev, parent_stat.st_ino) != expected_parent_identity:
            raise StorageObjectError(f"storage object root changed before lock bootstrap: {lock_path.parent}")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        try:
            descriptor = os.open(lock_path.name, flags, PRIVATE_LOCK_MODE, dir_fd=parent_descriptor)
        except OSError as exc:
            raise StorageObjectError(f"cannot exclusively create storage object lock {lock_path}: {exc}") from exc
        try:
            os.fchmod(descriptor, mode)
            opened = os.fstat(descriptor)
            created_identity = (opened.st_dev, opened.st_ino)
            named = os.stat(lock_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != created_identity:
                raise StorageObjectError(f"storage object lock changed during bootstrap: {lock_path}")
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(named.st_mode)
                or opened.st_size != 0
                or named.st_size != 0
                or stat.S_IMODE(opened.st_mode) != mode
                or stat.S_IMODE(named.st_mode) != mode
                or opened.st_gid != named.st_gid
                or (expected_gid is not None and opened.st_gid != expected_gid)
            ):
                raise StorageObjectError(f"storage object lock posture is invalid after bootstrap: {lock_path}")
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
            durable_named = os.stat(lock_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            if (durable_named.st_dev, durable_named.st_ino) != created_identity:
                raise StorageObjectError(f"storage object lock changed during durable bootstrap: {lock_path}")
            closing_parent = parent_descriptor
            parent_descriptor = None
            os.close(closing_parent)
            _acquire_flock(descriptor, lock_path, timeout_seconds=timeout_seconds)
            return descriptor, created_identity
        except BaseException as exc:
            raise StorageObjectPublicationUncertain(
                "storage object lock bootstrap did not complete after exclusive creation; "
                f"inspect persistent coordination state before retrying: {lock_path}"
            ) from exc
    except BaseException as primary_error:
        _close_descriptors_after_failure(
            ((descriptor, "new lock"), (parent_descriptor, "object root")),
            primary_error=primary_error,
        )
        raise


def release_lock(descriptor: int) -> None:
    """Release and close one descriptor-owned advisory lock."""

    if fcntl is None:  # pragma: no cover - acquisition rejects this platform
        os.close(descriptor)
        return
    unlock_error: OSError | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError as exc:
        unlock_error = exc
    try:
        os.close(descriptor)
    except OSError as close_error:
        if unlock_error is not None:
            raise OSError(f"lock release failed: {unlock_error}; descriptor close also failed: {close_error}") from (
                unlock_error
            )
        raise
    if unlock_error is not None:
        raise unlock_error
