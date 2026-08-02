"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/publication/snapshot.py

Private stable-filesystem snapshots for TriJunction bundle verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from dnadesign.trijunction.errors import TriJunctionBundleError

_MIB = 1024 * 1024
_StatIdentity = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class SnapshotRead:
    relative: Path
    context: str
    observed_bytes: int
    sha256: str
    content: bytes | None
    _descriptor: int
    _identity: _StatIdentity


class BundleFilesystemSnapshot:
    """Retain and later stabilize one descriptor-anchored bundle view."""

    def __init__(
        self,
        *,
        root: Path,
        root_descriptor: int,
        descriptor_stack: ExitStack,
        expected_files: frozenset[str],
        reject_undeclared_entries: bool,
        directory_snapshots: dict[Path, tuple[int, _StatIdentity]],
    ) -> None:
        self._root = root
        self._root_descriptor = root_descriptor
        self._descriptor_stack = descriptor_stack
        self._expected_files = expected_files
        self._reject_undeclared_entries = reject_undeclared_entries
        self._directory_snapshots = directory_snapshots
        self._reads: list[SnapshotRead] = []

    def read_file(
        self,
        relative: Path,
        *,
        limit: int,
        context: str,
        retain_content: bool,
    ) -> SnapshotRead:
        descriptor = _open_bundle_file(self._root_descriptor, relative, context=context)
        _retain_descriptor(self._descriptor_stack, descriptor)
        observed_bytes, observed_sha256, content, identity = _read_descriptor(
            descriptor,
            limit=limit,
            context=context,
            retain_content=retain_content,
        )
        result = SnapshotRead(
            relative=relative,
            context=context,
            observed_bytes=observed_bytes,
            sha256=observed_sha256,
            content=content,
            _descriptor=descriptor,
            _identity=identity,
        )
        self._reads.append(result)
        return result

    def assert_stable(self) -> None:
        """Require one stable view across final path and inventory checks.

        Mutations after the last observable identity check are necessarily
        outside a single verifier invocation's stable-snapshot boundary.
        """

        self._assert_retained_files()
        for result in self._reads:
            _revalidate_bundle_file(
                self._root_descriptor,
                result.relative,
                context=result.context,
                expected_identity=result._identity,
            )
        self._assert_retained_files()
        if self._reject_undeclared_entries:
            _reject_undeclared_entries(
                self._root_descriptor,
                expected_files=self._expected_files,
            )
        self._assert_retained_files()
        _assert_retained_directories(self._directory_snapshots)
        _revalidate_directory_paths(
            self._root,
            self._root_descriptor,
            self._directory_snapshots,
        )
        _assert_retained_directories(self._directory_snapshots)

    def _assert_retained_files(self) -> None:
        for result in self._reads:
            _assert_retained_file(
                result._descriptor,
                context=result.context,
                expected_identity=result._identity,
            )


@contextmanager
def open_bundle_snapshot(
    root: Path,
    *,
    expected_files: frozenset[str],
    reject_undeclared_entries: bool,
) -> Iterator[BundleFilesystemSnapshot]:
    with ExitStack() as descriptor_stack:
        root_descriptor = _open_bundle_directory(root)
        _retain_descriptor(descriptor_stack, root_descriptor)
        if reject_undeclared_entries:
            _reject_undeclared_entries(root_descriptor, expected_files=expected_files)
        directory_snapshots = _retain_directory_snapshots(
            root_descriptor,
            descriptor_stack,
            expected_files=expected_files,
        )
        yield BundleFilesystemSnapshot(
            root=root,
            root_descriptor=root_descriptor,
            descriptor_stack=descriptor_stack,
            expected_files=expected_files,
            reject_undeclared_entries=reject_undeclared_entries,
            directory_snapshots=directory_snapshots,
        )


def _descriptor_flags(*, directory: bool) -> int:
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        if directory:
            flags |= os.O_DIRECTORY
    except AttributeError as exc:
        raise TriJunctionBundleError("Secure TriJunction bundle verification is unavailable on this platform.") from exc
    return flags


def _retain_descriptor(descriptor_stack: ExitStack, descriptor: int) -> None:
    try:
        descriptor_stack.callback(os.close, descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _open_bundle_directory(root: Path) -> int:
    descriptor = os.open(root, _descriptor_flags(directory=True))
    try:
        opened = os.fstat(descriptor)
    except OSError:
        os.close(descriptor)
        raise
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise TriJunctionBundleError(f"TriJunction bundle is not a regular directory: {root}")
    return descriptor


def _open_bundle_file(root_descriptor: int, relative: Path, *, context: str) -> int:
    with ExitStack() as parent_descriptors:
        directory_descriptor = root_descriptor
        for part in relative.parts[:-1]:
            directory_descriptor = os.open(
                part,
                _descriptor_flags(directory=True),
                dir_fd=directory_descriptor,
            )
            _retain_descriptor(parent_descriptors, directory_descriptor)
            if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
                raise TriJunctionBundleError(f"{context} parent is not a regular directory: {relative}")
        descriptor = os.open(
            relative.parts[-1],
            _descriptor_flags(directory=False),
            dir_fd=directory_descriptor,
        )
        try:
            opened = os.fstat(descriptor)
        except OSError:
            os.close(descriptor)
            raise
        if not stat.S_ISREG(opened.st_mode):
            os.close(descriptor)
            raise TriJunctionBundleError(f"{context} is not a regular file: {relative}")
        return descriptor


def _open_bundle_subdirectory(root_descriptor: int, relative: Path) -> int:
    with ExitStack() as directory_descriptors:
        directory_descriptor = root_descriptor
        for part in relative.parts:
            directory_descriptor = os.open(
                part,
                _descriptor_flags(directory=True),
                dir_fd=directory_descriptor,
            )
            _retain_descriptor(directory_descriptors, directory_descriptor)
            if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
                raise TriJunctionBundleError(f"Bundle directory is not a regular directory: {relative}")
        return os.dup(directory_descriptor)


def _stat_identity(value: os.stat_result) -> _StatIdentity:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _read_descriptor(
    descriptor: int,
    *,
    limit: int,
    context: str,
    retain_content: bool,
) -> tuple[int, str, bytes | None, _StatIdentity]:
    initial_identity = _stat_identity(os.fstat(descriptor))
    if initial_identity[5] < 1:
        raise TriJunctionBundleError(f"{context} is no longer linked into the bundle.")
    if initial_identity[2] > limit:
        raise TriJunctionBundleError(f"{context} exceeds the {limit}-byte verification limit.")
    digest = sha256()
    observed = 0
    content = bytearray() if retain_content else None
    while chunk := os.read(descriptor, _MIB):
        observed += len(chunk)
        if observed > limit:
            raise TriJunctionBundleError(f"{context} exceeds the {limit}-byte verification limit.")
        digest.update(chunk)
        if content is not None:
            content.extend(chunk)
    if _stat_identity(os.fstat(descriptor)) != initial_identity:
        raise TriJunctionBundleError(f"{context} changed while it was being verified.")
    retained = bytes(content) if content is not None else None
    return observed, f"sha256:{digest.hexdigest()}", retained, initial_identity


def _revalidate_bundle_file(
    root_descriptor: int,
    relative: Path,
    *,
    context: str,
    expected_identity: _StatIdentity,
) -> None:
    descriptor = _open_bundle_file(root_descriptor, relative, context=context)
    try:
        observed_identity = _stat_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if observed_identity != expected_identity:
        raise TriJunctionBundleError(f"{context} changed after it was verified.")


def _assert_retained_file(
    descriptor: int,
    *,
    context: str,
    expected_identity: _StatIdentity,
) -> None:
    observed_identity = _stat_identity(os.fstat(descriptor))
    if observed_identity[5] < 1:
        raise TriJunctionBundleError(f"{context} is no longer linked into the bundle.")
    if observed_identity != expected_identity:
        raise TriJunctionBundleError(f"{context} changed after it was verified.")


def _expected_directory_paths(expected_files: frozenset[str]) -> tuple[Path, ...]:
    directories = {parent for relative in expected_files for parent in Path(relative).parents}
    return tuple(sorted(directories, key=lambda path: (len(path.parts), path.as_posix())))


def _retain_directory_snapshots(
    root_descriptor: int,
    descriptor_stack: ExitStack,
    *,
    expected_files: frozenset[str],
) -> dict[Path, tuple[int, _StatIdentity]]:
    snapshots: dict[Path, tuple[int, _StatIdentity]] = {}
    for relative in _expected_directory_paths(expected_files):
        if relative == Path("."):
            descriptor = root_descriptor
        else:
            descriptor = _open_bundle_subdirectory(root_descriptor, relative)
            _retain_descriptor(descriptor_stack, descriptor)
        identity = _stat_identity(os.fstat(descriptor))
        if identity[5] < 1:
            raise TriJunctionBundleError(f"Bundle directory is no longer linked: {relative}")
        snapshots[relative] = (descriptor, identity)
    return snapshots


def _assert_retained_directories(
    snapshots: dict[Path, tuple[int, _StatIdentity]],
) -> None:
    for relative, (descriptor, expected_identity) in snapshots.items():
        observed_identity = _stat_identity(os.fstat(descriptor))
        if observed_identity[5] < 1 or observed_identity != expected_identity:
            raise TriJunctionBundleError(f"Bundle directory changed during verification: {relative}")


def _revalidate_directory_paths(
    root: Path,
    root_descriptor: int,
    snapshots: dict[Path, tuple[int, _StatIdentity]],
) -> None:
    for relative, (_, expected_identity) in snapshots.items():
        if relative == Path("."):
            descriptor = _open_bundle_directory(root)
        else:
            descriptor = _open_bundle_subdirectory(root_descriptor, relative)
        try:
            observed_identity = _stat_identity(os.fstat(descriptor))
        finally:
            os.close(descriptor)
        if observed_identity != expected_identity:
            raise TriJunctionBundleError(f"Bundle directory path changed during verification: {relative}")


def _reject_undeclared_entries(
    root_descriptor: int,
    *,
    expected_files: frozenset[str],
) -> None:
    expected_directories = {path.as_posix() for path in _expected_directory_paths(expected_files) if path != Path(".")}

    def inspect(directory_descriptor: int, *, prefix: Path) -> None:
        with os.scandir(directory_descriptor) as entries:
            for entry in entries:
                relative_path = prefix / entry.name
                relative = relative_path.as_posix()
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise TriJunctionBundleError(f"TriJunction bundle must not contain symlinks: {relative}")
                if stat.S_ISREG(entry_stat.st_mode):
                    if relative not in expected_files:
                        raise TriJunctionBundleError(f"TriJunction bundle contains undeclared file: {relative}")
                    continue
                if stat.S_ISDIR(entry_stat.st_mode):
                    if relative not in expected_directories:
                        raise TriJunctionBundleError(f"TriJunction bundle contains undeclared directory: {relative}")
                    child_descriptor = os.open(
                        entry.name,
                        _descriptor_flags(directory=True),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        inspect(child_descriptor, prefix=relative_path)
                    finally:
                        os.close(child_descriptor)
                    continue
                raise TriJunctionBundleError(f"TriJunction bundle contains unsupported entry: {relative}")

    inspect(root_descriptor, prefix=Path())
