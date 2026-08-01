"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/artifacts/publication.py

Publish immutable directory artifacts atomically and without replacement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .errors import PublicationError, PublicationExistsError
from .owned_directory import (
    descriptor_matches_entry,
    owner_matches_descriptor,
    remove_descriptor_anchored_directory,
    remove_owned_directory,
    remove_owned_named_directory,
)
from .portable_paths import validate_publication_metadata_paths
from .recovery import (
    _OWNER_FILE,
    _PUBLICATION_OWNER_SCHEMA,
    _ROLLBACK_OWNER_SCHEMA,
    _ensure_owner_on_descriptor,
    _open_recoverable_owned_directory,
    _owner_payload,
    _recover_owned_adjacent_directories,
    _remove_owner_from_descriptor,
    _rollback_owner_payload,
    _write_owner,
)

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_DEFAULT_PUBLISHED_ROOT_MODE = 0o755


def _validate_published_root_mode(mode: int) -> int:
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise PublicationError("Artifact bundle published root mode must be an integer permission mode")
    if mode < 0 or mode > 0o777:
        raise PublicationError("Artifact bundle published root mode must contain only permission bits")
    if mode & 0o700 != 0o700:
        raise PublicationError("Artifact bundle published root mode must grant its owner read, write, and execute")
    return mode


def _lexical_absolute_path(path: Path) -> Path:
    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise PublicationError(f"Publication output path must not contain parent traversal: {path}")
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _preflight_existing_path_components(path: Path) -> None:
    """Reject redirects and non-directories before creating any missing parent."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            entry_stat = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(entry_stat.st_mode):
            raise PublicationError(f"Publication output contains a symlinked path component: {current}")
        if not stat.S_ISDIR(entry_stat.st_mode):
            raise PublicationError(f"Publication output parent component is not a directory: {current}")


def _open_or_create_directory(path: Path) -> int:
    if not path.is_absolute():
        raise PublicationError(f"Publication output parent must be absolute: {path}")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _entry_exists_at(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _copy_file(source: Path, parent_descriptor: int, name: str) -> None:
    source_flags = os.O_RDONLY
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
        destination_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, source_flags)
    try:
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise PublicationError(f"Bundle staging must contain regular files: {source}")
        destination_descriptor = os.open(
            name,
            destination_flags,
            _PRIVATE_FILE_MODE,
            dir_fd=parent_descriptor,
        )
        try:
            with os.fdopen(os.dup(source_descriptor), "rb") as source_handle:
                with os.fdopen(os.dup(destination_descriptor), "wb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle)
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)


def _copy_directory(source: Path, parent_descriptor: int, name: str) -> None:
    os.mkdir(name, mode=_PRIVATE_DIRECTORY_MODE, dir_fd=parent_descriptor)
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    destination_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        entries = sorted(os.scandir(source), key=lambda entry: (entry.name != _OWNER_FILE, entry.name))
        for entry in entries:
            entry_path = Path(entry.path)
            if entry.is_symlink():
                raise PublicationError(f"Bundle staging must not contain symlinks: {entry_path}")
            if entry.is_dir(follow_symlinks=False):
                _copy_directory(entry_path, destination_descriptor, entry.name)
            elif entry.is_file(follow_symlinks=False):
                _copy_file(entry_path, destination_descriptor, entry.name)
            else:
                raise PublicationError(f"Bundle staging contains an unsupported entry: {entry_path}")
    finally:
        os.close(destination_descriptor)


def _restore_published_modes(source: Path, destination_descriptor: int) -> None:
    """Restore staged modes while the renamed bundle root remains owner-only."""
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for entry in os.scandir(source):
        if entry.name == _OWNER_FILE:
            continue
        entry_path = Path(entry.path)
        source_stat = entry_path.stat(follow_symlinks=False)
        mode = stat.S_IMODE(source_stat.st_mode) & 0o777
        if entry.is_dir(follow_symlinks=False):
            child_descriptor = os.open(entry.name, flags, dir_fd=destination_descriptor)
            try:
                _restore_published_modes(entry_path, child_descriptor)
                os.fchmod(child_descriptor, mode)
            finally:
                os.close(child_descriptor)
        elif entry.is_file(follow_symlinks=False):
            os.chmod(entry.name, mode, dir_fd=destination_descriptor, follow_symlinks=False)
        else:
            raise PublicationError(f"Bundle staging contains an unsupported entry: {entry_path}")


def _rename_create_only(parent_descriptor: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(parent_descriptor, source_bytes, parent_descriptor, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(parent_descriptor, source_bytes, parent_descriptor, destination_bytes, 0x1)
    else:
        raise PublicationError("This platform does not support atomic create-only directory publication")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PublicationExistsError(f"Artifact bundle already exists and is immutable: {destination}")
    raise OSError(error, os.strerror(error), destination)


def _restore_entry_after_detachment_mismatch(
    parent_descriptor: int,
    rollback_name: str,
    final_name: str,
    published_descriptor: int,
    *,
    clear_owner: bool = True,
) -> None:
    restore_error: BaseException | None = None
    try:
        _rename_create_only(
            parent_descriptor,
            rollback_name,
            final_name,
        )
    except BaseException as exc:
        restore_error = exc
    if clear_owner:
        try:
            _remove_owner_from_descriptor(published_descriptor)
        except BaseException as owner_error:
            raise PublicationError(
                "Artifact bundle identity changed during atomic rollback detachment, "
                "and its rollback owner sentinel could not be cleared"
            ) from owner_error
    if restore_error is not None:
        raise PublicationError(
            "Artifact bundle identity changed during atomic rollback detachment, "
            "and the displaced entry could not be restored"
        ) from restore_error
    raise PublicationError(
        "Artifact bundle identity changed during atomic rollback detachment; "
        "the displaced entry was restored without cleanup"
    )


def _recover_final_directory(
    parent_descriptor: int,
    final: Path,
    *,
    uid: int | None,
) -> bool:
    owner_schemas = (_PUBLICATION_OWNER_SCHEMA, _ROLLBACK_OWNER_SCHEMA)
    opened = _open_recoverable_owned_directory(
        parent_descriptor,
        final.name,
        final=final,
        uid=uid,
        owner_schema=owner_schemas,
    )
    if opened is None:
        return False
    descriptor, owner = opened
    try:
        owner_pid = int(owner["pid"])
        owner_uid = owner["uid"]
        rollback_name = f".{final.name}.rollback-u{owner_uid}-p{owner_pid}-{uuid.uuid4().hex}"
        if not descriptor_matches_entry(parent_descriptor, final.name, descriptor):
            return False
        _rename_create_only(parent_descriptor, final.name, rollback_name)
        if not descriptor_matches_entry(parent_descriptor, rollback_name, descriptor):
            _restore_entry_after_detachment_mismatch(
                parent_descriptor,
                rollback_name,
                final.name,
                descriptor,
                clear_owner=False,
            )
        removed = remove_owned_directory(
            parent_descriptor,
            rollback_name,
            descriptor,
            owner,
            owner_file=_OWNER_FILE,
        )
        if not removed:
            raise PublicationError(f"Interrupted artifact bundle could not be recovered safely: {final}")
        return True
    finally:
        os.close(descriptor)


@dataclass
class CreateOnlyDirectoryPublication:
    """One bounded transaction that publishes a new immutable directory."""

    final: Path
    stage: Path
    adjacent_stage_name: str
    parent_descriptor: int
    _owner: dict[str, object]
    published_root_mode: int = _DEFAULT_PUBLISHED_ROOT_MODE
    _closed: bool = False
    _published_descriptor: int | None = field(default=None, init=False, repr=False)
    _rollback_name: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def prepare(
        cls,
        bundle_root: str | Path,
        *,
        published_root_mode: int = _DEFAULT_PUBLISHED_ROOT_MODE,
    ) -> CreateOnlyDirectoryPublication:
        """Prepare private staging after validating the publication policy."""

        validated_published_root_mode = _validate_published_root_mode(published_root_mode)
        final = _lexical_absolute_path(Path(bundle_root))
        owner = _owner_payload(final)
        _preflight_existing_path_components(final.parent)
        parent_descriptor = _open_or_create_directory(final.parent)
        try:
            uid = owner["uid"]
            recovery_uid = uid if isinstance(uid, int) else None
            _recover_final_directory(parent_descriptor, final, uid=recovery_uid)
            if _entry_exists_at(parent_descriptor, final.name):
                raise PublicationExistsError(f"Artifact bundle already exists and is immutable: {final}")
            target_digest = str(owner["target_sha256"])
            adjacent_prefix = f".{final.name}.staging-"
            rollback_prefix = f".{final.name}.rollback-"
            _recover_owned_adjacent_directories(
                parent_descriptor,
                final.parent,
                prefix=adjacent_prefix,
                final=final,
                uid=recovery_uid,
                owner_schema=_PUBLICATION_OWNER_SCHEMA,
            )
            _recover_owned_adjacent_directories(
                parent_descriptor,
                final.parent,
                prefix=rollback_prefix,
                final=final,
                uid=recovery_uid,
                owner_schema=(_PUBLICATION_OWNER_SCHEMA, _ROLLBACK_OWNER_SCHEMA),
            )
            private_parent = Path(tempfile.gettempdir()) / f"dnadesign-artifact-publication-{uid}"
            try:
                private_parent.mkdir(mode=0o700)
            except FileExistsError:
                private_stat = private_parent.lstat()
                if (
                    not stat.S_ISDIR(private_stat.st_mode)
                    or (isinstance(uid, int) and private_stat.st_uid != uid)
                    or stat.S_IMODE(private_stat.st_mode) & 0o077
                ):
                    raise PublicationError(f"Private publication staging root is not owner-only: {private_parent}")
            private_prefix = f"stage-{target_digest[:16]}-"
            private_flags = os.O_RDONLY | os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                private_flags |= os.O_NOFOLLOW
            private_descriptor = os.open(private_parent, private_flags)
            try:
                _recover_owned_adjacent_directories(
                    private_descriptor,
                    private_parent,
                    prefix=private_prefix,
                    final=final,
                    uid=recovery_uid,
                    owner_schema=_PUBLICATION_OWNER_SCHEMA,
                )
            finally:
                os.close(private_descriptor)
            stage = Path(tempfile.mkdtemp(prefix=private_prefix, dir=private_parent))
            stage.chmod(_PRIVATE_DIRECTORY_MODE)
            _write_owner(stage / _OWNER_FILE, owner)
            return cls(
                final=final,
                stage=stage,
                adjacent_stage_name=f"{adjacent_prefix}u{uid}-p{os.getpid()}-{uuid.uuid4().hex}",
                parent_descriptor=parent_descriptor,
                _owner=owner,
                published_root_mode=validated_published_root_mode,
            )
        except BaseException:
            os.close(parent_descriptor)
            raise

    def _parent_matches_anchor(self) -> bool:
        try:
            current = os.stat(self.final.parent, follow_symlinks=False)
        except FileNotFoundError:
            return False
        anchored = os.fstat(self.parent_descriptor)
        return stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (
            anchored.st_dev,
            anchored.st_ino,
        )

    def publish(self, *, required_manifest: str) -> None:
        if self._closed:
            raise PublicationError("Artifact publication is already closed")
        if self._published_descriptor is not None:
            raise PublicationError(f"Artifact bundle is already published: {self.final}")
        validated_published_root_mode = _validate_published_root_mode(self.published_root_mode)
        manifest_relative = Path(required_manifest)
        if not required_manifest.strip() or manifest_relative.is_absolute() or ".." in manifest_relative.parts:
            raise PublicationError("Artifact bundle required manifest must stay inside publication staging")
        manifest = self.stage / manifest_relative
        current = self.stage
        manifest_is_safe = True
        for index, part in enumerate(manifest_relative.parts):
            current /= part
            try:
                entry_stat = current.lstat()
            except FileNotFoundError:
                manifest_is_safe = False
                break
            if stat.S_ISLNK(entry_stat.st_mode):
                raise PublicationError("Artifact bundle required manifest must stay inside publication staging")
            if index < len(manifest_relative.parts) - 1 and not stat.S_ISDIR(entry_stat.st_mode):
                manifest_is_safe = False
                break
        if not manifest_is_safe or not manifest.is_file():
            raise PublicationError(f"Artifact bundle staging is incomplete: {manifest}")
        validate_publication_metadata_paths(
            self.stage,
            required_manifest=manifest_relative,
            owner_file=_OWNER_FILE,
        )
        if not self._parent_matches_anchor():
            raise PublicationError(f"Artifact bundle parent changed during publication: {self.final.parent}")
        if _entry_exists_at(self.parent_descriptor, self.adjacent_stage_name):
            raise PublicationError(f"Artifact bundle staging already exists: {self.adjacent_stage_name}")
        renamed = False
        published_descriptor: int | None = None
        try:
            _copy_directory(self.stage, self.parent_descriptor, self.adjacent_stage_name)
            if not self._parent_matches_anchor():
                raise PublicationError(f"Artifact bundle parent changed during publication: {self.final.parent}")
            final_flags = os.O_RDONLY | os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                final_flags |= os.O_NOFOLLOW
            published_descriptor = os.open(
                self.adjacent_stage_name,
                final_flags,
                dir_fd=self.parent_descriptor,
            )
            if not owner_matches_descriptor(
                published_descriptor,
                self._owner,
                owner_file=_OWNER_FILE,
            ):
                raise PublicationError("Artifact bundle publication owner sentinel is unavailable or unsafe")
            _rename_create_only(self.parent_descriptor, self.adjacent_stage_name, self.final.name)
            renamed = True
            if not descriptor_matches_entry(self.parent_descriptor, self.final.name, published_descriptor):
                raise PublicationError("Published artifact bundle identity changed after atomic rename")
            _restore_published_modes(self.stage, published_descriptor)
            os.fchmod(published_descriptor, validated_published_root_mode)
            os.unlink(_OWNER_FILE, dir_fd=published_descriptor)
            self._published_descriptor = published_descriptor
            published_descriptor = None
        except BaseException:
            if renamed and published_descriptor is not None:
                self._published_descriptor = published_descriptor
                published_descriptor = None
                self.rollback()
            elif published_descriptor is not None:
                remove_owned_directory(
                    self.parent_descriptor,
                    self.adjacent_stage_name,
                    published_descriptor,
                    self._owner,
                    owner_file=_OWNER_FILE,
                )
            else:
                remove_owned_named_directory(
                    self.parent_descriptor,
                    self.adjacent_stage_name,
                    self._owner,
                    owner_file=_OWNER_FILE,
                )
            raise
        finally:
            if published_descriptor is not None:
                os.close(published_descriptor)

    def rollback(self) -> bool:
        """Atomically hide this transaction's publication before recursive cleanup."""

        if self._closed:
            raise PublicationError("Artifact publication is already closed")
        published_descriptor = self._published_descriptor
        if published_descriptor is None:
            return False
        rollback_name = self._rollback_name
        if rollback_name is None:
            rollback_name = f".{self.final.name}.rollback-u{self._owner['uid']}-p{os.getpid()}-{uuid.uuid4().hex}"
            self._rollback_name = rollback_name
        rollback_owner = _rollback_owner_payload(self._owner)
        detached = descriptor_matches_entry(
            self.parent_descriptor,
            rollback_name,
            published_descriptor,
        )
        if not detached:
            if not descriptor_matches_entry(
                self.parent_descriptor,
                self.final.name,
                published_descriptor,
            ):
                return False
            _ensure_owner_on_descriptor(
                published_descriptor,
                rollback_owner,
                accepted_existing=self._owner,
            )
            if not descriptor_matches_entry(
                self.parent_descriptor,
                self.final.name,
                published_descriptor,
            ):
                _remove_owner_from_descriptor(published_descriptor)
                return False
            try:
                _rename_create_only(
                    self.parent_descriptor,
                    self.final.name,
                    rollback_name,
                )
            except BaseException:
                if not descriptor_matches_entry(
                    self.parent_descriptor,
                    rollback_name,
                    published_descriptor,
                ):
                    _remove_owner_from_descriptor(published_descriptor)
                raise
            if not descriptor_matches_entry(
                self.parent_descriptor,
                rollback_name,
                published_descriptor,
            ):
                _restore_entry_after_detachment_mismatch(
                    self.parent_descriptor,
                    rollback_name,
                    self.final.name,
                    published_descriptor,
                )
        else:
            _ensure_owner_on_descriptor(
                published_descriptor,
                rollback_owner,
                accepted_existing=self._owner,
            )
        removed = remove_descriptor_anchored_directory(
            self.parent_descriptor,
            rollback_name,
            published_descriptor,
            last_entry=_OWNER_FILE,
        )
        if removed:
            os.close(published_descriptor)
            self._published_descriptor = None
            self._rollback_name = None
        return removed

    def close(self) -> None:
        if self._closed:
            return
        try:
            shutil.rmtree(self.stage, ignore_errors=True)
            remove_owned_named_directory(
                self.parent_descriptor,
                self.adjacent_stage_name,
                self._owner,
                owner_file=_OWNER_FILE,
            )
            if self._published_descriptor is not None and self._rollback_name is not None:
                self.rollback()
        finally:
            if self._published_descriptor is not None:
                os.close(self._published_descriptor)
                self._published_descriptor = None
            self._rollback_name = None
            os.close(self.parent_descriptor)
            self._closed = True

    def __enter__(self) -> CreateOnlyDirectoryPublication:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()
