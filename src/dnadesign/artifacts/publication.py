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
import hashlib
import json
import os
import shutil
import socket
import stat
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .errors import PublicationError
from .owned_directory import (
    descriptor_matches_entry,
    owner_matches_descriptor,
    read_owner_from_descriptor,
    remove_owned_directory,
    remove_owned_named_directory,
)
from .portable_paths import validate_publication_metadata_paths

_OWNER_FILE = ".dnadesign-publication-owner.json"
_MAX_STALE_CANDIDATES = 64
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_FINAL_ROOT_MODE = 0o755


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
        raise PublicationError(f"Artifact bundle already exists and is immutable: {destination}")
    raise OSError(error, os.strerror(error), destination)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _owner_payload(final: Path) -> dict[str, object]:
    return {
        "schema": "dnadesign.artifact_publication_owner.v1",
        "target_sha256": hashlib.sha256(os.fsencode(final)).hexdigest(),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_unix": time.time(),
    }


def _write_owner(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(_PRIVATE_FILE_MODE)


def _is_recoverable_stale_stage(path: Path, *, final: Path, uid: int | None) -> bool:
    try:
        entry_stat = path.lstat()
        if not stat.S_ISDIR(entry_stat.st_mode) or (uid is not None and entry_stat.st_uid != uid):
            return False
        owner_path = path / _OWNER_FILE
        owner_stat = owner_path.lstat()
        if not stat.S_ISREG(owner_stat.st_mode):
            return False
        payload = json.loads(owner_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and _owner_payload_is_recoverable(payload, final=final, uid=uid)


def _owner_payload_is_recoverable(
    payload: dict[str, object],
    *,
    final: Path,
    uid: int | None,
) -> bool:
    try:
        owner_pid = int(payload.get("pid", -1))
    except (TypeError, ValueError):
        return False
    return (
        payload.get("schema") == "dnadesign.artifact_publication_owner.v1"
        and payload.get("target_sha256") == hashlib.sha256(os.fsencode(final)).hexdigest()
        and payload.get("uid") == uid
        and payload.get("host") == socket.gethostname()
        and not _pid_is_alive(owner_pid)
    )


def _remove_recoverable_stale_stage(
    parent_descriptor: int,
    name: str,
    *,
    final: Path,
    uid: int | None,
) -> bool:
    """Remove a stale stage only after descriptor-anchored owner revalidation."""

    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError:
        return False
    try:
        entry_stat = os.fstat(descriptor)
        if uid is not None and entry_stat.st_uid != uid:
            return False
        observed_owner = read_owner_from_descriptor(descriptor, owner_file=_OWNER_FILE)
        if observed_owner is None or not _owner_payload_is_recoverable(observed_owner, final=final, uid=uid):
            return False
        return remove_owned_directory(
            parent_descriptor,
            name,
            descriptor,
            observed_owner,
            owner_file=_OWNER_FILE,
        )
    finally:
        os.close(descriptor)


def _bounded_named_candidates(directory: Path, *, prefix: str) -> list[Path]:
    candidates: list[Path] = []
    for candidate in directory.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        candidates.append(candidate)
        if len(candidates) >= _MAX_STALE_CANDIDATES:
            break
    return candidates


def _is_recoverable_adjacent_stage(path: Path, *, final: Path, uid: int | None) -> bool:
    return _is_recoverable_stale_stage(path, final=final, uid=uid)


@dataclass
class CreateOnlyDirectoryPublication:
    """One bounded transaction that publishes a new immutable directory."""

    final: Path
    stage: Path
    adjacent_stage_name: str
    parent_descriptor: int
    _owner: dict[str, object]
    _closed: bool = False

    @classmethod
    def prepare(cls, bundle_root: str | Path) -> CreateOnlyDirectoryPublication:
        final = _lexical_absolute_path(Path(bundle_root))
        _preflight_existing_path_components(final.parent)
        parent_descriptor = _open_or_create_directory(final.parent)
        try:
            if _entry_exists_at(parent_descriptor, final.name):
                raise PublicationError(f"Artifact bundle already exists and is immutable: {final}")
            owner = _owner_payload(final)
            uid = owner["uid"]
            target_digest = str(owner["target_sha256"])
            adjacent_prefix = f".{final.name}.staging-"
            for candidate in _bounded_named_candidates(final.parent, prefix=adjacent_prefix):
                if _is_recoverable_adjacent_stage(
                    candidate,
                    final=final,
                    uid=uid if isinstance(uid, int) else None,
                ):
                    _remove_recoverable_stale_stage(
                        parent_descriptor,
                        candidate.name,
                        final=final,
                        uid=uid if isinstance(uid, int) else None,
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
                for candidate in _bounded_named_candidates(private_parent, prefix=private_prefix):
                    if _is_recoverable_stale_stage(
                        candidate,
                        final=final,
                        uid=uid if isinstance(uid, int) else None,
                    ):
                        _remove_recoverable_stale_stage(
                            private_descriptor,
                            candidate.name,
                            final=final,
                            uid=uid if isinstance(uid, int) else None,
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
            )
        except Exception:
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
            os.fchmod(published_descriptor, _FINAL_ROOT_MODE)
            os.unlink(_OWNER_FILE, dir_fd=published_descriptor)
        except Exception:
            if renamed and published_descriptor is not None:
                remove_owned_directory(
                    self.parent_descriptor,
                    self.final.name,
                    published_descriptor,
                    self._owner,
                    owner_file=_OWNER_FILE,
                )
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
        finally:
            os.close(self.parent_descriptor)
            self._closed = True

    def __enter__(self) -> CreateOnlyDirectoryPublication:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()
