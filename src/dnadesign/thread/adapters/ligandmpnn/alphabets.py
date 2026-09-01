"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/alphabets.py

Deterministic residue-alphabet sidecars for LigandMPNN.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnRequest


class LigandMpnnSidecarPublicationUncertainError(RuntimeError):
    """Raised when sidecar rollback cannot be confirmed durable."""


@dataclass(frozen=True)
class LigandMpnnResidueAlphabetSidecar:
    """Digest-bound receipt for one materialized omission JSON sidecar."""

    request_id: str
    path: Path
    sha256: str
    residue_count: int
    materialized_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.sha256.startswith("sha256:") or len(self.sha256) != 71:
            raise ValueError("sha256 must be a prefixed SHA256 digest")
        _require_json_path(self.path, field_name="path")
        if self.materialized_path is not None:
            _require_json_path(self.materialized_path, field_name="materialized_path")
        if isinstance(self.residue_count, bool) or self.residue_count <= 0:
            raise ValueError("residue_count must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": "thread.ligandmpnn.residue_alphabet_sidecar",
            "schema_version": 1,
            "request_id": self.request_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "residue_count": self.residue_count,
        }

    def validate_for(self, request: LigandMpnnRequest, *, execution_root: Path | None = None) -> None:
        """Fail if this receipt is not the exact sidecar for ``request``."""

        self._validate_request_binding(request)
        materialized_path = self._validation_path(execution_root=execution_root)
        materialized_bytes = _read_regular_file_bytes(materialized_path, label="residue alphabet sidecar")
        if materialized_bytes is None or _digest(materialized_bytes) != self.sha256:
            raise ValueError("residue alphabet sidecar file SHA256 does not match receipt")

    def validate_execution_file(self, request: LigandMpnnRequest, *, execution_root: Path) -> None:
        """Validate the final command-bound file after staging is promoted."""

        self._validate_request_binding(request)
        execution_path = _anchor_relative_path(self.path, execution_root=execution_root)
        execution_bytes = _read_regular_file_bytes(execution_path, label="execution residue alphabet sidecar")
        if execution_bytes is None or _digest(execution_bytes) != self.sha256:
            raise ValueError("execution residue alphabet sidecar SHA256 does not match receipt")

    def _validation_path(self, *, execution_root: Path | None) -> Path:
        if self.materialized_path is not None:
            return self.materialized_path
        return _anchor_relative_path(self.path, execution_root=execution_root)

    def _validate_request_binding(self, request: LigandMpnnRequest) -> None:
        if self.request_id != request.request_id:
            raise ValueError("residue alphabet sidecar request_id does not match request")
        expected = _canonical_bytes(request)
        if self.residue_count != len(request.residue_alphabets):
            raise ValueError("residue alphabet sidecar residue_count does not match request")
        if self.sha256 != _digest(expected):
            raise ValueError("residue alphabet sidecar digest does not match request")


def materialize_residue_alphabet_sidecar(
    request: LigandMpnnRequest,
    path: Path,
    *,
    write_path: Path | None = None,
) -> LigandMpnnResidueAlphabetSidecar:
    """Materialize omission JSON while binding the eventual execution path."""

    if not request.residue_alphabets:
        raise ValueError("request has no residue alphabets to materialize")
    _require_json_path(path, field_name="path")
    target = write_path or path
    _require_json_path(target, field_name="write_path")
    content = _canonical_bytes(request)
    _write_new_regular_file_or_validate_existing(target, content)
    return LigandMpnnResidueAlphabetSidecar(
        request_id=request.request_id,
        path=path,
        sha256=_digest(content),
        residue_count=len(request.residue_alphabets),
        materialized_path=target,
    )


def _canonical_bytes(request: LigandMpnnRequest) -> bytes:
    payload = {
        alphabet.residue.upstream_id: alphabet.omitted_amino_acids
        for alphabet in sorted(request.residue_alphabets, key=lambda item: item.residue.upstream_id)
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _require_json_path(path: object, *, field_name: str) -> None:
    if not isinstance(path, Path) or path.suffix.lower() != ".json":
        raise ValueError(f"{field_name} must be a Path ending in .json")
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain traversal")
    if str(path).startswith("~"):
        raise ValueError(f"{field_name} must not begin with '~'")
    if str(path).startswith("-"):
        raise ValueError(f"{field_name} must not begin with '-'")


def _anchor_relative_path(path: Path, *, execution_root: Path | None) -> Path:
    if path.is_absolute() or execution_root is None:
        return path
    if not isinstance(execution_root, Path) or not execution_root.is_absolute():
        raise ValueError("execution_root must be an absolute Path")
    return execution_root / path


def _write_new_regular_file_or_validate_existing(target: Path, content: bytes) -> None:
    try:
        directory_fd = _open_directory_path(target.parent, create=True)
    except OSError as error:
        raise ValueError("sidecar target directory could not be opened safely") from error
    try:
        _publish_complete_file(directory_fd, target, content)
    finally:
        os.close(directory_fd)


def _publish_complete_file(directory_fd: int, target: Path, content: bytes) -> None:
    temporary_name = f".{target.name}.{uuid.uuid4().hex}.tmp"
    temporary_present = False
    try:
        published_identity = _write_private_file(directory_fd, temporary_name, content)
        temporary_present = True
        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            os.unlink(temporary_name, dir_fd=directory_fd)
            temporary_present = False
            _validate_matching_existing_file(directory_fd, target, content)
            return
        except OSError as error:
            raise ValueError("sidecar target could not be published safely") from error
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
            temporary_present = False
            os.fsync(directory_fd)
        except OSError as publication_error:
            try:
                _rollback_publication(
                    directory_fd,
                    target.name,
                    temporary_name,
                    temporary_present=temporary_present,
                    published_identity=published_identity,
                )
                temporary_present = False
            except OSError as rollback_error:
                raise LigandMpnnSidecarPublicationUncertainError(
                    "LigandMPNN sidecar publication rollback durability is uncertain"
                ) from rollback_error
            raise ValueError("sidecar publication could not be made durable") from publication_error
    finally:
        if temporary_present:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def _write_private_file(directory_fd: int, temporary_name: str, content: bytes) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError("sidecar private file could not be created safely") from error
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("sidecar write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        return status.st_dev, status.st_ino
    except OSError as error:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise ValueError("sidecar could not be written completely") from error
    finally:
        os.close(descriptor)


def _rollback_publication(
    directory_fd: int,
    target_name: str,
    temporary_name: str,
    *,
    temporary_present: bool,
    published_identity: tuple[int, int],
) -> None:
    try:
        target_status = os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        target_status = None
    if target_status is not None:
        if (target_status.st_dev, target_status.st_ino) != published_identity:
            raise OSError("sidecar target changed before rollback")
        os.unlink(target_name, dir_fd=directory_fd)
    if temporary_present:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
    os.fsync(directory_fd)


def _validate_matching_existing_file(directory_fd: int, target: Path, content: bytes) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(target.name, flags, dir_fd=directory_fd)
    except FileNotFoundError as error:
        raise ValueError("sidecar target changed during materialization") from error
    except OSError as error:
        raise ValueError("sidecar target must be a regular file") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("sidecar target must be a regular file")
        existing = _read_descriptor_bytes(descriptor)
        if existing != content:
            raise FileExistsError(f"refusing to overwrite different residue alphabet sidecar: {target}")
        try:
            os.fsync(descriptor)
        except OSError as error:
            raise ValueError("matching sidecar could not be made durable") from error
    finally:
        os.close(descriptor)
    try:
        os.fsync(directory_fd)
    except OSError as error:
        raise ValueError("matching sidecar could not be made durable") from error


def _read_regular_file_bytes(path: Path, *, label: str) -> bytes | None:
    try:
        directory_fd = _open_directory_path(path.parent, create=False)
    except OSError as error:
        raise ValueError(f"{label} directory could not be opened safely") from error
    try:
        return _read_regular_file_from_directory(directory_fd, path.name, label=label)
    finally:
        os.close(directory_fd)


def _read_regular_file_from_directory(directory_fd: int, name: str, *, label: str) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"{label} must be a regular file") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file")
        return _read_descriptor_bytes(descriptor)
    finally:
        os.close(descriptor)


def _read_descriptor_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _open_directory_path(path: Path, *, create: bool) -> int:
    """Open one directory through an entirely no-follow descriptor chain."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    if path.is_absolute():
        current_fd = os.open(path.anchor, directory_flags)
        components = path.parts[1:]
    else:
        current_fd = os.open(".", directory_flags)
        components = path.parts
    try:
        for component in components:
            if component in {"", "."}:
                continue
            if component == "..":
                raise OSError("directory traversal is not allowed")
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                except FileExistsError:
                    pass
                else:
                    os.fsync(current_fd)
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise
