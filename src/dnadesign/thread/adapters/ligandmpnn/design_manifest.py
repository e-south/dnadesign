"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/design_manifest.py

Deterministic, no-follow manifests for published LigandMPNN design trees.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

_COMPLETION_RECORD_NAME = ".dnadesign-ligandmpnn-execution.json"


def build_design_output_manifest(output_root: Path) -> dict[str, object]:
    """Read one design tree through descriptors and bind every public entry."""

    try:
        root_fd = _open_directory_path(output_root)
    except OSError as error:
        raise ValueError(f"design output directory could not be opened safely: {output_root}") from error
    try:
        entries = _manifest_entries(root_fd, relative_parent=PurePosixPath())
    finally:
        os.close(root_fd)
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_id": "thread.ligandmpnn.design_output_manifest",
        "schema_version": 1,
        "artifact_count": sum(entry["type"] == "file" for entry in entries),
        "entry_count": len(entries),
        "tree_sha256": f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        "entries": entries,
    }


def _manifest_entries(directory_fd: int, *, relative_parent: PurePosixPath) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as error:
        raise ValueError("design output directory could not be listed safely") from error
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    for name in names:
        relative_path = relative_parent / name
        if relative_parent == PurePosixPath() and name == _COMPLETION_RECORD_NAME:
            continue
        try:
            status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"design output entry could not be inspected safely: {relative_path}") from error
        if stat.S_ISDIR(status.st_mode):
            try:
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(f"design output directory could not be opened safely: {relative_path}") from error
            try:
                entries.append({"path": relative_path.as_posix(), "type": "directory"})
                entries.extend(_manifest_entries(child_fd, relative_parent=relative_path))
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"design output entry must be regular: {relative_path}")
        try:
            file_fd = os.open(name, file_flags, dir_fd=directory_fd)
        except OSError as error:
            raise ValueError(f"design output artifact could not be opened safely: {relative_path}") from error
        try:
            opened_status = os.fstat(file_fd)
            if not stat.S_ISREG(opened_status.st_mode):
                raise ValueError(f"design output entry must be regular: {relative_path}")
            digest = hashlib.sha256()
            size_bytes = 0
            while payload := os.read(file_fd, 1024 * 1024):
                digest.update(payload)
                size_bytes += len(payload)
        except OSError as error:
            raise ValueError(f"design output artifact could not be read safely: {relative_path}") from error
        finally:
            os.close(file_fd)
        entries.append(
            {
                "path": relative_path.as_posix(),
                "type": "file",
                "size_bytes": size_bytes,
                "sha256": f"sha256:{digest.hexdigest()}",
            }
        )
    return entries


def _open_directory_path(path: Path) -> int:
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
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


__all__ = ["build_design_output_manifest"]
