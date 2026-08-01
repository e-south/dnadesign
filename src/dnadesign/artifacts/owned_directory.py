"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/artifacts/owned_directory.py

Descriptor-anchored validation and cleanup for owned publication directories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import stat


def descriptor_matches_entry(parent_descriptor: int, name: str, descriptor: int) -> bool:
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    anchored = os.fstat(descriptor)
    return stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (
        anchored.st_dev,
        anchored.st_ino,
    )


def read_owner_from_descriptor(
    descriptor: int,
    *,
    owner_file: str,
) -> dict[str, object] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        owner_descriptor = os.open(owner_file, flags, dir_fd=descriptor)
    except OSError:
        return None
    try:
        owner_stat = os.fstat(owner_descriptor)
        if not stat.S_ISREG(owner_stat.st_mode):
            return None
        with os.fdopen(os.dup(owner_descriptor), "r", encoding="utf-8") as handle:
            observed = json.load(handle)
        return observed if isinstance(observed, dict) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    finally:
        os.close(owner_descriptor)


def owner_matches_descriptor(
    descriptor: int,
    expected: dict[str, object],
    *,
    owner_file: str,
) -> bool:
    return read_owner_from_descriptor(descriptor, owner_file=owner_file) == expected


def _remove_contents(descriptor: int) -> None:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    for name in os.listdir(descriptor):
        entry_stat = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(entry_stat.st_mode):
            child_descriptor = os.open(name, directory_flags, dir_fd=descriptor)
            try:
                _remove_contents(child_descriptor)
            finally:
                os.close(child_descriptor)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def remove_owned_directory(
    parent_descriptor: int,
    name: str,
    owned_descriptor: int,
    expected_owner: dict[str, object],
    *,
    owner_file: str,
) -> bool:
    """Remove only the still-named directory proven to be this transaction's."""

    if not descriptor_matches_entry(parent_descriptor, name, owned_descriptor):
        return False
    if not owner_matches_descriptor(owned_descriptor, expected_owner, owner_file=owner_file):
        return False
    return remove_descriptor_anchored_directory(parent_descriptor, name, owned_descriptor)


def remove_descriptor_anchored_directory(
    parent_descriptor: int,
    name: str,
    owned_descriptor: int,
) -> bool:
    """Remove only the directory entry still matching an open descriptor."""

    if not descriptor_matches_entry(parent_descriptor, name, owned_descriptor):
        return False
    _remove_contents(owned_descriptor)
    if not descriptor_matches_entry(parent_descriptor, name, owned_descriptor):
        return False
    os.rmdir(name, dir_fd=parent_descriptor)
    return True


def remove_owned_named_directory(
    parent_descriptor: int,
    name: str,
    expected_owner: dict[str, object],
    *,
    owner_file: str,
) -> bool:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError:
        return False
    try:
        return remove_owned_directory(
            parent_descriptor,
            name,
            descriptor,
            expected_owner,
            owner_file=owner_file,
        )
    finally:
        os.close(descriptor)


__all__ = [
    "descriptor_matches_entry",
    "owner_matches_descriptor",
    "read_owner_from_descriptor",
    "remove_descriptor_anchored_directory",
    "remove_owned_directory",
    "remove_owned_named_directory",
]
