"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/pinned_checkout.py

Content identity checks for files in a pinned LigandMPNN checkout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def attested_working_tree_path_bytes(root: Path, commit: str, path: str) -> bytes | None:
    """Return working-tree bytes only when they match the pinned commit blob."""

    pinned_bytes = _pinned_path_bytes(root, commit, path)
    working_tree_bytes = _working_tree_path_bytes(root, path)
    if pinned_bytes is None or working_tree_bytes is None:
        return None
    return working_tree_bytes if working_tree_bytes == pinned_bytes else None


def working_tree_path_matches_commit(root: Path, commit: str, path: str) -> bool | None:
    """Report whether working-tree bytes match the path's pinned commit blob."""

    pinned_bytes = _pinned_path_bytes(root, commit, path)
    working_tree_bytes = _working_tree_path_bytes(root, path)
    if pinned_bytes is None or working_tree_bytes is None:
        return None
    return working_tree_bytes == pinned_bytes


def _pinned_path_bytes(root: Path, commit: str, path: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "--no-replace-objects", "-C", str(root), "show", f"{commit}:{path}"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _working_tree_path_bytes(root: Path, path: str) -> bytes | None:
    try:
        return (root / path).read_bytes()
    except OSError:
        return None
