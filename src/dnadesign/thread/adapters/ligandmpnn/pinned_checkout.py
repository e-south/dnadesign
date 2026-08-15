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


def working_tree_path_matches_commit(root: Path, commit: str, path: str) -> bool | None:
    """Compare working-tree bytes with the path's blob at the pinned commit."""

    try:
        pinned_bytes = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{commit}:{path}"],
            stderr=subprocess.DEVNULL,
        )
        return (root / path).read_bytes() == pinned_bytes
    except (OSError, subprocess.CalledProcessError):
        return None
