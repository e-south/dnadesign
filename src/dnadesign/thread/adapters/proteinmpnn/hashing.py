"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/hashing.py

Hash helpers for ProteinMPNN request manifests and sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the hex SHA256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_uri(path: Path) -> str:
    """Return the repository manifest hash string for one file."""

    return "sha256:" + sha256_file(path)
