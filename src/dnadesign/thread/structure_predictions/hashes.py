"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_predictions/hashes.py

Hash helpers for structure-prediction artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def text_sha256_uri(value: str) -> str:
    """Return a stable SHA-256 URI for text payloads."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256_uri(path: Path) -> str:
    """Return a stable SHA-256 URI for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
