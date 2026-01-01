"""Hashing helpers shared by lockfiles and manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    return sha256_bytes(path.read_bytes())
