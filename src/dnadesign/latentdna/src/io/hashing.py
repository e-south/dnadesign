"""
Hashing helpers for latentdna.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


def sha256_file(path: Path) -> str:
    candidate = path.resolve()
    stat = candidate.stat()
    return _sha256_file_for_stat(candidate.as_posix(), stat.st_mtime_ns, stat.st_size, stat.st_ino)


@lru_cache(maxsize=512)
def _sha256_file_for_stat(path: str, mtime_ns: int, size: int, inode: int) -> str:
    del mtime_ns, size, inode
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def sha256_path(path: Path) -> str:
    candidate = path.resolve()
    if candidate.is_file():
        return sha256_file(candidate)
    if candidate.is_dir():
        digest = hashlib.sha256()
        for child in sorted(entry for entry in candidate.rglob("*") if entry.is_file()):
            relative = child.relative_to(candidate).as_posix().encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(sha256_file(child).encode("utf-8"))
            digest.update(b"\0")
        return f"sha256:{digest.hexdigest()}"
    raise FileNotFoundError(f"path not found for hashing: {candidate}")
