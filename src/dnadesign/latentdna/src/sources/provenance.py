"""
Source provenance digest helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dnadesign.usr.src.overlays import overlay_parts

from ..io.hashing import sha256_path, sha256_payload

OVERLAY_INVENTORY_DIGEST_MODE = "inventory"


def overlay_inventory_paths(path: Path) -> list[str]:
    candidate = path.resolve()
    if candidate.is_file():
        return [candidate.as_posix()]
    if candidate.is_dir():
        return [part.resolve().as_posix() for part in overlay_parts(candidate)]
    raise FileNotFoundError(f"path not found for overlay inventory: {candidate}")


def overlay_inventory_digest(path: Path) -> str:
    candidate = path.resolve()
    if candidate.is_file():
        inventory = [candidate.name]
    elif candidate.is_dir():
        inventory = [part.name for part in overlay_parts(candidate)]
    else:
        raise FileNotFoundError(f"path not found for overlay inventory: {candidate}")
    return sha256_payload(
        {
            "path": candidate.as_posix(),
            "parts": inventory,
        }
    )


def source_provenance_digest(entry: Mapping[str, object]) -> str:
    path = Path(str(entry["path"]))
    digest_mode = str(entry.get("digest_mode") or "")
    if digest_mode == OVERLAY_INVENTORY_DIGEST_MODE:
        return overlay_inventory_digest(path)
    return sha256_path(path)
