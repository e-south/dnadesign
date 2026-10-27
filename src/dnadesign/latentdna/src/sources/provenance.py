"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/sources/provenance.py

Source provenance digest helpers for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dnadesign.usr import (
    OVERLAY_DIGEST_LEDGER_FILENAME,
    OVERLAY_PART_PREFIX,
    build_overlay_digest_ledger,
)

from ..io.hashing import sha256_path, sha256_payload

OVERLAY_INVENTORY_DIGEST_MODE = "inventory"
OVERLAY_LEDGER_PAYLOAD_DIGEST_MODE = "overlay_ledger_payload"


def overlay_inventory_paths(path: Path) -> list[str]:
    candidate = path.resolve()
    if candidate.is_file():
        return [candidate.as_posix()]
    if candidate.is_dir():
        return [part.resolve().as_posix() for part in _uncached_overlay_parts(candidate)]
    raise FileNotFoundError(f"path not found for overlay inventory: {candidate}")


def overlay_inventory_digest(path: Path) -> str:
    candidate = path.resolve()
    if candidate.is_file():
        inventory = [candidate.name]
    elif candidate.is_dir():
        inventory = [part.name for part in _uncached_overlay_parts(candidate)]
    else:
        raise FileNotFoundError(f"path not found for overlay inventory: {candidate}")
    return sha256_payload(
        {
            "path": candidate.as_posix(),
            "parts": inventory,
        }
    )


def _uncached_overlay_parts(path: Path) -> list[Path]:
    return sorted(path.glob(f"{OVERLAY_PART_PREFIX}*.parquet"))


def overlay_ledger_payload_digest(path: Path) -> str:
    candidate = path.resolve()
    if candidate.is_file():
        if candidate.name != OVERLAY_DIGEST_LEDGER_FILENAME:
            raise FileNotFoundError(f"path is not an overlay digest ledger: {candidate}")
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"path not found for overlay ledger payload: {candidate}")
    return sha256_payload(build_overlay_digest_ledger(candidate))


def source_provenance_digest(entry: Mapping[str, object]) -> str:
    path = Path(str(entry["path"]))
    digest_mode = str(entry.get("digest_mode") or "")
    if digest_mode == OVERLAY_INVENTORY_DIGEST_MODE:
        return overlay_inventory_digest(path)
    if digest_mode == OVERLAY_LEDGER_PAYLOAD_DIGEST_MODE:
        return overlay_ledger_payload_digest(path)
    return sha256_path(path)
