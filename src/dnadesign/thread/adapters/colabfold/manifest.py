"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/colabfold/manifest.py

ColabFold runtime manifest helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def runtime_parameters_hash(runtime_parameters: Mapping[str, Any]) -> str:
    """Return a stable hash URI for declared ColabFold runtime parameters."""

    encoded = json.dumps(dict(runtime_parameters), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256_uri(path: Path) -> str:
    """Return a streaming SHA-256 URI for one ColabFold lineage file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def ordered_positions_hash(positions: Sequence[int]) -> str:
    """Return the stable SHA-256 URI for an ordered one-based position correspondence."""

    payload = ",".join(str(position) for position in positions).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
