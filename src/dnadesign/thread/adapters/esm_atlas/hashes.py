"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/esm_atlas/hashes.py

Hash helpers for ESM Atlas semantic-audit artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def sequence_md5(sequence: str) -> str:
    """Return Atlas' MD5 hash for an amino-acid sequence."""

    normalized = "".join(str(sequence).split()).upper()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def raw_response_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable hash URI for one decoded API response."""

    return _stable_hash(payload)


def atlas_request_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable request hash URI for one Atlas materialization plan."""

    return _stable_hash(payload)


def atlas_query_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable hash URI for one per-sequence Atlas query."""

    return _stable_hash(payload)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
