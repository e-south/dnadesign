"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/hashes.py

Stable hashes for Biohub ESMC request and response contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def biohub_request_hash(payload: Mapping[str, Any]) -> str:
    """Hash one redacted Biohub ESMC materialization request."""

    return _sha256_json(payload)


def biohub_query_hash(payload: Mapping[str, Any]) -> str:
    """Hash one per-sequence Biohub ESMC query."""

    return _sha256_json(payload)


def raw_response_hash(payload: Mapping[str, Any]) -> str:
    """Hash a Biohub JSON response without writing the response to disk."""

    return _sha256_json(payload)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
