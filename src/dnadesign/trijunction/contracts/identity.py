"""Canonical serialization and content identities for TriJunction contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one JSON value with stable ordering and no insignificant bytes."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def sha256_bytes(content: bytes) -> str:
    """Return a namespaced SHA-256 content identity."""

    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def mapping_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical content identity for one mapping."""

    return sha256_bytes(canonical_json_bytes(value))
