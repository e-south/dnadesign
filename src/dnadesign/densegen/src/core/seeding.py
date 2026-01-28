"""
Seed derivation helpers for reproducible runs.
"""

from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, label: str) -> int:
    """
    Derive a stable integer seed from a base seed and label.

    The result is deterministic across platforms and Python versions.
    """
    payload = f"{int(base_seed)}:{label}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def derive_seed_map(base_seed: int, labels: list[str]) -> dict[str, int]:
    return {label: derive_seed(base_seed, label) for label in labels}
