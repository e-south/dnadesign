"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/ids.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
import hashlib
import json


def variant_id(
    *, job: str, ref: str, protocol: str, sequence: str, modifications: list[str]
) -> str:
    payload = {
        "job": job,
        "ref": ref,
        "protocol": protocol,
        "sequence": sequence,
        "modifications": modifications,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=16).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=").upper()[:12]


def derive_seed64(*, job: str, ref: str, protocol: str, params: dict) -> int:
    # deterministic 64-bit seed from stable knobs
    raw = json.dumps(
        {"job": job, "ref": ref, "protocol": protocol, "params": params},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    dig = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(dig, "big", signed=False)
