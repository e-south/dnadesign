"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/api/ids.py

Stable request identifiers for pure Permuter API workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from typing import Iterable


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def request_id(kind: str, ref_name: str, sequence: str, selector: Iterable[object]) -> str:
    return stable_id("permuter-request", kind, ref_name, sequence, ",".join(map(str, selector)))
