"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/evidence_integrity.py

Seal mutable response-window evidence frames between preview and publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import pandas as pd


def evidence_integrity_sha256(
    *,
    scalar_identity: dict[str, object],
    frames: Iterable[tuple[str, pd.DataFrame]],
) -> str:
    """Digest scalar provenance and ordered dataframe values without copying them."""

    digest = hashlib.sha256()
    digest.update(json.dumps(scalar_identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, frame in frames:
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(frame.columns.astype(str).tolist(), separators=(",", ":")).encode("utf-8"))
        digest.update(json.dumps(frame.dtypes.astype(str).tolist(), separators=(",", ":")).encode("utf-8"))
        digest.update(frame.to_json(orient="split", date_format="iso", date_unit="ns", double_precision=15).encode())
    return digest.hexdigest()


__all__ = ["evidence_integrity_sha256"]
