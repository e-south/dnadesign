"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/events/fingerprint.py

Fingerprint helpers for USR event payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pyarrow.parquet as pq

from ..contracts import Fingerprint


def _sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint_parquet(path: Path) -> Fingerprint:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fingerprint path does not exist: {p}")
    pf = pq.ParquetFile(str(p))
    meta = pf.metadata
    sha256 = _sha256_file(p) if os.getenv("USR_EVENT_SHA256") == "1" else None
    return Fingerprint(
        rows=meta.num_rows,
        cols=meta.num_columns,
        size_bytes=int(p.stat().st_size),
        sha256=sha256,
    )
