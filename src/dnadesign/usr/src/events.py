"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/events.py

Structured JSONL event logging for USR dataset mutations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from .io import now_utc
from .types import Fingerprint
from .version import __version__


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


def record_event(
    event_path: Path,
    action: str,
    *,
    dataset: str,
    args: Optional[dict] = None,
    target_path: Optional[Path] = None,
) -> None:
    if target_path is None:
        raise ValueError("target_path is required for event fingerprinting.")
    payload = {
        "timestamp_utc": now_utc(),
        "action": str(action),
        "dataset": str(dataset),
        "args": args or {},
        "fingerprint": fingerprint_parquet(target_path).to_dict(),
        "version": __version__,
    }
    event_path = Path(event_path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")
