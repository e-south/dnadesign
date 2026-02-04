"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/runs/signatures.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


def _sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def hash_signature(obj: dict) -> str:
    return _sha1_str(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def ids_hash(ids: list[str]) -> str:
    # stable hash of sorted ids
    return _sha1_str("|".join(sorted(map(str, ids))))


def file_fingerprint(p: Path) -> dict:
    st = p.stat()
    return {"mtime": int(st.st_mtime), "size": int(st.st_size)}


@dataclass
class InputSignature:
    source_kind: Literal["usr", "parquet", "csv"]
    source_ref: str
    key_col: str
    row_ids_hash: str
    x_spec: dict
    fingerprint: dict

    def dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hash_signature(self.dict())


@dataclass
class AlgoSignature:
    algo: Literal["leiden"]
    params: dict
    libs: dict

    def dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hash_signature(self.dict())


@dataclass
class UmapSignature:
    params: dict
    libs: dict

    def dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hash_signature(self.dict())
