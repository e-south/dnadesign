"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np


class ExitCodes:
    OK = 0
    BAD_ARGS = 2
    CONTRACT_VIOLATION = 3
    NOT_FOUND = 4
    INTERNAL_ERROR = 5


class OpalError(Exception):
    def __init__(self, message: str, exit_code: int | None = None):
        super().__init__(message)
        self.exit_code = ExitCodes.BAD_ARGS if exit_code is None else exit_code


def print_stdout(msg: str) -> None:
    print(msg)


def print_stderr(msg: str) -> None:
    import sys

    print(msg, file=sys.stderr)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    Path(path).write_text(json.dumps(data, indent=2))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def robust_center_scale(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(Y, axis=0)
    q75 = np.nanpercentile(Y, 75, axis=0)
    q25 = np.nanpercentile(Y, 25, axis=0)
    iqr = q75 - q25
    scale = np.where(iqr < 1e-12, 1.0, iqr / 1.349)
    return med, scale


def competition_rank(scores_desc_sorted: np.ndarray) -> np.ndarray:
    """Competition ranking (1,2,3,3,5) given scores sorted descending."""
    n = len(scores_desc_sorted)
    ranks = np.empty(n, dtype=int)
    if n == 0:
        return ranks
    rank = 1
    i = 0
    while i < n:
        j = i
        while j < n and scores_desc_sorted[j] == scores_desc_sorted[i]:
            j += 1
        ranks[i:j] = rank
        rank = j + 1
        i = j
    return ranks
