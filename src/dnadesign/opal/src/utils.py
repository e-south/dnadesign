"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/utils.py

General utilities: hashing, I/O, logging, QC flag helpers, errors & exits.
This module intentionally has no dependency on campaign-local code.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import dataclasses as _dc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .cli.pretty import console_err, console_out


# -----------------------
# Errors & Exit codes
# -----------------------
class ExitCodes:
    OK = 0
    BAD_ARGS = 2
    INTERNAL_ERROR = 3
    CONTRACT_VIOLATION = 4


class OpalError(RuntimeError):
    def __init__(self, message: str, exit_code: int = ExitCodes.BAD_ARGS):
        super().__init__(message)
        self.exit_code = exit_code


class ConfigError(OpalError):
    pass


class DataError(OpalError):
    pass


class LedgerError(OpalError):
    pass


class RunError(OpalError):
    pass


# -----------------------
# IO helpers
# -----------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    # Always UTC ISO without microseconds for stable run_id
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def print_stdout(s: str) -> None:
    c = console_out()
    if c is not None:
        # Rich console prints with markup if enabled
        c.print(s)
    else:
        print(s)


def print_stderr(s: str) -> None:
    c = console_err()
    if c is not None:
        c.print(s)
    else:
        print(s, file=os.sys.stderr)


def read_json(path: str | Path):
    """Read JSON into a Python object."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj, *, indent: int = 2) -> None:
    """Atomically write JSON to disk (ensures parent dir)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    tmp.replace(p)


# -----------------------
# Hashing params / JSON
# -----------------------
def _jsonable(obj: Any) -> Any:
    if _dc.is_dataclass(obj):
        return _dc.asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # numpy
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def params_hash(obj: Any) -> str:
    s = json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -----------------------
# QC flags
# -----------------------
@dataclass(frozen=True)
class QCFlag:
    code: str
    where: str  # data|pred|obj|sel|unc
    severity: str  # info|warn|error
    message: Optional[str] = None
    data: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"code": self.code, "where": self.where, "severity": self.severity}
        if self.message:
            d["message"] = self.message
        if self.data:
            d["data"] = {str(k): str(v) for k, v in self.data.items()}
        return d


def qc_nonfinite_pred(vec: Iterable[float]) -> Optional[QCFlag]:
    arr = np.asarray(list(vec), dtype=float)
    if not np.all(np.isfinite(arr)):
        return QCFlag(
            code="pred_nonfinite",
            where="pred",
            severity="error",
            message="Non-finite value in pred__y_hat_model.",
        )
    return None


def qc_dim_mismatch(vec: Iterable[float], y_dim: int) -> Optional[QCFlag]:
    arr = np.asarray(list(vec), dtype=float).ravel()
    if arr.size != y_dim:
        return QCFlag(
            code="pred_dim_mismatch",
            where="pred",
            severity="error",
            data={"expected": str(y_dim), "got": str(arr.size)},
        )
    return None


def qc_obj_nan_scalar(x: float) -> Optional[QCFlag]:
    if not np.isfinite(float(x)):
        return QCFlag(
            code="obj_nan_scalar",
            where="obj",
            severity="error",
            message="Objective scalar is not finite.",
        )
    return None


def qc_obj_clipped(kind: str) -> QCFlag:
    # kind: "high" | "low"
    return QCFlag(
        code=f"obj_clipped_{kind}",
        where="obj",
        severity="warn",
    )


def qc_sel_tie_comp() -> QCFlag:
    return QCFlag(code="sel_tie_competition", where="sel", severity="info")


def qc_sel_threshold_clip() -> QCFlag:
    return QCFlag(code="sel_threshold_clip", where="sel", severity="info")


# -----------------------
# Small helpers
# -----------------------
def coerce_list_floats(v: Any) -> List[float]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return [float(x) for x in np.asarray(v, dtype=float).ravel().tolist()]
    # strings like "[1,2]" are explicitly not supported here
    raise ValueError("Expected a sequence to coerce to list[float].")
