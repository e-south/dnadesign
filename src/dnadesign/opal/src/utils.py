"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/utils.py

Generic utilities: time, hashing, atomic I/O, ids, exit codes.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO8601)


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\-_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "campaign"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_bytes(target: Path, data: bytes) -> None:
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tf:
        tf.write(data)
        tmp = Path(tf.name)
    os.replace(tmp, target)


def atomic_write_text(target: Path, text: str) -> None:
    atomic_write_bytes(target, text.encode("utf-8"))


def write_json(target: Path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(target, json.dumps(obj, indent=indent, sort_keys=True))


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_records_sha256(records_path: Path) -> str:
    # hash the whole parquet file (simple and robust)
    return file_sha256(records_path)


def usr_compatible_id(bio_type: str, sequence: str) -> str:
    s = f"{bio_type}|{sequence.upper()}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def timer() -> Iterable[float]:
    t0 = time.perf_counter()
    yield
    time.perf_counter() - t0
    # consumer measures inside if needed


@dataclass
class ExitCodes:
    SUCCESS = 0
    SUCCESS_WITH_WARNINGS = 2
    CONTRACT_VIOLATION = 3
    NOT_FOUND = 4
    BAD_ARGS = 5
    EXISTS_NEEDS_RESUME = 6
    LOCK_FAILED = 7
    CHECKSUM_MISMATCH = 8
    INTERNAL_ERROR = 9


class OpalError(RuntimeError):
    """Base class for OPAL controlled failures (maps to specific exit codes)."""

    def __init__(self, message: str, code: int = ExitCodes.CONTRACT_VIOLATION):
        super().__init__(message)
        self.exit_code = code


def print_stderr(msg: str) -> None:
    import sys

    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


def print_stdout(msg: str) -> None:
    import sys

    sys.stdout.write(msg.rstrip() + "\n")
    sys.stdout.flush()
