"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/artifacts/atomic_write.py

Atomic write helpers for artifacts and metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _atomic_tmp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    return Path(tmp_name)


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = _atomic_tmp_path(path)
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    allow_nan: bool = True,
) -> None:
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys, allow_nan=allow_nan)
    atomic_write_text(path, text)


def atomic_write_yaml(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = False,
    default_flow_style: bool = False,
) -> None:
    text = yaml.safe_dump(payload, sort_keys=sort_keys, default_flow_style=default_flow_style)
    atomic_write_text(path, text)


def atomic_write_parquet(df, path: Path, *, engine: str = "pyarrow", index: bool = False) -> None:
    tmp_path = _atomic_tmp_path(path)
    try:
        df.to_parquet(tmp_path, engine=engine, index=index)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
