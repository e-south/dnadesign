"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_io.py

Shared JSON/JSONL helpers for published YIU view contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_text


def write_json_payload(path: Path, payload: object) -> Path:
    atomic_write_json(path, payload, indent=2, sort_keys=False, allow_nan=False)
    return path


def write_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    text = "".join(f"{json.dumps(row, allow_nan=False)}\n" for row in rows)
    atomic_write_text(path, text)
    return path


def load_contract_rows(contract_path: Path, *, input_kind: str) -> list[dict[str, Any]]:
    if input_kind == "jsonl":
        return [json.loads(line) for line in contract_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if input_kind != "json":
        raise ValueError(f"unsupported YIU view input kind: {input_kind!r}")

    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not all(isinstance(item, Mapping) for item in payload):
            raise ValueError(f"render input must decode to mappings only: {contract_path}")
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping):
        return [dict(payload)]
    raise ValueError(f"render input must decode to a mapping or list: {contract_path}")


__all__ = [
    "load_contract_rows",
    "write_json_payload",
    "write_jsonl_rows",
]
