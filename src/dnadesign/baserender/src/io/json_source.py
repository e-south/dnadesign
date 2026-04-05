"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/io/json_source.py

Strict JSON and JSONL row readers for contract-first baserender inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..core import SchemaError


def iter_json_rows(path: str | Path) -> Iterable[dict]:
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"JSON input does not exist: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SchemaError(f"Could not parse JSON input: {p}") from exc
    if isinstance(payload, dict):
        yield payload
        return
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise SchemaError(f"JSON array item {index} must be an object")
            yield item
        return
    raise SchemaError("JSON input must contain a single object or an array of objects")


def iter_jsonl_rows(path: str | Path) -> Iterable[dict]:
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"JSONL input does not exist: {p}")
    with p.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise SchemaError(f"Could not parse JSONL line {line_number} in {p}") from exc
            if not isinstance(payload, dict):
                raise SchemaError(f"JSONL line {line_number} must be an object")
            yield payload
