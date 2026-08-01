"""
--------------------------------------------------------------------------------
dnadesign
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


def _text(path: Path, *, content: bytes | None, kind: str) -> str:
    if content is None:
        if not path.exists():
            raise SchemaError(f"{kind} input does not exist: {path}")
        return path.read_text(encoding="utf-8")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"Could not decode {kind} input as UTF-8: {path}") from exc


def iter_json_rows(path: str | Path, *, content: bytes | None = None) -> Iterable[dict]:
    p = Path(path)
    try:
        payload = json.loads(_text(p, content=content, kind="JSON"))
    except SchemaError:
        raise
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


def iter_jsonl_rows(path: str | Path, *, content: bytes | None = None) -> Iterable[dict]:
    p = Path(path)
    for line_number, raw_line in enumerate(_text(p, content=content, kind="JSONL").splitlines(), start=1):
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
