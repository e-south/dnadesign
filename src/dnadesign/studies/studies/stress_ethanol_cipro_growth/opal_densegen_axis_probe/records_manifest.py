"""Scratch records manifest checks for the DenseGen axis probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def records_manifest_path(records_path: Path) -> Path:
    return records_path.parent / "records_manifest.json"


def records_manifest_payload(src: Path, dst: Path) -> dict[str, Any]:
    stat = src.stat()
    return {
        "source_path": str(src.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "scratch_path": str(dst.resolve()),
    }


def records_manifest_problems(records_path: Path, source_path: Path) -> list[str]:
    if not records_path.exists():
        return []
    if not source_path.exists():
        return ["scratch_records_source_missing"]
    manifest_path = records_manifest_path(records_path)
    if not manifest_path.exists():
        return ["scratch_records_manifest_missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["scratch_records_manifest_malformed"]
    expected = records_manifest_payload(source_path, records_path)
    problems: list[str] = []
    for key in ("source_path", "source_size", "source_mtime_ns"):
        if manifest.get(key) != expected[key]:
            problems.append(f"scratch_records_manifest_{key}_mismatch")
    if records_path.stat().st_size != expected["source_size"]:
        problems.append("scratch_records_size_mismatch")
    return problems
