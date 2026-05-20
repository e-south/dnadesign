"""Scratch records manifest checks for the DenseGen axis probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def records_manifest_path(records_path: Path) -> Path:
    return records_path.parent / "records_manifest.json"


def records_manifest_payload(
    src: Path,
    dst: Path,
    *,
    copy_mode: str = "clone",
    row_count: int | None = None,
    subset_id_count: int | None = None,
    subset_ids_sha256: str | None = None,
) -> dict[str, Any]:
    stat = src.stat()
    payload: dict[str, Any] = {
        "source_path": str(src.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "scratch_path": str(dst.resolve()),
        "copy_mode": str(copy_mode),
    }
    if row_count is not None:
        payload["row_count"] = int(row_count)
    if subset_id_count is not None:
        payload["subset_id_count"] = int(subset_id_count)
    if subset_ids_sha256 is not None:
        payload["subset_ids_sha256"] = str(subset_ids_sha256)
    return payload


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
    if manifest.get("copy_mode") == "subset":
        try:
            import pyarrow.parquet as pq

            row_count = int(pq.ParquetFile(records_path).metadata.num_rows)
        except Exception:
            row_count = -1
        if "row_count" in manifest and row_count != int(manifest["row_count"]):
            problems.append("scratch_records_row_count_mismatch")
    elif records_path.stat().st_size != expected["source_size"]:
        problems.append("scratch_records_size_mismatch")
    return problems
