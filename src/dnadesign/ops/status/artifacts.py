"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/artifacts.py

Artifact and file-inspection helpers for read-only OPS status surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml


def namespace_column_counts(columns: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for column in columns:
        if "__" not in column:
            continue
        namespace = column.split("__", 1)[0].strip()
        if namespace:
            counts[namespace] += 1
    return counts


def overlay_namespace_names(dataset_dir: Path) -> list[str]:
    derived_dir = dataset_dir / "_derived"
    if not derived_dir.exists() or not derived_dir.is_dir():
        return []
    namespaces: list[str] = []
    for entry in sorted(derived_dir.iterdir(), key=lambda item: item.name):
        if entry.is_file() and entry.suffix == ".parquet":
            namespaces.append(entry.stem)
            continue
        if entry.is_dir() and any(entry.glob("part-*.parquet")):
            namespaces.append(entry.name)
    return namespaces


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def file_count(path: Path) -> int:
    return sum(1 for candidate in path.rglob("*") if candidate.is_file())


def load_yaml_mapping(path: Path, *, label: str) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping: {path}")
    return payload


def parquet_row_count(records_path: Path) -> int:
    return int(pq.ParquetFile(str(records_path)).metadata.num_rows)


__all__ = [
    "file_count",
    "line_count",
    "load_yaml_mapping",
    "namespace_column_counts",
    "overlay_namespace_names",
    "parquet_row_count",
]
