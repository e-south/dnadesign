"""Fail-fast JSON and CSV helpers for Stage B notebook visual registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..review_plots import REALIZED_REVIEW_PLOT_MANIFEST_SCHEMA_VERSION
from ..slot_plots import SLOT_DIAGNOSTIC_PLOT_MANIFEST_SCHEMA_VERSION
from .contracts import COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION


def read_collection_visual_index(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION:
        raise ValueError(f"Unsupported OPAL collection visual index schema: {payload.get('schema_version')!r}")
    mapping_list(payload.get("visuals"), field="visuals")
    mapping_list(payload.get("comparison_sets"), field="comparison_sets")
    return payload


def read_realized_plot_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != REALIZED_REVIEW_PLOT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage B realized review plot manifest schema: {payload.get('schema_version')!r}")
    if not mapping_list(payload.get("plots"), field="plots"):
        raise ValueError("Stage B realized review plot manifest contains no plots")
    return payload


def read_slot_plot_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != SLOT_DIAGNOSTIC_PLOT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage B slot diagnostic plot manifest schema: {payload.get('schema_version')!r}")
    if not mapping_list(payload.get("plots"), field="plots"):
        raise ValueError("Stage B slot diagnostic plot manifest contains no plots")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def mapping_list(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"OPAL collection visual index field {field!r} must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"OPAL collection visual index field {field!r} must contain objects")
    return list(value)


def require_existing_file(path: Path, *, role: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Stage B realized review {role} not found: {path}")


def csv_row_count(path: Path, *, role: str) -> int:
    require_existing_file(path, role=role)
    with path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)
