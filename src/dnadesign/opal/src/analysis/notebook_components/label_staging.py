"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/label_staging.py

Builds notebook rows for observed-label staging CSVs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ._support import compact_path, mapping, sequence

READER_VEC8_BATCH0_REQUIRED_COLUMNS = (
    "id",
    "sequence",
    "v00",
    "v10",
    "v01",
    "v11",
    "y00_star",
    "y10_star",
    "y01_star",
    "y11_star",
    "intensity_log2_offset_delta",
)


def discover_label_staging_inputs(workdir: str | Path) -> list[dict[str, Any]]:
    """Return small inventory rows for campaign-local label input CSVs."""

    root = Path(workdir)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("inputs/r*/reader_vec8_batch0*.csv")):
        rows.append(_label_staging_row(path, workdir=root))
    return rows


def build_notebook_label_staging_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return notebook-facing rows for label-staging input status."""

    campaign = mapping(view_model.get("campaign"))
    workdir = campaign.get("workdir")
    rows = []
    for row in sequence(view_model.get("label_staging")):
        item = mapping(row)
        rows.append(
            {
                "status": item.get("status") or "unknown",
                "round": item.get("round") or "",
                "rows": item.get("rows") or 0,
                "distinct_ids": item.get("distinct_ids") or 0,
                "missing_columns": ", ".join(sequence(item.get("missing_columns"))),
                "path": compact_path(item.get("path"), base=workdir),
            }
        )
    return rows


def _label_staging_row(path: Path, *, workdir: Path) -> dict[str, Any]:
    round_label = path.parent.name if path.parent.name.startswith("r") else ""
    row: dict[str, Any] = {
        "path": str(path),
        "path_label": compact_path(path, base=workdir),
        "round": round_label,
        "status": "ready",
        "rows": 0,
        "distinct_ids": 0,
        "missing_columns": [],
    }
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return {
            **row,
            "status": "read_error",
            "error": str(exc),
        }
    missing = [column for column in READER_VEC8_BATCH0_REQUIRED_COLUMNS if column not in frame.columns]
    row["rows"] = int(len(frame))
    row["distinct_ids"] = int(frame["id"].astype(str).nunique()) if "id" in frame.columns else 0
    row["missing_columns"] = missing
    if missing:
        row["status"] = "schema_attention"
    elif frame.empty:
        row["status"] = "empty"
    return row


__all__ = [
    "READER_VEC8_BATCH0_REQUIRED_COLUMNS",
    "build_notebook_label_staging_rows",
    "discover_label_staging_inputs",
]
