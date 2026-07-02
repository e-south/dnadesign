"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/sae_window_summary/io.py

I/O helpers for Eco1 SAE window-summary materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def read_mask_rows(path: Path) -> list[dict[str, Any]]:
    """Read generated mask-set residue rows."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("residues"), list):
        raise ValueError(f"{path} must contain mask-set residues")
    return [dict(row) for row in payload["residues"]]


def read_profiles(path: Path) -> list[dict[str, Any]]:
    """Read accepted SAE profile rows."""

    rows = pq.read_table(
        path,
        columns=[
            "candidate_id",
            "sequence_hash",
            "model",
            "sae_model",
            "sequence_length",
            "feature_dictionary_size",
            "status",
        ],
    ).to_pylist()
    return [dict(row) for row in rows if str(row.get("status") or "") == "accepted"]


def read_candidate_design_classes(path: Path | None) -> dict[str, str]:
    """Read design class ids by candidate id."""

    if path is None or not path.exists():
        return {}
    rows = pq.read_table(path, columns=["candidate_id", "design_class_id"]).to_pylist()
    return {str(row["candidate_id"]): str(row.get("design_class_id") or "") for row in rows}


def read_feature_catalog(path: Path | None) -> dict[tuple[str, int], dict[str, str]]:
    """Read concise SAE feature labels when the exact catalog is available."""

    if path is None or not path.exists():
        return {}
    rows = pq.read_table(path, columns=["sae_model", "feature_index", "label", "description"]).to_pylist()
    return {
        (str(row["sae_model"]), int(row["feature_index"])): {
            "label": str(row.get("label") or ""),
            "description": str(row.get("description") or ""),
        }
        for row in rows
    }


def write_summary(path: Path, rows: list[dict[str, Any]], *, metadata: dict[str, str]) -> None:
    """Write SAE window-summary rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    encoded = {key.encode(): value.encode() for key, value in metadata.items()}
    pq.write_table(table.replace_schema_metadata(encoded), path)
