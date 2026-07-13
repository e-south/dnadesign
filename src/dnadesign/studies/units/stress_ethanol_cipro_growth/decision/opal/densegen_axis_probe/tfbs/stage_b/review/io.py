"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/review/io.py

Artifact readers for DenseGen TFBS Stage B realized-label review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ....core.selection_artifacts import read_probe_selection
from ..slot_diagnostics.contracts import SLOT_LABEL_SPECS


def read_review_manifest(path: Path) -> dict[str, Any]:
    """Read and validate a Stage B config manifest JSON object."""

    if not path.exists():
        raise FileNotFoundError(f"Stage B config manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage B config manifest must be a JSON object: {path}")
    return payload


def campaign_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return realized-review campaign rows from a PASS config manifest."""

    if manifest.get("status") != "PASS":
        raise ValueError("Stage B realized review requires config manifest status PASS")
    rows = manifest.get("campaigns")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage B realized review requires non-empty campaigns")
    return [row for row in rows if isinstance(row, Mapping)]


def pair_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return positive/null pair rows from a config manifest."""

    rows = manifest.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage B realized review requires non-empty positive/null pairs")
    return [row for row in rows if isinstance(row, Mapping)]


def has_slot_pairs(manifest: Mapping[str, Any]) -> bool:
    """Return whether the manifest contains slot-count diagnostic label pairs."""

    return any(str(row.get("label_name")) in SLOT_LABEL_SPECS for row in pair_rows(manifest))


def label_table(path: Path, *, label_name: str) -> pd.DataFrame:
    """Read a Stage B label table with a unique string id column."""

    if not path.exists():
        raise FileNotFoundError(f"Stage B label table not found: {path}")
    frame = pd.read_parquet(path)
    missing = sorted({"id", label_name} - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B label table missing column(s): {missing}")
    out = frame.copy()
    out["id"] = out["id"].astype(str)
    if out["id"].duplicated().any():
        duplicates = out.loc[out["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"Stage B label table contains duplicate id(s): {duplicates}")
    out[label_name] = pd.to_numeric(out[label_name], errors="raise")
    return out


def selection_table(workdir: Path, *, round_index: int) -> pd.DataFrame:
    """Read one OPAL selection artifact for a Stage B campaign round."""

    return read_probe_selection(workdir, round_index)


def campaign_workdir(config_path: Path) -> Path:
    """Resolve the OPAL campaign workdir from its config path."""

    if config_path.name != "campaign.yaml" or config_path.parent.name != "configs":
        raise ValueError(f"Stage B config path does not follow campaign/configs/campaign.yaml layout: {config_path}")
    return config_path.parent.parent
