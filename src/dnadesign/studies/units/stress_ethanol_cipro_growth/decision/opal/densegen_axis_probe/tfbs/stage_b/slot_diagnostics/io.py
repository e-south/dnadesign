"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/slot_diagnostics/io.py

Filesystem and table contracts for Stage B slot diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ....core.selection_artifacts import read_probe_selection
from .contracts import SLOT_LABEL_SPECS, SlotLabelSpec


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stage B slot diagnostics config manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage B slot diagnostics config manifest must be a JSON object: {path}")
    return payload


def _slot_pair_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = manifest.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage B slot diagnostics require non-empty positive/null pairs")
    slot_rows = [row for row in rows if isinstance(row, Mapping) and str(row.get("label_name")) in SLOT_LABEL_SPECS]
    if not slot_rows:
        raise ValueError("Stage B slot diagnostics require at least one slot-family label pair")
    return slot_rows


def _campaign_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if manifest.get("status") != "PASS":
        raise ValueError("Stage B slot diagnostics require config manifest status PASS")
    rows = manifest.get("campaigns")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage B slot diagnostics require non-empty campaigns")
    return [row for row in rows if isinstance(row, Mapping)]


def _slot_label_table(path: Path, *, spec: SlotLabelSpec) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stage B slot label table not found: {path}")
    frame = pd.read_parquet(path)
    missing = sorted({"id", spec.label_name, spec.target_family_count_column} - set(frame.columns))
    if missing:
        if spec.target_family_count_column in missing:
            raise ValueError(
                "Stage B slot diagnostics missing target-family count column "
                f"{spec.target_family_count_column!r} in {path}"
            )
        raise ValueError(f"Stage B slot label table missing column(s): {missing}")
    out = frame.loc[:, ["id", spec.label_name, spec.target_family_count_column]].copy()
    out["id"] = out["id"].astype(str)
    if out["id"].duplicated().any():
        duplicates = out.loc[out["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"Stage B slot label table contains duplicate id(s): {duplicates}")
    out[spec.label_name] = pd.to_numeric(out[spec.label_name], errors="raise").astype(float)
    out[spec.target_family_count_column] = pd.to_numeric(out[spec.target_family_count_column], errors="raise").astype(
        int
    )
    invalid_counts = out.loc[
        ~out[spec.target_family_count_column].between(0, spec.max_target_family_count),
        spec.target_family_count_column,
    ]
    if not invalid_counts.empty:
        raise ValueError(
            "Stage B slot diagnostics found invalid target-family count(s): "
            f"{sorted(set(map(int, invalid_counts.tolist())))}"
        )
    return out


def _selection_table(workdir: Path, *, round_index: int) -> pd.DataFrame:
    return read_probe_selection(workdir, round_index)


def _campaign_workdir(config_path: Path) -> Path:
    if config_path.name != "campaign.yaml" or config_path.parent.name != "configs":
        raise ValueError(f"Stage B config path does not follow campaign/configs/campaign.yaml layout: {config_path}")
    return config_path.parent.parent


def _reject_duplicate_ids(ids: Sequence[str], *, path: Path, round_index: int) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"Stage B slot diagnostics selection artifact has duplicate id(s): {path}, round={round_index}"
        )
