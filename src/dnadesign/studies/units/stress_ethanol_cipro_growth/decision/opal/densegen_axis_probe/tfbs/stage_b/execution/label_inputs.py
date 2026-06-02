"""Label-input materialization contracts for Stage B execution."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def write_label_input_for_ids(
    *,
    path: Path,
    label_table_path: Path,
    records_path: Path,
    label_name: str,
    ids: Sequence[str],
) -> None:
    if not ids:
        raise ValueError("Stage B follow-up label input requires at least one selected id")
    label_table = pd.read_parquet(label_table_path)
    required = {"id", label_name}
    missing = sorted(required - set(label_table.columns))
    if missing:
        raise ValueError(f"Stage B label table missing column(s): {missing}")
    frame = label_table.copy()
    if "sequence" not in frame.columns:
        identity = pd.read_parquet(records_path, columns=["id", "sequence"])
        frame = frame.merge(identity, on="id", how="left", validate="one_to_one")
    if "sequence" not in frame.columns:
        raise ValueError("Stage B follow-up label input requires sequence")
    wanted = [str(value) for value in ids]
    if len(set(wanted)) != len(wanted):
        raise ValueError("Stage B follow-up label input selected ids must be unique")
    selected = frame.loc[frame["id"].astype(str).isin(set(wanted)), ["id", "sequence", label_name]].copy()
    found = set(selected["id"].astype(str).tolist())
    missing_ids = sorted(set(wanted) - found)
    if missing_ids:
        raise ValueError(f"Stage B label table missing selected id(s): {missing_ids[:10]}")
    order = {candidate_id: index for index, candidate_id in enumerate(wanted)}
    selected["__order__"] = selected["id"].astype(str).map(order)
    selected = selected.sort_values("__order__").drop(columns=["__order__"])
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(path, index=False, compression="zstd")


def observed_label_ids_for_round(*, sidecar_path: Path, round_index: int) -> set[str]:
    if not sidecar_path.exists():
        return set()
    frame = pd.read_parquet(sidecar_path, columns=["id", "observed_round"])
    round_frame = frame.loc[pd.to_numeric(frame["observed_round"], errors="coerce") == int(round_index)]
    return {str(value).strip() for value in round_frame["id"].dropna().tolist()}
