"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/move_stats.py

Normalize optimizer move-stat rows into a strict typed frame for downstream analysis.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOVE_STATS_COLUMNS = [
    "sweep_idx",
    "chain",
    "phase",
    "move_kind",
    "attempted",
    "accepted",
    "delta",
    "delta_hamming",
    "gibbs_changed",
]


def _empty_move_stats_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MOVE_STATS_COLUMNS)


def move_stats_frame(
    move_stats: object,
    *,
    phase: str | None = None,
) -> pd.DataFrame:
    if not isinstance(move_stats, list) or not move_stats:
        return _empty_move_stats_frame()

    records = [row for row in move_stats if isinstance(row, dict)]
    if not records:
        return _empty_move_stats_frame()
    if phase and not any("phase" in row for row in records):
        return _empty_move_stats_frame()

    frame = pd.DataFrame.from_records(records, columns=MOVE_STATS_COLUMNS)
    if frame.empty:
        return _empty_move_stats_frame()

    if phase:
        frame = frame.loc[frame["phase"] == phase].copy()
        if frame.empty:
            return _empty_move_stats_frame()

    sweep = pd.to_numeric(frame.get("sweep_idx"), errors="coerce")
    attempted = pd.to_numeric(frame.get("attempted"), errors="coerce")
    accepted = pd.to_numeric(frame.get("accepted"), errors="coerce")
    move_kind = frame.get("move_kind")
    if move_kind is None:
        return _empty_move_stats_frame()
    move_kind = move_kind.astype("string")
    sweep_values = sweep.to_numpy(dtype=float)
    attempted_values = attempted.to_numpy(dtype=float)
    accepted_values = accepted.to_numpy(dtype=float)
    move_kind_len = move_kind.str.len().fillna(0).to_numpy(dtype=int)
    valid = (
        np.isfinite(sweep_values)
        & np.isfinite(attempted_values)
        & np.isfinite(accepted_values)
        & (attempted_values >= 0.0)
        & (accepted_values >= 0.0)
        & (accepted_values <= attempted_values)
        & (move_kind_len > 0)
    )
    filtered = frame.loc[valid].copy()
    if filtered.empty:
        return _empty_move_stats_frame()

    chain_source = pd.to_numeric(filtered.get("chain"), errors="coerce")
    chain = chain_source.where(np.isfinite(chain_source), 0.0).fillna(0.0)

    filtered["sweep_idx"] = sweep.loc[filtered.index].astype(int)
    filtered["attempted"] = attempted.loc[filtered.index].astype(int)
    filtered["accepted"] = accepted.loc[filtered.index].astype(int)
    filtered["chain"] = chain.astype(int)
    filtered["move_kind"] = move_kind.loc[filtered.index].astype(str)
    filtered["delta"] = pd.to_numeric(filtered.get("delta"), errors="coerce")
    filtered["delta_hamming"] = pd.to_numeric(filtered.get("delta_hamming"), errors="coerce")
    if "gibbs_changed" in filtered.columns:
        filtered["gibbs_changed"] = filtered["gibbs_changed"].where(filtered["gibbs_changed"].notna(), None)
    else:
        filtered["gibbs_changed"] = None
    return filtered.loc[:, MOVE_STATS_COLUMNS].reset_index(drop=True)
