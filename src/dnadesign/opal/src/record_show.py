"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/record_show.py

Builds a compact per-record summary from the records table:
- id, length (and sequence if requested),
- ground-truth label + source round (derived from label history),
- per-round stored y_pred/rank/selected for completed rounds.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .utils import ExitCodes, OpalError


def build_record_report(
    df: pd.DataFrame,
    campaign_slug: str,
    id_: Optional[str] = None,
    sequence: Optional[str] = None,
    with_sequence: bool = False,
) -> Dict[str, Any]:
    if id_:
        row = df.loc[df["id"] == id_]
    else:
        row = df.loc[df["sequence"].str.upper() == (sequence or "").upper()]
    if row.empty:
        raise OpalError("Record not found.", ExitCodes.NOT_FOUND)

    row = row.iloc[0]
    out: Dict[str, Any] = {
        "id": row["id"],
        "length": (
            int(row["length"])
            if "length" in row and pd.notna(row["length"])
            else len(str(row["sequence"]))
        ),
    }
    if with_sequence:
        out["sequence"] = row["sequence"]

    # ground truth + source round
    # derive from label history
    lh_col = f"opal__{campaign_slug}__label_hist"
    if lh_col in df.columns and isinstance(row.get(lh_col), list) and row[lh_col]:
        best = max(row[lh_col], key=lambda h: int(h.get("r", -1)))
        out["ground_truth_label"] = float(best["y"])
        out["ground_truth_src_round"] = int(best["r"])

    # per-round predictions if present
    per_round = {}
    prefix = f"opal__{campaign_slug}__r"
    for col in df.columns:
        if col.startswith(prefix) and col.endswith("__pred"):
            r = int(col[len(prefix) :].split("__")[0])
            y_pred = row[col]
            rank_col = f"{prefix}{r}__rank_competition"
            sel_col = f"{prefix}{r}__selected_top_k_bool"
            per_round[r] = {
                "y_pred": None if pd.isna(y_pred) else float(y_pred),
                "rank": None if pd.isna(row.get(rank_col)) else int(row.get(rank_col)),
                "selected": bool(row.get(sel_col)),
            }
    if per_round:
        out["per_round"] = dict(sorted(per_round.items(), key=lambda kv: kv[0]))
    return out
