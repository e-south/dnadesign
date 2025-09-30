"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/record_show.py

Record-centric report: gathers label history from records.parquet AND all
run_pred entries for the record from outputs/events.parquet, respecting
each event's as_of_round.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .data_access import RecordsStore


def _relpath(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(Path(p).resolve())


def build_record_report(
    df_records: pd.DataFrame,
    campaign_slug: str,
    *,
    id_: Optional[str] = None,
    sequence: Optional[str] = None,
    with_sequence: bool = True,
    events_path: Optional[Path] = None,
    records_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if id_ is None and sequence is None:
        raise ValueError("Provide id or sequence.")
    row = None
    if id_ is not None:
        m = df_records["id"].astype(str) == str(id_)
        if m.any():
            row = df_records.loc[m].head(1)
    if row is None and sequence is not None:
        m = (
            (df_records.get("sequence") == sequence)
            if "sequence" in df_records.columns
            else None
        )
        if m is not None and m.any():
            row = df_records.loc[m].head(1)
    if row is None:
        return {"error": "record not found"}

    row = row.iloc[0]
    rid = str(row["id"])
    # Prefer sequence from records if present; else we can pick it from events later.
    seq_val = None
    if "sequence" in df_records.columns:
        try:
            v = row.get("sequence", None)
            seq_val = None if pd.isna(v) else str(v)
        except Exception:
            seq_val = None

    report: Dict[str, Any] = {
        "id": rid,
        "sequence": (seq_val if with_sequence else None),
        "label_hist_column": f"opal__{campaign_slug}__label_hist",
    }

    lh_col = f"opal__{campaign_slug}__label_hist"
    hist = row.get(lh_col)
    report["labels"] = (
        RecordsStore._normalize_hist_cell(hist) if hist is not None else []
    )

    # Sources (succinct, relative) for clarity when many parquets exist
    srcs: Dict[str, str] = {}
    if records_path is not None:
        srcs["records"] = _relpath(Path(records_path))
    if events_path is not None:
        srcs["events"] = _relpath(Path(events_path))
    if srcs:
        report["sources"] = srcs

    latest_rank_comp: Optional[int] = None
    avg_rank_comp: Optional[float] = None

    if events_path is not None and events_path.exists():
        ev = pd.read_parquet(events_path)
        col = (
            "event"
            if "event" in ev.columns
            else ("kind" if "kind" in ev.columns else None)
        )
        if col is None:
            report["runs"] = []
            return report
        ev = ev[(ev[col] == "run_pred") & (ev["id"].astype(str) == rid)]
        # compact per-round view
        cols = [
            c
            for c in ev.columns
            if c.startswith("pred__") or c.startswith("unc__") or c.startswith("sel__")
        ]
        cols = ["as_of_round", "run_id", "sequence"] + cols

        out = ev[cols].sort_values(["as_of_round", "run_id"])
        report["runs"] = out.to_dict(orient="records")

        # If we didn't have a sequence in records, adopt first non-null from events.
        if with_sequence and report.get("sequence") is None:
            try:
                nonnull = out["sequence"].dropna()
                if not nonnull.empty:
                    report["sequence"] = str(nonnull.iloc[0])
            except Exception:
                pass

        # Rank summaries (competition rank). Robust to missing col.
        if "sel__rank_competition" in out.columns and not out.empty:
            try:
                # latest round = max as_of_round; within that, prefer the max run_id (most recent)
                lr = int(out["as_of_round"].max())
                latest = out[out["as_of_round"] == lr]
                # pick the last lexicographic run_id (they include ISO timestamp) → most recent
                latest = latest.sort_values(["run_id"]).tail(1)
                v = latest["sel__rank_competition"].iloc[0]
                latest_rank_comp = int(v) if pd.notna(v) else None
            except Exception:
                latest_rank_comp = None
            try:
                avg_rank_comp = float(
                    pd.to_numeric(out["sel__rank_competition"], errors="coerce")
                    .dropna()
                    .mean()
                )
            except Exception:
                avg_rank_comp = None
    else:
        report["runs"] = []

    # Attach summaries outside the branch for clarity
    report["latest_rank_competition"] = latest_rank_comp
    report["avg_rank_competition_across_rounds"] = avg_rank_comp

    return report
