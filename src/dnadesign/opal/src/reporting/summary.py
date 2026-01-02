"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/reporting/summary.py

Helpers to summarize ledger run_meta entries and round.log.jsonl events.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ..core.utils import OpalError
from ..storage.ledger import LedgerReader


def list_runs(reader: LedgerReader, *, round_selector: Optional[int] = None) -> pd.DataFrame:
    df = reader.read_runs(
        columns=[
            "run_id",
            "as_of_round",
            "model__name",
            "objective__name",
            "selection__name",
            "training__y_ops",
            "stats__n_train",
            "stats__n_scored",
            "objective__summary_stats",
        ]
    )
    if round_selector is not None:
        df = df[df["as_of_round"] == int(round_selector)]
    return df.sort_values(["as_of_round", "run_id"]).reset_index(drop=True)


def select_run_meta(
    df_runs: pd.DataFrame,
    *,
    round_sel: Optional[int] = None,
    run_id: Optional[str] = None,
) -> pd.Series:
    if df_runs.empty:
        raise OpalError("No runs found in ledger.runs.parquet.")
    if run_id is not None:
        sel = df_runs[df_runs["run_id"].astype(str) == str(run_id)]
        if sel.empty:
            raise OpalError(f"run_id not found in ledger: {run_id}")
        return sel.sort_values(["run_id"]).tail(1).iloc[0]
    if round_sel is None:
        round_sel = int(df_runs["as_of_round"].max())
    sel = df_runs[df_runs["as_of_round"] == int(round_sel)]
    if sel.empty:
        raise OpalError(f"No runs found for as_of_round={int(round_sel)}.")
    return sel.sort_values(["run_id"]).tail(1).iloc[0]


def summarize_run_meta(row: pd.Series) -> Dict[str, Any]:
    return {
        "run_id": str(row.get("run_id", "")),
        "as_of_round": int(row.get("as_of_round", -1)),
        "model": row.get("model__name"),
        "objective": row.get("objective__name"),
        "selection": row.get("selection__name"),
        "y_ops": row.get("training__y_ops") or [],
        "stats_n_train": int(row.get("stats__n_train", 0)),
        "stats_n_scored": int(row.get("stats__n_scored", 0)),
        "objective_summary_stats": row.get("objective__summary_stats") or {},
    }


def _parse_ts(val: Any) -> Optional[datetime]:
    if not val:
        return None
    try:
        return datetime.fromisoformat(str(val))
    except Exception:
        return None


def load_round_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise OpalError(f"round.log.jsonl not found: {path}")
    events: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception as e:
            raise OpalError(f"Failed to parse round.log.jsonl: {e}")
    return events


def summarize_round_log(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    events = list(events)
    if not events:
        return {"events": 0}

    stages: Dict[str, int] = {}
    ts_list: List[datetime] = []
    for e in events:
        stage = str(e.get("stage", "unknown"))
        stages[stage] = stages.get(stage, 0) + 1
        ts = _parse_ts(e.get("ts"))
        if ts is not None:
            ts_list.append(ts)

    def _first_ts(stage_name: str) -> Optional[datetime]:
        for e in events:
            if e.get("stage") == stage_name:
                return _parse_ts(e.get("ts"))
        return None

    def _last_ts(stage_name: str) -> Optional[datetime]:
        for e in reversed(events):
            if e.get("stage") == stage_name:
                return _parse_ts(e.get("ts"))
        return None

    start_ts = _first_ts("start")
    done_ts = _last_ts("done")
    fit_start = _first_ts("fit_start")
    fit_done = _last_ts("fit")

    predict_batches = sum(1 for e in events if e.get("stage") == "predict_batch")
    predict_rows = int(sum(int(e.get("rows", 0)) for e in events if e.get("stage") == "predict_batch"))

    total_sec = (done_ts - start_ts).total_seconds() if start_ts and done_ts else None
    fit_sec = (fit_done - fit_start).total_seconds() if fit_start and fit_done else None

    return {
        "events": len(events),
        "stage_counts": stages,
        "predict_batches": predict_batches,
        "predict_rows": predict_rows,
        "start_ts": start_ts.isoformat() if start_ts else None,
        "done_ts": done_ts.isoformat() if done_ts else None,
        "duration_sec_total": total_sec,
        "duration_sec_fit": fit_sec,
    }
