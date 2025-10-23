"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/_events_util.py

Utilities for reading the **ledger** sinks and resolving setpoints for plots.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Set, Union

import pandas as pd
import pyarrow.compute as pc
from pyarrow import dataset as ds


def resolve_events_path(context) -> Path:
    """
    Resolve a *handle* into the outputs/ root. We no longer require the thin index to exist.
    - If the campaign YAML provides a path under 'events', use it (for legacy runs).
    - Otherwise return <campaign_dir>/outputs/ledger.index.parquet (even if missing),
      because callers only need its parent (outputs/) to find typed sinks.
    """
    p = context.data_paths.get("events")
    if p is None:
        p = context.campaign_dir / "outputs" / "ledger.index.parquet"
    return Path(p)


def load_events_with_setpoint(
    events_path: Path,
    base_columns: Iterable[str],
    round_selector: Optional[Union[str, int, List[int]]] = None,
) -> pd.DataFrame:
    """
    Read the minimum columns needed for a plot **from the ledger** and join
    the setpoint from `ledger.runs` via `objective__params.setpoint_vector`.
    `events_path` should point to outputs/ledger.index.parquet (thin index).
    """
    want: Set[str] = set(map(str, base_columns)) | {"run_id"}
    root = events_path.parent
    pred_dir = root / "ledger.predictions"
    runs_file = root / "ledger.runs.parquet"

    # Assert required ledger sinks exist
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Missing predictions sink: {pred_dir}. Run a round to produce it."
        )
    if not runs_file.exists():
        raise FileNotFoundError(
            f"Missing runs sink: {runs_file}. Run a round to produce it."
        )

    def _arrow_filter_for_rounds(d: ds.Dataset):
        if round_selector is None or round_selector == "all":
            return None
        # Compute single-round target for 'latest'/'unspecified'
        if round_selector in ("latest", "unspecified"):
            # Read only the as_of_round column to find max
            t = d.to_table(columns=["as_of_round"])
            if t.num_rows == 0:
                return None
            latest = int(pd.Series(t.column("as_of_round").to_pylist()).max())
            return pc.field("as_of_round") == latest
        # List[int] or int
        if isinstance(round_selector, list):
            vals = [int(x) for x in round_selector]
            return pc.field("as_of_round").isin(vals)
        try:
            r = int(round_selector)
            return pc.field("as_of_round") == r
        except Exception:
            return None

    def _read_pred(columns: list[str]) -> tuple[pd.DataFrame, set[str]]:
        """Return (df, names) from ledger.predictions (strict; no fallbacks)."""
        d = ds.dataset(str(pred_dir))
        names = {f.name for f in d.schema}
        cols = [c for c in columns if c in names]
        filt = _arrow_filter_for_rounds(d)
        tbl = d.to_table(columns=cols, filter=filt)
        df = tbl.to_pandas()
        return df, names

    df, names = _read_pred(sorted(want))

    # Validate required columns (except obj__diag__setpoint – handled below)
    missing = sorted([c for c in want if c not in names and c != "obj__diag__setpoint"])

    if missing:
        raise ValueError(f"ledger.predictions missing columns: {missing}")
    if df.empty:
        raise ValueError(
            f"ledger.predictions had zero rows after projecting columns: {sorted(want)}"
        )

    # Always join setpoint from ledger.runs (canonical)
    # Read run metadata (single Parquet file)
    need = ["run_id", "objective__params"]
    meta = pd.read_parquet(runs_file, columns=need)

    def _extract_setpoint(obj):
        try:
            return [float(x) for x in (obj or {}).get("setpoint_vector", [])]
        except Exception:
            return None

    meta["obj__diag__setpoint"] = meta["objective__params"].map(_extract_setpoint)
    meta = meta[["run_id", "obj__diag__setpoint"]]
    out = df.merge(meta, on="run_id", how="left")

    if out["obj__diag__setpoint"].dropna().empty:
        raise ValueError(
            "Could not resolve setpoint for any rows: no obj__diag__setpoint column and run_meta lacks objective__params.setpoint_vector."  # noqa
        )
    return out
