"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/opal/join.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _ledger_paths(campaign_dir: Path) -> dict:
    # We expect OPAL 'outputs' folder with ledger files
    base = Path(campaign_dir) / "outputs"
    return {
        "index": base / "ledger.index.parquet",
        "predictions": base / "ledger.predictions",
        "labels": base / "ledger.labels",
    }


def _read_parquet_parts(
    d: Path, columns: list[str] | None = None, newest_only: bool = False
) -> pd.DataFrame:
    """Read a directory of parquet parts (or a single parquet). By default, concatenate all parts."""
    if not d.exists():
        raise FileNotFoundError(f"Path not found: {d}")
    if d.is_dir():
        files = sorted(d.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {d}")
        if newest_only:
            files = [max(files, key=lambda p: p.stat().st_mtime)]
        dfs = [
            pd.read_parquet(p, columns=columns) if columns else pd.read_parquet(p)
            for p in files
        ]
        return pd.concat(dfs, ignore_index=True)
    return pd.read_parquet(d, columns=columns) if columns else pd.read_parquet(d)


def select_ids(
    campaign_dir: Path,
    run_selector: str = "latest",
    scope: str = "scored_pool",
    ids_path: Path | None = None,
) -> set[str]:
    paths = _ledger_paths(campaign_dir)
    if scope == "custom":
        if not ids_path:
            raise ValueError("--ids is required for scope=custom.")
        df = (
            pd.read_parquet(ids_path)
            if str(ids_path).endswith(".parquet")
            else pd.read_csv(ids_path)
        )
        col = "id" if "id" in df.columns else df.columns[0]
        return set(map(str, df[col].tolist()))
    # For now, fallback to predictions ledger for scored_pool
    preds = (
        _read_parquet_parts(paths["predictions"], columns=["run_id", "id"])
        if Path(paths["predictions"]).exists()
        else None
    )

    if preds is None:
        raise FileNotFoundError("OPAL predictions ledger not found.")
    if run_selector == "latest" and "run_id" in preds.columns:
        rid = preds["run_id"].iloc[-1]
        sub = preds[preds["run_id"] == rid]
    elif run_selector.startswith("round:") and "run_id" in preds.columns:
        # naive mapping: pick last run_id (approximation)
        sub = preds
    else:
        sub = preds
    if scope == "scored_pool":
        return set(map(str, sub["id"].unique().tolist()))
    if scope == "selected_top_k":
        if "selected" not in sub.columns:
            raise KeyError("'selected' column not found in predictions ledger.")
        return set(
            map(str, sub.loc[sub["selected"].astype(bool), "id"].unique().tolist())
        )
    raise ValueError(f"Unsupported ids scope: {scope}")


def join_fields(
    df: pd.DataFrame,
    campaign_dir: Path,
    run_selector: str,
    fields: list[str],
    as_of_round: Optional[int] = None,
) -> pd.DataFrame:
    paths = _ledger_paths(campaign_dir)
    preds = (
        _read_parquet_parts(paths["predictions"])
        if Path(paths["predictions"]).exists()
        else None
    )

    if preds is None:
        raise FileNotFoundError("OPAL predictions ledger not found.")
    # Reduce to the requested slice
    preds["id"] = preds["id"].astype(str)
    if as_of_round is not None and "as_of_round" in preds.columns:
        preds = preds[preds["as_of_round"] == int(as_of_round)]
        if preds.empty:
            raise FileNotFoundError(
                f"No predictions with as_of_round={as_of_round} found in {paths['predictions']}"
            )
    elif run_selector == "latest" and "run_id" in preds.columns:
        rid = preds["run_id"].dropna().astype(str).max()
        preds = preds[preds["run_id"] == rid]
    # Assert every requested field exists
    missing = [f for f in fields if f not in preds.columns]
    if missing:
        raise KeyError(f"Fields not present in OPAL predictions: {missing}")
    cols = ["id"] + list(fields)
    join_df = preds[cols].drop_duplicates("id", keep="last")
    return df.merge(join_df, on="id", how="left")
