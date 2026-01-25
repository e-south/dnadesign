"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/reporting/verify_outputs.py

Validates selection outputs against ledger predictions for a run. Provides
strict selection output checks and artifact resolution helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..core.utils import OpalError


def read_selection_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise OpalError(f"Selection file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def resolve_selection_score_column(df: pd.DataFrame) -> str:
    if "pred__y_obj_scalar" in df.columns:
        return "pred__y_obj_scalar"
    raise OpalError("Selection data missing pred__y_obj_scalar.")


def _extract_artifact_path(val: Any) -> Path | None:
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return Path(val[1])
    if isinstance(val, str):
        return Path(val)
    return None


def resolve_selection_path_from_artifacts(
    artifacts: dict | None,
    *,
    run_id: str | None,
) -> Path | None:
    if not isinstance(artifacts, dict):
        return None
    preferred_keys = []
    if run_id:
        preferred_keys.extend(
            [
                f"selection/selection_top_k__run_{run_id}.csv",
            ]
        )
    preferred_keys.extend(["selection/selection_top_k.csv"])
    for key in preferred_keys:
        if key not in artifacts:
            continue
        candidate = _extract_artifact_path(artifacts.get(key))
        if candidate is not None:
            return candidate
    return None


def compare_selection_to_ledger(
    selection_df: pd.DataFrame,
    ledger_df: pd.DataFrame,
    *,
    eps: float = 1e-6,
    id_col: str = "id",
    ledger_score_col: str = "pred__y_obj_scalar",
) -> tuple[dict, pd.DataFrame]:
    if id_col not in selection_df.columns:
        raise OpalError(f"Selection data missing '{id_col}' column.")
    if id_col not in ledger_df.columns:
        raise OpalError(f"Ledger predictions missing '{id_col}' column.")
    if ledger_score_col not in ledger_df.columns:
        raise OpalError(f"Ledger predictions missing '{ledger_score_col}' column.")

    selection_score_col = resolve_selection_score_column(selection_df)

    sel = selection_df[[id_col, selection_score_col]].copy()
    led = ledger_df[[id_col, ledger_score_col]].copy()
    sel[id_col] = sel[id_col].astype(str)
    led[id_col] = led[id_col].astype(str)
    sel[selection_score_col] = pd.to_numeric(sel[selection_score_col], errors="coerce")
    led[ledger_score_col] = pd.to_numeric(led[ledger_score_col], errors="coerce")

    sel = sel.rename(columns={selection_score_col: "selection_score"})
    led = led.rename(columns={ledger_score_col: "ledger_score"})

    joined = sel.merge(led, on=id_col, how="inner")
    if joined.empty:
        raise OpalError("No overlapping IDs between selection output and ledger predictions.")

    diff = joined["selection_score"] - joined["ledger_score"]
    abs_diff = diff.abs()
    joined = joined.assign(diff=diff, abs_diff=abs_diff)
    mismatch_mask = (abs_diff > float(eps)) | abs_diff.isna()
    mismatch_df = joined.loc[mismatch_mask, [id_col, "selection_score", "ledger_score", "diff", "abs_diff"]]
    mismatch_df = mismatch_df.sort_values(["abs_diff"], ascending=False)

    max_abs = abs_diff.max()
    summary = {
        "rows_compared": int(joined.shape[0]),
        "mismatch_count": int(mismatch_df.shape[0]),
        "max_abs_diff": None if pd.isna(max_abs) else float(max_abs),
        "eps": float(eps),
        "selection_score_col": selection_score_col,
        "ledger_score_col": ledger_score_col,
    }
    return summary, mismatch_df
