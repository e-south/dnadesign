"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/caches.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..core.utils import OpalError, now_iso


@dataclass(frozen=True)
class RecordCaches:
    campaign_slug: str

    def latest_as_of_round_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_as_of_round"

    def latest_pred_scalar_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_pred_scalar"

    def latest_pred_run_id_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_pred_run_id"

    def latest_pred_as_of_round_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_pred_as_of_round"

    def latest_pred_written_at_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_pred_written_at"

    def ensure_cache_columns(
        self,
        df: pd.DataFrame,
        *,
        include_label_hist: bool,
        label_hist_col: str,
        include_pred_provenance: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Ensure OPAL cache columns exist in records.parquet. Returns (df, added_columns).
        Adds nullable columns without mutating existing data.
        """
        out = df.copy()
        added: list[str] = []
        n = len(out)
        if include_label_hist:
            if label_hist_col not in out.columns:
                out[label_hist_col] = [None] * n
                added.append(label_hist_col)
        col_r = self.latest_as_of_round_col()
        if col_r not in out.columns:
            out[col_r] = pd.Series([pd.NA] * n, dtype="Int64")
            added.append(col_r)
        col_s = self.latest_pred_scalar_col()
        if col_s not in out.columns:
            out[col_s] = pd.Series([np.nan] * n, dtype="Float64")
            added.append(col_s)
        if include_pred_provenance:
            col_run = self.latest_pred_run_id_col()
            if col_run not in out.columns:
                out[col_run] = pd.Series([pd.NA] * n, dtype="string")
                added.append(col_run)
            col_pred_r = self.latest_pred_as_of_round_col()
            if col_pred_r not in out.columns:
                out[col_pred_r] = pd.Series([pd.NA] * n, dtype="Int64")
                added.append(col_pred_r)
            col_ts = self.latest_pred_written_at_col()
            if col_ts not in out.columns:
                out[col_ts] = pd.Series([pd.NA] * n, dtype="string")
                added.append(col_ts)
        return out, added

    def update_latest_cache(
        self,
        df: pd.DataFrame,
        *,
        latest_as_of_round: int,
        latest_scalar_by_id: Dict[str, float],
        require_columns_present: bool,
        latest_pred_run_id: str | None = None,
        latest_pred_as_of_round: int | None = None,
        latest_pred_written_at: str | None = None,
    ) -> pd.DataFrame:
        out = df.copy()
        col_r = self.latest_as_of_round_col()
        col_s = self.latest_pred_scalar_col()
        col_run = self.latest_pred_run_id_col()
        col_pred_r = self.latest_pred_as_of_round_col()
        col_ts = self.latest_pred_written_at_col()
        if require_columns_present:
            missing = [c for c in (col_r, col_s, col_run, col_pred_r, col_ts) if c not in out.columns]
            if missing:
                raise OpalError(
                    "Required cache columns missing (needed for run-aware provenance): "
                    f"{missing}. Run `opal init` to add cache columns or set "
                    "`safety.write_back_requires_columns_present: false` to allow backfill."
                )
        else:
            if col_r not in out.columns:
                out[col_r] = None
            if col_s not in out.columns:
                out[col_s] = None
        if not require_columns_present:
            if col_run not in out.columns:
                out[col_run] = None
            if col_pred_r not in out.columns:
                out[col_pred_r] = None
            if col_ts not in out.columns:
                out[col_ts] = None

        incoming = pd.Series(latest_scalar_by_id, dtype="float64")
        non_finite = ~np.isfinite(incoming.to_numpy())
        if non_finite.any():
            bad = incoming[non_finite]
            preview = [{"id": str(k), "value": (None if pd.isna(v) else float(v))} for k, v in bad.head(15).items()]
            raise OpalError(
                "update_latest_cache received non‑finite values for "
                f"{col_s} ({int(non_finite.sum())} offender(s)). Sample: {preview}"
            )

        id_series = out["id"].astype(str)
        mapped = id_series.map(incoming.to_dict())
        mask_new = mapped.notna()
        out.loc[mask_new, col_s] = mapped[mask_new]
        out.loc[mask_new, col_r] = int(latest_as_of_round)
        if col_run in out.columns and latest_pred_run_id is not None:
            out.loc[mask_new, col_run] = str(latest_pred_run_id)
        pred_round = latest_pred_as_of_round if latest_pred_as_of_round is not None else latest_as_of_round
        if col_pred_r in out.columns:
            out.loc[mask_new, col_pred_r] = int(pred_round)
        if col_ts in out.columns:
            ts_val = latest_pred_written_at or now_iso()
            out.loc[mask_new, col_ts] = str(ts_val)
        return out
