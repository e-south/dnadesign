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

from ..core.utils import OpalError


@dataclass(frozen=True)
class RecordCaches:
    campaign_slug: str

    def latest_as_of_round_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_as_of_round"

    def latest_pred_scalar_col(self) -> str:
        return f"opal__{self.campaign_slug}__latest_pred_scalar"

    def ensure_cache_columns(
        self,
        df: pd.DataFrame,
        *,
        include_label_hist: bool,
        label_hist_col: str,
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
        return out, added

    def update_latest_cache(
        self,
        df: pd.DataFrame,
        *,
        latest_as_of_round: int,
        latest_scalar_by_id: Dict[str, float],
        require_columns_present: bool,
    ) -> pd.DataFrame:
        out = df.copy()
        col_r = self.latest_as_of_round_col()
        col_s = self.latest_pred_scalar_col()
        if require_columns_present:
            missing = [c for c in (col_r, col_s) if c not in out.columns]
            if missing:
                raise OpalError(f"Required cache columns missing: {missing}")
        else:
            if col_r not in out.columns:
                out[col_r] = None
            if col_s not in out.columns:
                out[col_s] = None

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
        return out
