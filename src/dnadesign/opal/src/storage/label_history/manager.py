from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from ...core.utils import OpalError
from .labels import (
    append_labels_from_df,
    count_entries,
    training_labels_from_y,
    training_labels_with_round,
)
from .parsing import normalize_hist_cell, parse_hist_cell_strict
from .predictions import append_predictions_from_arrays


@dataclass(frozen=True)
class LabelHistory:
    campaign_slug: str

    def label_hist_col(self) -> str:
        return f"opal__{self.campaign_slug}__label_hist"

    normalize_hist_cell = staticmethod(normalize_hist_cell)
    parse_hist_cell_strict = staticmethod(parse_hist_cell_strict)

    def validate_label_hist(self, df: pd.DataFrame, *, require: bool = True) -> None:
        lh = self.label_hist_col()
        if lh not in df.columns:
            if require:
                raise OpalError(f"Expected label history column '{lh}' not found in records.parquet.")
            return
        bad: List[Dict[str, str]] = []
        for _id, cell in df[["id", lh]].itertuples(index=False, name=None):
            try:
                _ = self.parse_hist_cell_strict(cell)
            except Exception as e:
                bad.append({"id": str(_id), "error": str(e)})
                if len(bad) >= 5:
                    break
        if bad:
            raise OpalError(f"label_hist validation failed (sample={bad}).")

    def repair_label_hist(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
        """
        Coerce label_hist cells into normalized list-of-dicts; drop malformed entries.
        Returns (clean_df, report).
        """
        lh = self.label_hist_col()
        if lh not in df.columns:
            raise OpalError(f"Expected label history column '{lh}' not found in records.parquet.")

        out = df.copy()
        changed_rows = 0
        dropped_total = 0
        for idx, cell in out[lh].items():
            before = count_entries(cell)
            cleaned = self.normalize_hist_cell(cell)
            after = len(cleaned)
            dropped_total += max(0, before - after)
            if before != after or not isinstance(cell, list):
                changed_rows += 1
            out.at[idx, lh] = cleaned

        report = {
            "rows_changed": int(changed_rows),
            "entries_dropped": int(dropped_total),
        }
        return out, report

    def append_labels_from_df(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        r: int,
        *,
        src: str = "ingest_y",
        fail_if_any_existing_labels: bool = True,
        if_exists: str = "fail",
    ) -> pd.DataFrame:
        return append_labels_from_df(
            self,
            df,
            labels,
            r,
            src=src,
            fail_if_any_existing_labels=fail_if_any_existing_labels,
            if_exists=if_exists,
        )

    def training_labels_with_round(
        self,
        df: pd.DataFrame,
        as_of_round: int,
        *,
        cumulative_training: bool,
        dedup_policy: str,
    ) -> pd.DataFrame:
        return training_labels_with_round(
            self,
            df,
            as_of_round,
            cumulative_training=cumulative_training,
            dedup_policy=dedup_policy,
        )

    def training_labels_from_y(self, df: pd.DataFrame, as_of_round: int) -> pd.DataFrame:
        return training_labels_from_y(self, df, as_of_round)

    def append_predictions_from_arrays(
        self,
        df: pd.DataFrame,
        *,
        ids: List[str],
        y_hat: Any,
        as_of_round: int,
        run_id: str,
        objective: Dict[str, Any],
        metrics_by_name: Dict[str, List[float]],
        selection_rank: Any,
        selection_top_k: Any,
        ts: str | None = None,
    ) -> pd.DataFrame:
        return append_predictions_from_arrays(
            self,
            df,
            ids=ids,
            y_hat=y_hat,
            as_of_round=as_of_round,
            run_id=run_id,
            objective=objective,
            metrics_by_name=metrics_by_name,
            selection_rank=selection_rank,
            selection_top_k=selection_top_k,
            ts=ts,
        )
