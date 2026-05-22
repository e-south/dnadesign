"""
Value and identity helpers for `opal ingest-y` policy checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ....core.utils import OpalError


def required_columns_for_new_rows(
    *,
    records_df: pd.DataFrame,
    x_column_name: str,
    require_x_column: bool,
    shared_label_source: bool,
) -> list[str]:
    required_cols = ["bio_type", "alphabet"]
    if require_x_column and not shared_label_source:
        if x_column_name not in records_df.columns:
            raise OpalError(f"records.parquet missing required X column '{x_column_name}'.")
        required_cols.append(x_column_name)
    return required_cols


def build_unknown_mask(
    frame: pd.DataFrame,
    *,
    known_ids: set[str],
    known_sequences: set[str],
) -> pd.Series:
    id_known = pd.Series(False, index=frame.index)
    if "id" in frame.columns and known_ids:
        id_series = frame["id"]
        id_known = id_series.notna() & id_series.astype(str).isin(known_ids)
    seq_known = pd.Series(False, index=frame.index)
    if "sequence" in frame.columns and known_sequences:
        seq_series = frame["sequence"]
        seq_known = seq_series.notna() & seq_series.astype(str).isin(known_sequences)
    return ~(id_known | seq_known)


def infer_default(series: pd.Series) -> str | None:
    values = series.dropna().astype(str)
    if values.empty:
        return None
    mode = values.mode()
    if mode.empty:
        return None
    return str(mode.iloc[0])


def is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0:
        return True
    return False


def col_is_listlike(series: pd.Series) -> bool:
    for value in series.head(20).tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        return isinstance(value, (list, tuple, np.ndarray))
    return False


def col_has_str(series: pd.Series) -> bool:
    return any(isinstance(value, str) for value in series.head(20).tolist())
