"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/ingest_y/x_checks.py

X-column checks for new `opal ingest-y` rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from ....core.utils import OpalError
from ....runtime.ingest import IngestPreview
from .values import col_has_str, col_is_listlike, is_missing_value


def raise_on_stringified_list_x(
    *,
    records_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    preview: IngestPreview,
    unknown_sequences_policy: str,
    required_cols: list[str],
    x_column_name: str,
) -> None:
    if preview.unknown_sequences and unknown_sequences_policy == "create" and x_column_name in required_cols:
        if x_column_name in records_df.columns and x_column_name in csv_df.columns:
            if col_is_listlike(records_df[x_column_name]) and col_has_str(csv_df[x_column_name]):
                raise OpalError(
                    f"Column '{x_column_name}' appears list-valued in records.parquet, but CSV values are strings. "
                    "Use Parquet input (or a true list column) when adding new sequences with X."
                )


def raise_on_unknown_missing_x(
    *,
    csv_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    unknown_mask: pd.Series,
    x_column_name: str,
) -> None:
    missing_x_mask = pd.Series(False, index=labels_df.index)
    if x_column_name not in csv_df.columns:
        missing_x_mask = unknown_mask.copy()
    else:
        missing_x_values = csv_df[x_column_name].map(is_missing_value).fillna(True)
        seq_has_x = pd.Series(False, index=labels_df.index)
        id_has_x = pd.Series(False, index=labels_df.index)
        if "sequence" in labels_df.columns and "sequence" in csv_df.columns:
            seq_with_x = set(
                csv_df.loc[
                    ~missing_x_values & csv_df["sequence"].notna(),
                    "sequence",
                ]
                .astype(str)
                .tolist()
            )
            seq_series = labels_df["sequence"]
            seq_has_x = seq_series.notna() & seq_series.astype(str).isin(seq_with_x)
        if "id" in labels_df.columns and "id" in csv_df.columns:
            id_with_x = set(csv_df.loc[~missing_x_values & csv_df["id"].notna(), "id"].astype(str).tolist())
            id_series = labels_df["id"]
            id_has_x = id_series.notna() & id_series.astype(str).isin(id_with_x)
        if not seq_has_x.any() and not id_has_x.any():
            missing_x_mask = unknown_mask.copy()
        else:
            missing_x_mask = ~(seq_has_x | id_has_x) & unknown_mask
    missing_x_count = int(missing_x_mask.sum())
    if missing_x_count > 0:
        raise OpalError(
            f"{missing_x_count} unknown sequences are missing required X column '{x_column_name}'. "
            "Provide X values for new rows or use --unknown-sequences drop to skip unknown rows."
        )
