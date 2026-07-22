"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/ingest_y/metadata.py

Required metadata handling for new `opal ingest-y` rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from ....core.utils import OpalError
from .values import infer_default, is_missing_value


def missing_required_values_for_unknown(
    *,
    csv_df: pd.DataFrame,
    unknown_sequences: list[str],
    required_cols: list[str],
    x_column_name: str,
) -> list[str]:
    if not unknown_sequences:
        return []
    if "sequence" not in csv_df.columns:
        raise OpalError("Input missing sequence column; cannot validate required metadata for new sequences.")
    csv_unknown_mask = csv_df["sequence"].astype(str).isin(unknown_sequences)
    missing_cols: list[str] = []
    for column in required_cols:
        if column == x_column_name:
            continue
        if column in csv_df.columns:
            missing_mask = csv_df[column].map(is_missing_value).fillna(True) & csv_unknown_mask
            if missing_mask.any():
                missing_cols.append(column)
    return missing_cols


def metadata_defaults_for_missing(*, records_df: pd.DataFrame, missing_set: set[str]) -> dict[str, str]:
    defaults: dict[str, str] = {}
    if "bio_type" in missing_set:
        defaults["bio_type"] = infer_default(records_df.get("bio_type", pd.Series(dtype=object)))
    if "alphabet" in missing_set:
        defaults["alphabet"] = infer_default(records_df.get("alphabet", pd.Series(dtype=object)))
    if any(value is None for value in defaults.values()):
        raise OpalError(
            "Missing required metadata for new sequences, and defaults could not be inferred. "
            "Provide the missing columns or use --unknown-sequences drop."
        )
    return defaults


def apply_defaults_for_unknown(
    *,
    csv_df: pd.DataFrame,
    unknown_sequences: list[str],
    defaults: dict[str, str],
) -> None:
    if not defaults:
        return
    if not unknown_sequences:
        for column, value in defaults.items():
            if column not in csv_df.columns:
                csv_df[column] = value
        return
    if "sequence" not in csv_df.columns:
        raise OpalError("Input missing sequence column; cannot apply defaults for new sequences.")
    csv_unknown_mask = csv_df["sequence"].astype(str).isin(unknown_sequences)
    for column, value in defaults.items():
        if column not in csv_df.columns:
            csv_df[column] = value
            continue
        missing_mask = csv_df[column].map(is_missing_value).fillna(True) & csv_unknown_mask
        if missing_mask.any():
            csv_df.loc[missing_mask, column] = value
