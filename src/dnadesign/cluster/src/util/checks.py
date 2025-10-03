"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/util/checks.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd


class ClusterError(RuntimeError):
    pass


def assert_no_duplicate_ids(
    df: pd.DataFrame, key_col: str, policy: str = "error"
) -> pd.DataFrame:
    if key_col not in df.columns:
        raise ClusterError(f"Required key column '{key_col}' is missing.")
    dupe_mask = df.duplicated(subset=[key_col], keep=False)
    if not dupe_mask.any():
        return df
    if policy == "error":
        dupes = df.loc[dupe_mask, key_col].value_counts().head(10).index.tolist()
        raise ClusterError(
            f"Found duplicate values in '{key_col}' (e.g., {dupes[:5]}...).\n"
            f"Choose a dedupe policy: --dedupe-policy keep-first|keep-last."
        )
    keep = "first" if policy == "keep-first" else "last"
    return df.drop_duplicates(subset=[key_col], keep=keep).copy()


def assert_id_sequence_bijection(
    df: pd.DataFrame,
    id_col: str = "id",
    seq_col: str = "sequence",
    bio_type_col: str | None = "bio_type",
) -> None:
    if id_col not in df.columns or seq_col not in df.columns:
        return
    # Case-insensitive comparison for sequences
    if bio_type_col and bio_type_col in df.columns:
        # Key combines bio_type (lower) + sequence (upper) to avoid false positives across types
        upper = (
            df[bio_type_col].astype(str).str.lower()
            + "|"
            + df[seq_col].astype(str).str.upper()
        )
    else:
        upper = df[seq_col].astype(str).str.upper()
    id_to_seq = df.groupby(id_col)[seq_col].nunique()
    if (id_to_seq > 1).any():
        bad = id_to_seq[id_to_seq > 1].head(5).index.tolist()
        raise ClusterError(
            f"Each id must map to exactly one sequence (case-insensitive). Violations for ids: {bad}. "
            f"Hint: run 'usr validate <dataset> --strict' and, if needed, 'usr dedupe-sequences <dataset>'."
        )
    seq_to_id = df.assign(_u=upper).groupby("_u")[id_col].nunique()
    if (seq_to_id > 1).any():
        bad = seq_to_id[seq_to_id > 1].head(5).index.tolist()
        raise ClusterError(
            "Each sequence must map to exactly one id (case-insensitive). "
            f"Violations for sequences (uppercased; bio_type|SEQ): {bad[:3]}. "
            f"Hint: run 'usr validate <dataset> --strict' and then 'usr dedupe-sequences <dataset>' to repair."
        )
