"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/ingest.py

Transform tidy inputs -> model-ready labels with a structured preview.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .registries.transforms_y import get_transform_y
from .round_context import PluginCtx
from .utils import OpalError


@dataclass
class IngestPreview:
    # Input overview
    total_rows_in_csv: int
    rows_with_id: int
    rows_with_sequence: int

    # Resolution counts
    resolved_ids_by_sequence: int
    unknown_sequences: int

    # Vector checks
    y_expected_length: Optional[int]
    y_length_ok: int
    y_length_bad: int

    # What we will write
    y_column_name: str
    duplicate_policy: str
    duplicate_key: str
    duplicates_found: int
    duplicates_dropped: int

    # Notes/warnings
    warnings: List[str]


def _vec_len(v: Any) -> int:
    if isinstance(v, list):
        return len(v)
    if isinstance(v, (np.ndarray,)):
        return int(v.shape[-1])
    return 1


def _apply_transform_via_registry(
    name: str,
    params: Dict[str, Any],
    csv_df: pd.DataFrame,
    *,
    ctx: PluginCtx,
) -> pd.DataFrame:
    """
    Call the registered Y-ingest transform. We tolerate (csv_df, params)
    or (csv_df, **params) call patterns for plugin friendliness.
    """
    fn = get_transform_y(name)
    try:
        return fn(csv_df, params, ctx=ctx)  # required signature
    except TypeError as e:
        raise OpalError(
            f"Y transform '{name}' has an unsupported signature. Expected: fn(df_tidy, params: dict, ctx=PluginCtx)."
        ) from e


def run_ingest(
    records_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    *,
    transform_name: str,
    transform_params: Dict[str, Any],
    y_expected_length: Optional[int],
    y_column_name: str,
    duplicate_policy: str,
    ctx: PluginCtx,
) -> Tuple[pd.DataFrame, IngestPreview]:
    """
    Returns:
      labels_df: DataFrame with at least ['sequence','y'] and, where resolvable, 'id'
      preview:   IngestPreview
    """
    if ctx is None:
        raise OpalError("run_ingest requires a PluginCtx for transform_y.")

    # 1) Transform tidy -> (sequence, y)
    labels = _apply_transform_via_registry(transform_name, transform_params, csv_df, ctx=ctx)

    # 1b) Validate Y vectors strictly (no fallbacks)
    for i, v in enumerate(labels["y"].tolist()):
        arr = np.asarray(v, dtype=float).ravel()
        if y_expected_length is not None and arr.size != int(y_expected_length):
            raise OpalError(f"Y length mismatch at row {i}: expected {int(y_expected_length)}, got {int(arr.size)}")
        if not np.all(np.isfinite(arr)):
            raise OpalError(f"Y contains non-finite values at row {i}.")

    # 2) Try to resolve ids by sequence (existing rows only; new rows remain without id for now)
    seq2id = {}
    if "sequence" in records_df.columns and "id" in records_df.columns:
        seq2id = records_df.drop_duplicates(subset=["sequence"]).set_index("sequence")["id"].astype(str).to_dict()
    if "id" not in labels.columns:
        labels["id"] = labels["sequence"].map(seq2id)

    # 3) Duplicate handling (assertive, policy-driven)
    policy = str(duplicate_policy or "error").strip().lower()
    if policy not in {"error", "keep_first", "keep_last"}:
        raise OpalError(
            f"Unknown ingest.duplicate_policy={duplicate_policy!r} (expected: error | keep_first | keep_last)."
        )
    # Build a stable key (prefer id if present; else sequence)
    if "id" in labels.columns and labels["id"].notna().any():
        key = labels["id"].astype("string")
        key = key.fillna(labels["sequence"].astype("string"))
        key_name = "id"
    else:
        key = labels["sequence"].astype("string")
        key_name = "sequence"
    if key.isna().any():
        raise OpalError("Ingest requires id or sequence for every row (found missing values).")
    dup_mask = key.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    dropped = 0
    if dup_count > 0:
        dup_keys = key[dup_mask].astype(str).unique().tolist()[:10]
        if policy == "error":
            raise OpalError(f"Duplicate {key_name} values found (sample={dup_keys}).")
        keep = "first" if policy == "keep_first" else "last"
        before = len(labels)
        labels = labels.loc[~key.duplicated(keep=keep)].copy()
        dropped = before - len(labels)

    # 4) Preview stats
    total = int(len(csv_df))
    rows_with_id = int("id" in csv_df.columns)
    rows_with_seq = int("sequence" in csv_df.columns)

    resolved = int(labels["id"].notna().sum())
    unknown = int((~labels["id"].notna()).sum())

    # vector length checks (full, already validated above)
    y_len_ok = 0
    y_len_bad = 0
    if y_expected_length:
        for v in labels["y"].tolist():
            if _vec_len(v) == y_expected_length:
                y_len_ok += 1
            else:
                y_len_bad += 1

    # Logic bounds warning (best-effort): try to read first 4 entries as logic
    warnings: List[str] = []
    try:
        v = np.asarray([yy[:4] for yy in labels["y"].tolist()], dtype=float)
        if (v < -1e-9).any() or (v > 1 + 1e-9).any():
            warnings.append("logic_out_of_bounds_detected")
    except Exception:
        pass

    preview = IngestPreview(
        total_rows_in_csv=total,
        rows_with_id=rows_with_id,
        rows_with_sequence=rows_with_seq,
        resolved_ids_by_sequence=resolved,
        unknown_sequences=unknown,
        y_expected_length=y_expected_length,
        y_length_ok=y_len_ok,
        y_length_bad=y_len_bad,
        y_column_name=y_column_name,
        duplicate_policy=policy,
        duplicate_key=key_name,
        duplicates_found=dup_count,
        duplicates_dropped=dropped,
        warnings=warnings,
    )

    # 5) Return labels (sequence, id?, y) and the preview
    cols = ["sequence", "id", "y"] if "id" in labels.columns else ["sequence", "y"]
    return labels[cols], preview
