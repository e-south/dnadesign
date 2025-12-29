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

    # Notes/warnings
    warnings: List[str]


def _vec_len(v: Any) -> int:
    if isinstance(v, list):
        return len(v)
    if isinstance(v, (np.ndarray,)):
        return int(v.shape[-1])
    return 1


def _apply_transform_via_registry(name: str, params: Dict[str, Any], csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Call the registered Y-ingest transform. We tolerate (csv_df, params)
    or (csv_df, **params) call patterns for plugin friendliness.
    """
    fn = get_transform_y(name)
    try:
        return fn(csv_df, params)  # fn(df, params)
    except TypeError:
        try:
            return fn(csv_df, **(params or {}))  # fn(df, **params)
        except TypeError as e:
            raise OpalError(f"Y transform '{name}' has an unsupported signature.") from e


def run_ingest(
    records_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    *,
    transform_name: str,
    transform_params: Dict[str, Any],
    y_expected_length: Optional[int],
    y_column_name: str,
) -> Tuple[pd.DataFrame, IngestPreview]:
    """
    Returns:
      labels_df: DataFrame with at least ['sequence','y'] and, where resolvable, 'id'
      preview:   IngestPreview
    """
    # 1) Transform tidy -> (sequence, y)
    labels = _apply_transform_via_registry(transform_name, transform_params, csv_df)

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

    # 3) Preview stats
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
        warnings=warnings,
    )

    # 4) Return labels (sequence, id?, y) and the preview
    cols = ["sequence", "id", "y"] if "id" in labels.columns else ["sequence", "y"]
    return labels[cols], preview
