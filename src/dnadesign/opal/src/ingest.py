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


def _transform_vec8_from_table(
    csv_df: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    seq_col = params.get("sequence_column", "sequence")
    logic_cols = params.get("logic_columns", ["v00", "v10", "v01", "v11"])
    inten_cols = params.get(
        "intensity_columns", ["y00_star", "y10_star", "y01_star", "y11_star"]
    )

    need = [seq_col, *logic_cols, *inten_cols]
    missing = [c for c in need if c not in csv_df.columns]
    if missing:
        raise OpalError(f"Input CSV missing columns: {missing}")

    df = csv_df.copy()
    # bounds check for logic in [0,1]
    logic = df[logic_cols].to_numpy(dtype=float)
    if np.any(~np.isfinite(logic)):
        raise OpalError("Non-finite values in logic columns.")
    if (logic < -1e-9).any() or (logic > 1 + 1e-9).any():
        # don't hard fail; log later in preview
        pass

    inten = df[inten_cols].to_numpy(dtype=float)
    if np.any(~np.isfinite(inten)):
        raise OpalError("Non-finite values in intensity columns.")

    y = np.hstack([logic, inten]).tolist()
    out = pd.DataFrame({"sequence": df[seq_col].astype(str).tolist(), "y": y})
    return out


def _apply_transform(
    name: str, params: Dict[str, Any], csv_df: pd.DataFrame
) -> pd.DataFrame:
    # For now, support the shipped demo transform name explicitly.
    if name == "sfxi_vec8_from_table_v1":
        return _transform_vec8_from_table(csv_df, params or {})
    # If you have additional transforms registered via a registry, you can wire them here.
    raise OpalError(f"Unknown Y transform: {name!r}")


def run_ingest(
    records_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    *,
    transform_name: str,
    transform_params: Dict[str, Any],
    y_expected_length: Optional[int],
    setpoint_vector: Optional[List[float]] = None,  # kept for future previews
    y_column_name: str,
) -> Tuple[pd.DataFrame, IngestPreview]:
    """
    Returns:
      labels_df: DataFrame with at least ['sequence','y'] and, where resolvable, 'id'
      preview:   IngestPreview
    """
    # 1) Transform tidy -> (sequence, y)
    labels = _apply_transform(transform_name, transform_params, csv_df)

    # 2) Try to resolve ids by sequence (existing rows only; new rows remain without id for now)
    seq2id = {}
    if "sequence" in records_df.columns and "id" in records_df.columns:
        seq2id = (
            records_df.drop_duplicates(subset=["sequence"])
            .set_index("sequence")["id"]
            .astype(str)
            .to_dict()
        )
    labels["id"] = labels["sequence"].map(seq2id)

    # 3) Preview stats
    total = int(len(csv_df))
    rows_with_id = int("id" in csv_df.columns)
    rows_with_seq = int("sequence" in csv_df.columns)

    resolved = int(labels["id"].notna().sum())
    unknown = int((~labels["id"].notna()).sum())

    # vector length checks (best-effort)
    y_len_ok = 0
    y_len_bad = 0
    if y_expected_length:
        for v in labels["y"].head(200).tolist():  # sample
            if _vec_len(v) == y_expected_length:
                y_len_ok += 1
            else:
                y_len_bad += 1

    # Logic bounds warning (best-effort)
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
    return labels[["sequence", "id", "y"]], preview
