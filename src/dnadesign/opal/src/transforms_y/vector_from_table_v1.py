"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/transforms_y/vector_from_table_v1.py

Label-transform plugin logic for vector from table v1 OPAL transforms y.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..core.round_context import roundctx_contract
from ..registries.transforms_y import register_transform_y


@roundctx_contract(category="transform_y", requires=[], produces=[])
@register_transform_y("vector_from_table_v1")
def vector_from_table_v1(
    csv_df: pd.DataFrame,
    params: Dict,
    ctx=None,
) -> pd.DataFrame:
    """
    Input columns:
      - (optional) id column    [name via params.id_column, or defaults to 'id' if present]
      - sequence                [required only when id is absent]
      - value_columns           [finite numeric target channels]

    Output:
      DataFrame with either:
        • columns ['id','sequence','y']   when id and sequence are present
        • columns ['id','y']              when id is present and sequence omitted
        • columns ['sequence','y']        when id is absent
    """
    p = params or {}
    id_col = p.get("id_column", None)
    if id_col is None and "id" in csv_df.columns:
        id_col = "id"
    seq_col = str(p.get("sequence_column", "sequence"))
    value_cols: List[str] = [str(col) for col in p.get("value_columns", [])]
    if not value_cols:
        raise ValueError("value_columns must contain at least one target column.")
    if len(set(value_cols)) != len(value_cols):
        raise ValueError("value_columns must not contain duplicates.")

    has_id = id_col is not None
    need = set(value_cols)
    if has_id:
        need.add(str(id_col))
    else:
        need.add(seq_col)
    missing = [column for column in sorted(need) if column not in csv_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    values = csv_df[value_cols].to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] != len(value_cols):
        raise ValueError("value_columns did not produce a rectangular numeric target matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Vector Y columns must be finite.")

    out = pd.DataFrame({"y": [row.astype(float).tolist() for row in values]})
    if has_id:
        id_series = csv_df[str(id_col)]
        if id_series.isna().any() or id_series.astype(str).str.strip().eq("").any():
            raise ValueError("id column contains null/empty values.")
        out["id"] = id_series.astype(str).str.strip()
    if seq_col in csv_df.columns:
        seq_series = csv_df[seq_col]
        if seq_series.isna().any() or seq_series.astype(str).str.strip().eq("").any():
            raise ValueError("sequence column contains null/empty values.")
        out["sequence"] = seq_series.astype(str).str.strip()
    elif not has_id:
        raise ValueError("sequence column is required when id_column is absent.")

    cols = [column for column in ("id", "sequence", "y") if column in out.columns]
    return out[cols]
