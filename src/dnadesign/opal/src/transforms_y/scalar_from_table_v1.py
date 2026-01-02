"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms_y/scalar_from_table_v1.py

Simple scalar Y ingest from a tidy table.

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..core.round_context import roundctx_contract
from ..registries.transforms_y import register_transform_y


@roundctx_contract(category="transform_y", requires=[], produces=[])
@register_transform_y("scalar_from_table_v1")
def scalar_from_table_v1(
    csv_df: pd.DataFrame,
    params: Dict,
    ctx=None,
) -> pd.DataFrame:
    """
    Input columns:
      - (optional) id           [must be literally named 'id' if present]
      - sequence                [required]
      - y                        [scalar measurement]

    Output:
      DataFrame with either:
        • columns ['id','sequence','y']   when id column is present
        • columns ['sequence','y']        when id column is absent (OPAL will resolve ids by sequence)
    """
    p = params or {}
    id_col = p.get("id_column", None)
    if id_col is not None and id_col != "id":
        raise ValueError("id_column, if set, must be exactly 'id'.")

    seq_col = p.get("sequence_column", "sequence")
    y_col = p.get("y_column", "y")

    missing = [c for c in (seq_col, y_col) if c not in csv_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    y_vals = csv_df[y_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(y_vals)):
        raise ValueError("Scalar Y column must be finite.")

    out = pd.DataFrame(
        {
            "sequence": csv_df[seq_col].astype(str),
            "y": [[float(v)] for v in y_vals],
        }
    )
    if id_col and id_col in csv_df.columns:
        out["id"] = csv_df[id_col].astype(str)

    cols = ["id", "sequence", "y"] if "id" in out.columns else ["sequence", "y"]
    return out[cols]
