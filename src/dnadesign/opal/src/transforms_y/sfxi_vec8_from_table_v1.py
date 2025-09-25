"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms_y/sfxi_vec8_from_table_v1.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..registries.transforms_y import register_ingest_transform


def _clip01(x: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(x, 0.0 + eps, 1.0 - eps)


@register_ingest_transform("sfxi_vec8_from_table_v1")
def sfxi_vec8_from_table_v1(
    csv_df: pd.DataFrame,
    params: Dict,
    setpoint_vector: List[float],
    records_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Input columns:
      - (optional) id           [must be literally named 'id' if present]
      - sequence                [required when id is absent]
      - v00, v10, v01, v11      [logic, in [0,1]]
      - y00_star..y11_star      [intensity, in log2 space]

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
    logic_cols = p.get("logic_columns", ["v00", "v10", "v01", "v11"])
    inten_cols = p.get(
        "intensity_columns", ["y00_star", "y10_star", "y01_star", "y11_star"]
    )
    strict = bool(p.get("strict_bounds", True))
    eps = float(p.get("clip_bounds_eps", 1e-6))

    need = set([seq_col, *logic_cols, *inten_cols])
    missing = [c for c in need if c not in csv_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Logic in [0,1]
    L = csv_df[logic_cols].to_numpy(dtype=float)
    if strict:
        if (
            np.any(~np.isfinite(L))
            or np.any(L < 0.0 - 1e-12)
            or np.any(L > 1.0 + 1e-12)
        ):
            raise ValueError("Logic columns must be finite and in [0,1].")
        Lc = np.clip(L, 0.0, 1.0)
    else:
        Lc = _clip01(L, eps)

    # Intensities: already in log2, just coerce to float
    Ystar = csv_df[inten_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(Ystar)):
        raise ValueError("Intensity (log2) columns must be finite.")

    vec8 = np.concatenate([Lc, Ystar], axis=1).tolist()

    out = pd.DataFrame({"sequence": csv_df[seq_col].astype(str), "y": vec8})
    if id_col and id_col in csv_df.columns:
        out["id"] = csv_df[id_col].astype(str)

    # Deduplicate by (id) or sequence — keep the first occurrence
    if "id" in out.columns:
        out = out.drop_duplicates(subset=["id"], keep="first")
    else:
        out = out.drop_duplicates(subset=["sequence"], keep="first")

    # Reorder columns for downstream convenience
    cols = ["id", "sequence", "y"] if "id" in out.columns else ["sequence", "y"]
    return out[cols]
