"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms_y/sfxi_vec8_from_table_v1.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..core.round_context import roundctx_contract
from ..registries.transforms_y import register_transform_y


def _clip01(x: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(x, 0.0 + eps, 1.0 - eps)


def _load_reader_delta(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"sfxi_log_json not found: {p}")
    data = json.loads(p.read_text())
    try:
        delta = data["semantics"]["y_star"]["delta"]
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("sfxi_log_json missing semantics.y_star.delta") from exc
    if not np.isfinite(delta) or float(delta) < 0.0:
        raise ValueError(f"sfxi_log_json has invalid delta: {delta}")
    return float(delta)


def _extract_delta_from_csv(csv_df: pd.DataFrame) -> float | None:
    for col in ("intensity_log2_offset_delta", "log2_offset_delta"):
        if col not in csv_df.columns:
            continue
        series = pd.to_numeric(csv_df[col], errors="coerce")
        if series.isna().any():
            raise ValueError(f"{col} contains null/NaN values.")
        uniq = np.unique(series.to_numpy(dtype=float))
        if uniq.size != 1:
            raise ValueError(f"{col} must be constant across all rows.")
        return float(uniq[0])
    return None


@roundctx_contract(category="transform_y", requires=[], produces=[])
@register_transform_y("sfxi_vec8_from_table_v1")
def sfxi_vec8_from_table_v1(
    csv_df: pd.DataFrame,
    params: Dict,
    ctx=None,
) -> pd.DataFrame:
    """
    Input columns:
      - (optional) id column    [name via params.id_column, or defaults to 'id' if present]
      - sequence                [required only when id is absent]
      - v00, v10, v01, v11      [logic, in [0,1]]
      - y00_star..y11_star      [intensity, in log2 space]
      - intensity_log2_offset_delta (optional, constant) OR log2_offset_delta (optional, constant)

    Output:
      DataFrame with either:
        • columns ['id','sequence','y']   when id column is present and sequence provided
        • columns ['id','y']              when id column is present and sequence omitted
        • columns ['sequence','y']        when id column is absent (OPAL will resolve ids by sequence)
    """
    p = params or {}
    id_col = p.get("id_column", None)
    if id_col is None and "id" in csv_df.columns:
        id_col = "id"
    sfxi_log_json = p.get("sfxi_log_json", None)
    enforce_delta_match = bool(p.get("enforce_log2_offset_match", True))
    expected_delta = p.get("expected_log2_offset_delta", None)
    if expected_delta is None:
        expected_delta = p.get("intensity_log2_offset_delta", 0.0)
    seq_col = p.get("sequence_column", "sequence")
    logic_cols: List[str] = p.get("logic_columns", ["v00", "v10", "v01", "v11"])
    inten_cols: List[str] = p.get("intensity_columns", ["y00_star", "y10_star", "y01_star", "y11_star"])
    strict = bool(p.get("strict_bounds", True))
    eps = float(p.get("clip_bounds_eps", 1e-6))

    has_id = id_col is not None
    seq_required = not has_id

    need = set([*logic_cols, *inten_cols])
    if has_id:
        need.add(id_col)
    if seq_required:
        need.add(seq_col)
    missing = [c for c in need if c not in csv_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    seq_present = seq_col in csv_df.columns

    source_delta = _extract_delta_from_csv(csv_df)
    if source_delta is None and sfxi_log_json is not None:
        source_delta = _load_reader_delta(sfxi_log_json)

    if enforce_delta_match:
        if source_delta is None:
            raise ValueError(
                "Delta enforcement enabled but no delta source found. "
                "Provide intensity_log2_offset_delta column or sfxi_log_json."
            )
        if not np.isfinite(expected_delta) or float(expected_delta) < 0.0:
            raise ValueError(f"expected_log2_offset_delta must be >= 0; got {expected_delta}")
        if not np.isclose(source_delta, float(expected_delta), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"SFXI delta mismatch: reader_log={source_delta} vs expected={float(expected_delta)}. "
                "Ensure OPAL objective intensity_log2_offset_delta matches Reader log2_offset_delta."
            )

    def _invalid_required(series: pd.Series) -> pd.Series:
        s = series
        return s.isna() | s.astype(str).str.strip().eq("")

    def _coerce_str(series: pd.Series) -> pd.Series:
        s = series.copy()
        mask = s.isna()
        s = s.astype(object).where(~mask, pd.NA)
        return s.map(lambda v: str(v).strip() if pd.notna(v) else pd.NA)

    id_out = None
    if has_id:
        id_series = csv_df[id_col]
        if _invalid_required(id_series).any():
            raise ValueError("id column contains null/empty values.")
        id_out = _coerce_str(id_series)

    seq_out = None
    if seq_required:
        seq_series = csv_df[seq_col]
        if _invalid_required(seq_series).any():
            raise ValueError("sequence column contains null/empty values.")
        seq_out = _coerce_str(seq_series)
    elif seq_present:
        seq_out = _coerce_str(csv_df[seq_col])

    # Logic in [0,1]
    L = csv_df[logic_cols].to_numpy(dtype=float)
    if strict:
        if np.any(~np.isfinite(L)) or np.any(L < 0.0 - 1e-12) or np.any(L > 1.0 + 1e-12):
            raise ValueError("Logic columns must be finite and in [0,1].")
        Lc = np.clip(L, 0.0, 1.0)
    else:
        Lc = _clip01(L, eps)

    # Intensities: already in log2, just coerce to float
    Ystar = csv_df[inten_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(Ystar)):
        raise ValueError("Intensity (log2) columns must be finite.")

    vec8 = np.concatenate([Lc, Ystar], axis=1).tolist()

    out = pd.DataFrame({"y": vec8})
    if seq_out is not None:
        out["sequence"] = seq_out
    if id_out is not None:
        out["id"] = id_out

    # Reorder columns for downstream convenience (dedup is handled by ingest policy)
    cols = []
    if "id" in out.columns:
        cols.append("id")
    if "sequence" in out.columns:
        cols.append("sequence")
    cols.append("y")
    return out[cols]
