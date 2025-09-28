"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/preflight.py

Preflight checks prior to training/scoring a round.

- Validates essentials & safety constraints.
- Determines X dimension robustly (without assuming transform meta type).
- Optionally performs label-history backfill (hooked; default off in this file).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .data_access import ESSENTIAL_COLS, RecordsStore
from .utils import OpalError


@dataclass
class PreflightReport:
    """Result of preflight checks."""

    x_dim: int
    backfill: Dict[str, Any]


def _first_non_null(series: pd.Series) -> Optional[Any]:
    """Return the first non-null value in a Series, or None."""
    if series is None:
        return None
    mask = series.notna()
    if hasattr(mask, "any") and mask.any():
        return series[mask].iloc[0]
    return None


def _infer_dim_from_x_cell(xcell: Any) -> int:
    """
    Try to infer dimensionality from a single X-cell that is expected to be
    list/tuple/np.ndarray-like.
    """
    if xcell is None:
        return 0
    if isinstance(xcell, (list, tuple, np.ndarray, pd.Series)):
        try:
            arr = np.asarray(xcell, dtype=float).ravel()
            return int(arr.size)
        except Exception:
            return 0
    # JSON string like "[...]"?
    if isinstance(xcell, str):
        s = xcell.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parts = (
                    []
                    if s[1:-1].strip() == ""
                    else [p.strip() for p in s[1:-1].split(",")]
                )
                arr = np.asarray([float(p) for p in parts], dtype=float)
                return int(arr.size)
            except Exception:
                return 0
    # scalars not supported for multi-dim X
    return 0


def _determine_x_dim_safely(
    store: RecordsStore, df: pd.DataFrame, sample_ids: Iterable[str]
) -> int:
    """
    Call the transform in a small sample and try safe ways to get x_dim.
    Do not assume any particular meta type from the transform.
    """
    sample_ids = list(sample_ids)
    if sample_ids:
        try:
            X, _ = store.transform_matrix(df, sample_ids)
            if hasattr(X, "shape") and X.ndim == 2 and X.shape[0] > 0:
                return int(X.shape[1])
        except OpalError:
            # Re-raise OpalError verbatim — it's already user-focused.
            raise
        except Exception as e:
            raise OpalError(
                f"Failed to determine X dimension using transform '{store.x_transform_name}': {e}"
            ) from e

        # Fallback to first non-null X-cell among the sample
        xcell = _first_non_null(
            df.loc[df["id"].astype(str).isin(sample_ids), store.x_col]
        )
        dim = _infer_dim_from_x_cell(xcell)
        if dim > 0:
            return dim

    # Global fallback: try to infer from the whole column
    xcell = _first_non_null(df.get(store.x_col, pd.Series(dtype=object)))
    return _infer_dim_from_x_cell(xcell)


def preflight_run(
    store: RecordsStore,
    df: pd.DataFrame,
    round_index: int,
    fail_on_mixed_biotype_or_alphabet: bool = True,
    auto_backfill: bool = True,
) -> PreflightReport:
    """
    Perform preflight checks and return a typed report.

    Notes:
    - This function is intentionally conservative and does not write by default.
      If you choose to enable actual backfill writes here, return 'backfilled' > 0.
    """
    # 1) Essentials present
    missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
    if missing:
        raise OpalError(f"Missing essential columns: {missing}")

    # 2) X column present
    if store.x_col not in df.columns:
        raise OpalError(f"Missing X column: {store.x_col}")

    # 3) Optional safety: biotype/alphabet uniformity
    if fail_on_mixed_biotype_or_alphabet:
        for col in ("bio_type", "alphabet"):
            if col in df.columns:
                vals = sorted({str(v) for v in df[col].dropna().unique().tolist()})
                if len(vals) > 1:
                    raise OpalError(
                        f"Mixed '{col}' detected: {vals}. "
                        "Set safety.fail_on_mixed_biotype_or_alphabet=false to bypass."
                    )

    # 4) Determine X dimension robustly using a small sample (<= 128 ids)
    ids_series = (
        df["id"].astype(str) if "id" in df.columns else pd.Series([], dtype=str)
    )
    sample_ids = ids_series.head(128).tolist()
    x_dim = _determine_x_dim_safely(store, df, sample_ids)

    if x_dim <= 0:
        # Give a very explicit, helpful error
        raise OpalError(
            "Unable to infer X dimensionality. "
            f"X column='{store.x_col}' must contain fixed-length numeric vectors."
        )

    # 5) (Hook) Backfill legacy label history if desired.
    # For now we only *report*; we do not mutate df here (robust default).
    backfill_report = {
        "checked": True,
        "backfilled": 0,
        "details": [],  # if you implement actual backfills, put id samples here
    }
    # If at some point you enable writes here, remember to persist via store.save_atomic(df2).

    return PreflightReport(x_dim=int(x_dim), backfill=backfill_report)
