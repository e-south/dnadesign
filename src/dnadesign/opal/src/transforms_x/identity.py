"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms_x/identity.py

A permissive identity transform:
- Accepts scalar, 1-D, or nested 2-D-with-single-row inputs per record.
- Coerces to a 1-D list[float] with all finite values.
- Rejects non-numeric/empty inputs with an informative ValueError.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..registries.transforms_x import register_transform_x


@register_transform_x("identity")
def _factory(params: Optional[Dict[str, Any]] = None):
    """
    Identity transform — pass-through with robust parsing.
    Inputs per cell may be:
      * scalar number
      * list/tuple/ndarray/pandas.Series of numbers
      * JSON string "[...]" of numbers
    Output:
      * np.ndarray shape (N,F) with dtype=float
    Optional params:
      * expected_length: int — assert all rows have this width
    """
    expected_len = None
    if params:
        expected_len = params.get("expected_length")
        if expected_len is not None:
            expected_len = int(expected_len)

    def _parse_cell(v: Any) -> np.ndarray:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            raise ValueError("X cell is null/NaN")
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(v, dtype=float).ravel()
            return arr
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = np.asarray(json.loads(s), dtype=float).ravel()
                except Exception as e:
                    raise ValueError(f"Invalid JSON array in X cell: {s[:48]}…") from e
                return arr
            # scalar-like string
            return np.asarray([float(s)], dtype=float)
        # numeric scalar
        return np.asarray([float(v)], dtype=float)

    def _transform(series: pd.Series) -> np.ndarray:
        rows = [_parse_cell(v) for v in series.tolist()]
        lengths = {int(r.size) for r in rows}
        if len(lengths) != 1:
            raise ValueError(
                f"identity transform requires consistent vector length; saw lengths={sorted(lengths)}"
            )
        width = lengths.pop()
        X = np.vstack([r.reshape(1, width) for r in rows])
        if expected_len is not None and width != expected_len:
            raise ValueError(
                f"identity transform expected_length={expected_len} but got {width}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("identity transform produced non-finite values.")
        return X

    return _transform
