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
from typing import Any

import numpy as np
import pandas as pd


def _parse_string_vector(s: str) -> list[float]:
    """
    Accepts JSON-like '[1, 2, 3]' strings. We *do not* accept arbitrary CSV-like
    strings to avoid silent mis-parsing. Raise on bad tokens.
    """
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError("string input must be a JSON array like '[1,2,3]'")
    try:
        arr = json.loads(s)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"failed to parse JSON array from string: {e}") from e
    if not isinstance(arr, list):
        raise ValueError("parsed JSON value is not a list")
    try:
        return [float(x) for x in arr]
    except Exception as e:
        raise ValueError(f"vector contains non-numeric values: {e}") from e


def _as_1d_vector(x: Any) -> list[float]:
    """
    Coerce per-row X into a 1-D list[float].
    Accepts: float/int/np scalar => [x]
             list/tuple/np.ndarray/pd.Series => flattened 1-D
             nested 2-D (1, d) => flattened
             string "[...]" => parsed JSON list
    Enforces: len(vec) >= 1, all finite.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("X is missing (None/NaN)")

    if isinstance(x, (int, float, np.floating, np.integer)):
        vec = [float(x)]
    elif isinstance(x, str):
        vec = _parse_string_vector(x)
    elif isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:  # scalar-like
            vec = [float(arr)]
        elif arr.ndim == 1:
            vec = arr.tolist()
        elif arr.ndim == 2 and arr.shape[0] == 1:
            vec = arr.ravel().tolist()
        else:
            # We keep the transform *per-row*, so higher ranks are not allowed.
            raise ValueError(
                f"X has unsupported shape {arr.shape}; expected scalar or 1-D"
            )
    else:
        # Last-chance scalar coercion
        try:
            vec = [float(x)]
        except Exception as e:  # pragma: no cover
            raise ValueError(f"X value is not numeric: {type(x)}") from e

    if len(vec) == 0:
        raise ValueError("X vector is empty")
    if not np.all(np.isfinite(vec)):
        raise ValueError("X vector contains non-finite values (NaN/Inf)")

    return [float(v) for v in vec]


def identity_transform(value: Any, *, params: dict | None = None) -> list[float]:
    """
    Transform *one* row's X “as-is”, normalized to a 1-D list[float].
    This is intentionally permissive for pragmatic usability.
    """
    return _as_1d_vector(value)
