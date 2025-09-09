"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms/identity.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pandas as pd

from ..utils import ExitCodes, OpalError


class IdentityTransform:
    """
    Identity transform (stateless). Accepts:
      - Arrow/Pandas list<float> per row
      - JSON-like string "[...]" per row
    Emits dense float32 matrix (n_samples, d) with fixed d.
    """

    def __init__(self, **_: object) -> None:
        pass

    def _coerce_row(self, v) -> list[float]:
        if isinstance(v, (list, tuple)):
            try:
                return [float(x) for x in v]
            except Exception as e:
                raise OpalError(f"Non-numeric item in representation row: {v}") from e
        if isinstance(v, str):
            try:
                arr = json.loads(v)
                if not isinstance(arr, list):
                    raise ValueError("JSON not a list")
                return [float(x) for x in arr]
            except Exception as e:
                raise OpalError(
                    f"Failed to parse JSON array in representation: {v[:80]}"
                ) from e
        raise OpalError(
            f"Unsupported representation item type: {type(v).__name__}",
            ExitCodes.CONTRACT_VIOLATION,
        )

    def fit_transform(self, series: pd.Series) -> Tuple[np.ndarray, int]:
        # For v1, stateless => fit_transform == transform
        return self.transform(series)

    def transform(self, series: pd.Series) -> Tuple[np.ndarray, int]:
        rows = [self._coerce_row(v) for v in series.tolist()]
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), 0
        d = len(rows[0])
        for idx, r in enumerate(rows):
            if len(r) != d:
                raise OpalError(
                    f"Ragged dimensions in representation column at row {idx}: expected {d}, got {len(r)}"
                )
        mat = np.asarray(rows, dtype=np.float32)
        if mat.ndim != 2:
            raise OpalError("Representation coercion did not produce 2-D matrix")
        return mat, d
