"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/transforms_x/identity.py

Feature-transform plugin logic for identity OPAL transforms x.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..core.round_context import PluginCtx, roundctx_contract
from ..registries.transforms_x import register_transform_x


@roundctx_contract(
    category="transform_x",
    requires=[],
    produces=["transform_x/<self>/x_dim"],
)
@register_transform_x("identity")
def _factory(params: Optional[Dict[str, Any]] = None):
    """
    Identity transform for canonical vector cells.
    Inputs per cell must already be vector-like:
      * list/tuple/ndarray/pandas.Series of numbers
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
        as_py = getattr(v, "as_py", None)
        if callable(as_py):
            v = as_py()
        to_pylist = getattr(v, "to_pylist", None)
        if callable(to_pylist):
            v = to_pylist()
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size == 0:
                raise ValueError("identity transform requires non-empty vector cells")
            return arr
        raise ValueError(
            "identity transform requires vector cells; normalize scalar or JSON-string X before campaign execution"
        )

    def _transform(series: pd.Series, ctx: Optional[PluginCtx] = None) -> np.ndarray:
        rows = [_parse_cell(v) for v in series.tolist()]
        lengths = {int(r.size) for r in rows}
        if len(lengths) != 1:
            raise ValueError(f"identity transform requires consistent vector length; saw lengths={sorted(lengths)}")
        width = lengths.pop()
        if ctx is not None:
            ctx.set("transform_x/<self>/x_dim", int(width))
        X = np.vstack([r.reshape(1, width) for r in rows])
        if expected_len is not None and width != expected_len:
            raise ValueError(f"identity transform expected_length={expected_len} but got {width}")
        if not np.all(np.isfinite(X)):
            raise ValueError("identity transform produced non-finite values.")
        return X

    return _transform
