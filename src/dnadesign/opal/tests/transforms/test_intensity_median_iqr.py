"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/transforms/test_intensity_median_iqr.py

Tests intensity_median_iqr Y-op transform and uncertainty inversion semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal.src.transforms_y.intensity_median_iqr import (
    _fit,
    _inverse_std,
    _Params,
    _transform,
)


class _Ctx:
    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value


def test_intensity_median_iqr_inverse_std_scales_intensity_channels() -> None:
    params = _Params(min_labels=1, eps=1e-8)
    Y = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 4.0, 8.0],
            [0.2, 0.3, 0.4, 0.5, 3.0, 4.0, 8.0, 12.0],
            [0.3, 0.4, 0.5, 0.6, 5.0, 8.0, 10.0, 14.0],
        ],
        dtype=float,
    )
    ctx = _Ctx()
    _fit(Y, params, ctx=ctx)
    scale = np.asarray(ctx.get("yops/intensity_median_iqr/scale"), dtype=float)

    y_std = np.full((2, 8), 0.5, dtype=float)
    out = _inverse_std(y_std, params, ctx=ctx)
    np.testing.assert_allclose(out[:, :4], y_std[:, :4], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out[:, 4:8], y_std[:, 4:8] * scale[None, :], rtol=1e-12, atol=1e-12)


def test_intensity_median_iqr_transform_rejects_missing_fit_state() -> None:
    params = _Params(min_labels=1, eps=1e-8)
    Y = np.ones((2, 8), dtype=float)
    ctx = _Ctx()
    ctx.set("yops/intensity_median_iqr/enabled", True)

    with pytest.raises(ValueError, match="missing required context key"):
        _ = _transform(Y, params, ctx=ctx)


def test_intensity_median_iqr_inverse_std_rejects_nonpositive_scale() -> None:
    params = _Params(min_labels=1, eps=1e-8)
    y_std = np.full((2, 8), 0.5, dtype=float)
    ctx = _Ctx()
    ctx.set("yops/intensity_median_iqr/enabled", True)
    ctx.set("yops/intensity_median_iqr/scale", [1.0, 0.0, 1.0, 1.0])
    ctx.set("yops/intensity_median_iqr/center", [0.0, 0.0, 0.0, 0.0])
    ctx.set("yops/intensity_median_iqr/eps", 1e-8)

    with pytest.raises(ValueError, match="positive finite"):
        _ = _inverse_std(y_std, params, ctx=ctx)
