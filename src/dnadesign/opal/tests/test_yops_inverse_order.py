"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_yops_inverse_order.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np
from pydantic import BaseModel

from dnadesign.opal.src.config.types import PluginRef
from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.registries.transforms_y import list_y_ops, register_y_op, run_y_ops_pipeline


class _Params(BaseModel):
    pass


def _register_test_ops() -> None:
    if "test_add1" not in list_y_ops():

        @register_y_op("test_add1")
        def _add1():
            def fit(Y, params, ctx=None):
                return None

            def transform(Y, params, ctx=None):
                return np.asarray(Y, dtype=float) + 1.0

            def inverse(Y, params, ctx=None):
                return np.asarray(Y, dtype=float) - 1.0

            return fit, transform, inverse, _Params

    if "test_mul2" not in list_y_ops():

        @register_y_op("test_mul2")
        def _mul2():
            def fit(Y, params, ctx=None):
                return None

            def transform(Y, params, ctx=None):
                return np.asarray(Y, dtype=float) * 2.0

            def inverse(Y, params, ctx=None):
                return np.asarray(Y, dtype=float) / 2.0

            return fit, transform, inverse, _Params


def test_yops_inverse_order_multiop():
    _register_test_ops()
    Y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    reg = PluginRegistryView("rf", "sfxi_v1", "top_n", "identity", "sfxi_vec8_from_table_v1")
    rctx = RoundCtx(core={"core/round_index": 0}, registry=reg)
    ops = [PluginRef("test_add1", {}), PluginRef("test_mul2", {})]
    Yt = run_y_ops_pipeline(stage="fit_transform", y_ops=ops, Y=Y, ctx=rctx)
    Yr = run_y_ops_pipeline(stage="inverse", y_ops=ops, Y=Yt, ctx=rctx)
    assert np.allclose(Y, Yr)
