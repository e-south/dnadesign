"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_yops_intensity_scalar.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np

from dnadesign.opal.src.config.types import PluginRef
from dnadesign.opal.src.registries.transforms_y import run_y_ops_pipeline
from dnadesign.opal.src.round_context import PluginRegistryView, RoundCtx


def test_intensity_median_iqr_round_trip():
    Y = np.array([[0, 0, 0, 0, 1, 2, 3, 4], [0, 0, 0, 0, 5, 6, 7, 8]], dtype=float)
    reg = PluginRegistryView(
        "rf", "sfxi_v1", "top_n", "identity", "sfxi_vec8_from_table_v1"
    )
    rctx = RoundCtx(core={"core/round_index": 0}, registry=reg)
    ops = [PluginRef("intensity_median_iqr", {"min_labels": 1})]
    Yt = run_y_ops_pipeline(stage="fit_transform", y_ops=ops, Y=Y, ctx=rctx)
    Yr = run_y_ops_pipeline(stage="inverse", y_ops=ops, Y=Yt, ctx=rctx)
    assert np.allclose(Y, Yr)
