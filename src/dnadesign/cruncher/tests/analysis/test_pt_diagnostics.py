"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_pt_diagnostics.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics


def test_pt_diagnostics_uses_cold_chain_only() -> None:
    score = np.random.default_rng(0).normal(size=(3, 10))
    trace_idata = az.from_dict(posterior={"score": score})
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=trace_idata,
        sequences_df=pd.DataFrame({"sequence": []}),
        elites_df=pd.DataFrame(),
        tf_names=["tf1"],
        optimizer_kind="pt",
        optimizer={},
        optimizer_stats={},
    )
    metrics = diagnostics["metrics"]["trace"]
    assert metrics.get("rhat") is None
