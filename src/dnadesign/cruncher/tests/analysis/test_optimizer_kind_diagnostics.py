"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_optimizer_kind_diagnostics.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics


def test_diagnostics_rejects_non_gibbs_optimizer_kind() -> None:
    score = np.random.default_rng(0).normal(size=(3, 10))
    trace_idata = az.from_dict(posterior={"score": score})
    with pytest.raises(ValueError, match="optimizer kind.*gibbs_anneal"):
        summarize_sampling_diagnostics(
            trace_idata=trace_idata,
            sequences_df=pd.DataFrame({"sequence": []}),
            elites_df=pd.DataFrame(),
            elites_hits_df=None,
            tf_names=["tf1"],
            optimizer_kind="pt",
            optimizer={},
            optimizer_stats={},
        )
