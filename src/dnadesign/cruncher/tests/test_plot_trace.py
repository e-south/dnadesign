"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_plot_trace.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from dnadesign.cruncher.workflows.analyze.plots.diagnostics import plot_trace


def test_plot_trace_constant_score(tmp_path: Path) -> None:
    idata = az.from_dict(posterior={"score": np.ones((2, 10))})
    out_dir = tmp_path / "plots"
    plot_trace(idata, out_dir)
    assert (out_dir / "diag__trace_score.png").exists()
