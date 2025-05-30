"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/plots/autocorr.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path

import arviz as az


def plot_autocorr(idata: az.InferenceData, out_dir: Path) -> None:
    """Autocorrelation up to lag=100 → autocorr_score.png"""
    az.plot_autocorr(idata, var_names=["score"], max_lag=100)
    out = out_dir / "autocorr_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)
    import matplotlib.pyplot as plt

    plt.savefig(out, dpi=300)
    plt.close()
