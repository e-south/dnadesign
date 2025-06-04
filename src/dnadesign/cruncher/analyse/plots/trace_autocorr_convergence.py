"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/analyse/plots/trace_autocorr_convergence.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns


def plot_trace(idata: az.InferenceData, out_dir: Path) -> None:
    """
    One-line-per-chain trace + density → trace_score.png
    """
    out = out_dir / "trace_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    _ = az.plot_trace(idata, var_names=["score"])
    fig = plt.gcf()
    for ax in fig.axes:
        sns.despine(ax=ax)

    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_autocorr(idata: az.InferenceData, out_dir: Path, max_lag: int = 100) -> None:
    """
    Autocorrelation up to lag=max_lag → autocorr_score.png
    """
    out = out_dir / "autocorr_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)
    az.plot_autocorr(idata, var_names=["score"], max_lag=max_lag)
    plt.savefig(out, dpi=300)
    plt.close()


def report_convergence(idata: az.InferenceData, out_dir: Path) -> None:
    """
    Compute R-hat & ESS for “score” variable, save → convergence.txt
    """
    out = out_dir / "convergence.txt"
    out_dir.mkdir(exist_ok=True, parents=True)

    rhat = az.rhat(idata, var_names=["score"])["score"].item()
    ess = az.ess(idata, var_names=["score"])["score"].item()

    with out.open("w") as fh:
        fh.write(f"rhat: {rhat:.3f}\n")
        fh.write(f"ess:  {ess:.1f}\n")
