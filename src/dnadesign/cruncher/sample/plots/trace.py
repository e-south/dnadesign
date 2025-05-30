"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/plots/trace.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns


def plot_trace(idata: az.InferenceData, out_dir: Path) -> None:
    """
    One-line-per-chain score trace plot → trace_score.png
    Uses seaborn ticks style, a colorblind palette, and removes top/right spines.
    """
    out = out_dir / "trace_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) set seaborn style & palette
    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    # 2) draw the standard ArviZ two-panel trace + density
    #    ArviZ returns an array of Axes, not the Figure
    _ = az.plot_trace(idata, var_names=["score"])

    # 3) grab the current Figure and strip top/right spines on every Axes
    fig = plt.gcf()
    for ax in fig.axes:
        sns.despine(ax=ax)

    # 4) save and close
    fig.savefig(out, dpi=300)
    plt.close(fig)
