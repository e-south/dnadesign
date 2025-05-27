"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/plots/scatter_pwm.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dnadesign.cruncher.config import CruncherConfig
from dnadesign.cruncher.sample.state import SequenceState
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.motif.model import PWM


def plot_scatter_pwm(
    sample_dir: Path,
    pwms: dict[str, PWM],
    cfg: CruncherConfig
) -> None:
    """
    Scatter two-PWM scores:
      • light gray cloud = random sequences
      • colored points   = MCMC samples over iterations (viridis)
      • big stars        = consensus for each PWM
    """
    samp_path = sample_dir / "samples.csv"
    rand_path = sample_dir / "random_samples.csv"
    hits_path = sample_dir / "hits.csv"

    if not (samp_path.exists() and rand_path.exists() and hits_path.exists()):
        return

    # load MCMC samples and random reference
    df = pd.read_csv(samp_path)
    dr = pd.read_csv(rand_path)

    # figure out which two TFs we're plotting
    x_tf, y_tf = cfg.regulator_sets[0][:2]

    # determine sequence length from the hits file
    hits = pd.read_csv(hits_path)
    L = len(hits["sequence"].iloc[0])

    # only need a scorer for the two PWMs in question
    pair_scorer = Scorer(
        {x_tf: pwms[x_tf], y_tf: pwms[y_tf]},
        bidirectional=cfg.sample.bidirectional
    )

    # compute the consensus‐only points (one star per PWM)
    rng0 = np.random.default_rng(0)
    consensus_points = []
    for tf in (x_tf, y_tf):
        consensus_seq = SequenceState.from_consensus(
            {tf: pwms[tf]},
            mode="longest",
            target_length=L,
            pad_with="background",
            rng=rng0
        )
        sc0, sc1 = pair_scorer.score_per_pwm(consensus_seq.seq)
        consensus_points.append((sc0, sc1, tf))

    # subsample for clarity
    Nsub = min(len(df), 2000)
    df_sub = df.sort_values("iter").sample(Nsub, random_state=0)
    Nrnd = min(len(dr), Nsub)
    dr_sub = dr.sample(Nrnd, random_state=0)

    # styling
    sns.set_style("ticks", {"axes.grid": False})
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # 1) random baseline
    ax.scatter(
        dr_sub[f"score_{x_tf}"],
        dr_sub[f"score_{y_tf}"],
        c="lightgray",
        alpha=0.4,
        s=20,
        label="random"
    )

    # 2) MCMC trajectory colored by iteration
    cmap = sns.color_palette("viridis", as_cmap=True)
    sc = ax.scatter(
        df_sub[f"score_{x_tf}"],
        df_sub[f"score_{y_tf}"],
        c=df_sub["iter"],
        cmap=cmap,
        alpha=0.6,
        s=30,
        linewidth=0
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("iteration", rotation=270, labelpad=15)

    # 3) consensus stars
    for cx, cy, tf in consensus_points:
        ax.scatter(
            cx, cy,
            marker="*",
            s=200,
            edgecolor="black",
            facecolor="yellow",
            label=f"consensus_{tf}"
        )
        ax.text(
            cx, cy,
            f" {tf}",
            va="center",
            ha="left",
            fontsize=10
        )

    # finalize
    ax.set_xlabel(f"score_{x_tf}", fontsize=12)
    ax.set_ylabel(f"score_{y_tf}", fontsize=12)
    ax.set_title(
        f"{x_tf} vs {y_tf} over MCMC\n(random = light gray)",
        fontsize=14
    )
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)

    out = sample_dir / "scatter_pwm.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()