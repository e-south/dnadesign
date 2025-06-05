"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/analyse/plots/diagnostics.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr


def plot_trace(idata: az.InferenceData, out_dir: Path) -> None:
    """
    Overlay all chains' score-traces + posterior density → trace_score.png.

    • Left panel: Posterior density of “score”
    • Right panel: Trace of “score” versus iteration
    """
    out = out_dir / "trace_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    with az.style.context("arviz-doc"):
        axes = az.plot_trace(idata, var_names=["score"], combined=False)

        # Figure‐level title
        fig = plt.gcf()
        fig.suptitle("Score Trace & Posterior Density", fontsize=14)

        # Add explicit titles/labels if ArviZ did not set them
        if isinstance(axes, plt.Axes):
            # Single‐axis fallback: treat as trace only
            ax_trace = axes
            ax_trace.set_title("Trace of Score (all chains)", fontsize=12)
            ax_trace.set_xlabel("Iteration")
            ax_trace.set_ylabel("Score")
            sns.despine(ax=ax_trace)
        else:
            try:
                ax_density, ax_trace = axes[0], axes[1]
            except Exception:
                flat_axes = list(_flatten_axes(axes))
                if len(flat_axes) >= 2:
                    ax_density, ax_trace = flat_axes[0], flat_axes[1]
                elif len(flat_axes) == 1:
                    ax_density, ax_trace = None, flat_axes[0]
                else:
                    return

            # Posterior‐density panel
            if ax_density is not None:
                ax_density.set_title("Posterior Density of Score", fontsize=12)
                ax_density.set_xlabel("Score")
                ax_density.set_ylabel("Density")
                sns.despine(ax=ax_density)

            # Trace panel
            ax_trace.set_title("Trace of Score (all chains)", fontsize=12)
            ax_trace.set_xlabel("Iteration")
            ax_trace.set_ylabel("Score")
            sns.despine(ax=ax_trace)

        # Save with tight bounding box to avoid clipping
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_autocorr(idata: az.InferenceData, out_dir: Path, max_lag: int = 100) -> None:
    """
    Plot autocorrelation of “score” up to lag=max_lag → autocorr_score.png.
    """
    out = out_dir / "autocorr_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    with az.style.context("arviz-doc"):
        az.plot_autocorr(idata, var_names=["score"], max_lag=max_lag)

        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("Autocorrelation of Score", fontsize=14)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, linestyle="--", alpha=0.5)
        sns.despine(ax=ax)

        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def report_convergence(idata: az.InferenceData, out_dir: Path) -> None:
    """
    Compute R-hat & ESS for the “score” variable and write to convergence.txt.
    """
    out = out_dir / "convergence.txt"
    out_dir.mkdir(exist_ok=True, parents=True)

    rhat = az.rhat(idata, var_names=["score"])["score"].item()
    ess = az.ess(idata, var_names=["score"])["score"].item()

    with out.open("w") as fh:
        fh.write(f"rhat: {rhat:.3f}\n")
        fh.write(f"ess:  {ess:.1f}\n")


def plot_rank_diagnostic(idata: az.InferenceData, out_dir: Path) -> None:
    """
    Draw a rank-plot for “score” across chains → rank_plot_score.png.
    """
    out = out_dir / "rank_plot_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    with az.style.context("arviz-doc"):
        az.plot_rank(idata, var_names=["score"], kind="bars")

        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("Rank Plot of Score Across Chains", fontsize=14)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Rank of Score among Chains")
        ax.grid(True, linestyle="--", alpha=0.3)
        sns.despine(ax=ax)

        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_ess(idata: az.InferenceData, out_dir: Path) -> None:
    """
    Plot how the Effective Sample Size (ESS) for “score” evolves → ess_evolution_score.png.
    """
    out = out_dir / "ess_evolution_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    with az.style.context("arviz-doc"):
        az.plot_ess(idata, var_names=["score"], kind="evolution")

        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title("ESS Evolution for Score", fontsize=14)
        ax.set_xlabel("Fraction of Draws")
        ax.set_ylabel("Effective Sample Size")
        ax.grid(True, linestyle="--", alpha=0.4)
        sns.despine(ax=ax)

        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_ess_local_and_quantile(idata: az.InferenceData, out_dir: Path) -> None:
    """
    Plot both:
      1) Local ESS (sliding window) → ess_local_score.png
      2) ESS by quantile → ess_quantile_score.png
    """
    out1 = out_dir / "ess_local_score.png"
    out2 = out_dir / "ess_quantile_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    with az.style.context("arviz-doc"):
        # 1) Local ESS
        az.plot_ess(idata, var_names=["score"], kind="local")
        fig1 = plt.gcf()
        ax1 = plt.gca()
        ax1.set_title("Local ESS of Score (sliding window)", fontsize=14)
        ax1.set_xlabel("Fraction of Draws")
        ax1.set_ylabel("ESS in Window")
        ax1.grid(True, linestyle="--", alpha=0.4)
        sns.despine(ax=ax1)
        fig1.savefig(out1, dpi=300, bbox_inches="tight")
        plt.close(fig1)

        # 2) ESS by quantile
        az.plot_ess(idata, var_names=["score"], kind="quantile", quartiles=[0.1, 0.5, 0.9])
        fig2 = plt.gcf()
        ax2 = plt.gca()
        ax2.set_title("ESS by Quantile of Score", fontsize=14)
        ax2.set_xlabel("Quantile")
        ax2.set_ylabel("ESS")
        ax2.grid(True, linestyle="--", alpha=0.4)
        sns.despine(ax=ax2)
        fig2.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close(fig2)


def make_pair_idata(sample_dir: Path) -> az.InferenceData:
    """
    Reads <sample_dir>/sequences.csv and builds an InferenceData that contains
    two DataArrays: score_cpxR and score_soxR (dims=('chain','draw')).
    """
    import pandas as pd

    df = pd.read_csv(sample_dir / "sequences.csv")
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"].copy()

    chains = sorted(df["chain"].unique())
    draws = sorted(df["draw"].unique())
    n_chains = len(chains)
    n_draws = len(draws)

    arr_x = np.zeros((n_chains, n_draws))
    arr_y = np.zeros((n_chains, n_draws))

    for i, c in enumerate(chains):
        sub = df[df["chain"] == c].sort_values("draw")
        arr_x[i, :] = sub["score_cpxR"].to_numpy()
        arr_y[i, :] = sub["score_soxR"].to_numpy()

    da_x = xr.DataArray(
        arr_x,
        dims=("chain", "draw"),
        coords={"chain": chains, "draw": draws},
        name="score_cpxR",
    )
    da_y = xr.DataArray(
        arr_y,
        dims=("chain", "draw"),
        coords={"chain": chains, "draw": draws},
        name="score_soxR",
    )

    ds = da_x.to_dataset().merge(da_y.to_dataset())
    return az.InferenceData(posterior=ds)


def plot_pair_pwm_scores(idata_pair: az.InferenceData, out_dir: Path) -> None:
    """
    Draw a 2D KDE + marginals for (score_cpxR, score_soxR) → pair_pwm_scores.png.
    """
    out = out_dir / "pair_pwm_scores.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    with az.style.context("arviz-doc"):
        az.plot_pair(
            idata_pair,
            var_names=["score_cpxR", "score_soxR"],
            kind="kde",
            marginals=True,
            point_estimate=None,
        )

        fig = plt.gcf()
        fig.suptitle("Joint & Marginal KDE of PWM Scores (cpxR vs. soxR)", fontsize=14)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_parallel_pwm_scores(idata_pair: az.InferenceData, out_dir: Path) -> None:
    """
    Draw a Parallel Coordinates plot of (score_cpxR, score_soxR) → parallel_pwm_scores.png.

    Each draw (across all chains) is a line between x=0 (“cpxR”) and x=1 (“soxR”),
    colored by chain.
    """
    out = out_dir / "parallel_pwm_scores.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Convert to a flat DataFrame with columns: chain, draw, score_cpxR, score_soxR
    df = idata_pair.posterior.to_dataframe().reset_index()

    chains = sorted(df["chain"].unique())
    palette = dict(zip(chains, sns.color_palette("colorblind", n_colors=len(chains))))

    sns.set_style("ticks", {"axes.grid": False})

    fig, ax = plt.subplots(figsize=(4, 6))
    for _, row in df.iterrows():
        ax.plot(
            [0, 1],
            [row["score_cpxR"], row["score_soxR"]],
            color=palette[row["chain"]],
            alpha=0.3,
            linewidth=0.5,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cpxR", "soxR"], fontsize=10)
    ax.set_ylabel("Score value", fontsize=10)
    ax.set_title("Parallel Coordinates of PWM Scores", fontsize=14)
    sns.despine(ax=ax)

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _flatten_axes(obj: object) -> list[plt.Axes]:
    """
    Helper: given nested lists/tuples/ndarrays of Axes, recursively flatten.
    """
    flat: list[plt.Axes] = []
    if isinstance(obj, plt.Axes):
        return [obj]
    if isinstance(obj, (list, tuple, np.ndarray)):
        for sub in obj:
            flat.extend(_flatten_axes(sub))
    return flat
