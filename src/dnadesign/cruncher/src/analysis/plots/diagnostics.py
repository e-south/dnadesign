"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/diagnostics.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.artifacts.layout import sequences_path

logger = logging.getLogger(__name__)


def _score_stats(idata: az.InferenceData) -> tuple[xr.DataArray | None, int, int]:
    score = idata.posterior.get("score") if hasattr(idata, "posterior") else None
    if score is None:
        return None, 0, 0
    n_chains = int(score.sizes.get("chain", score.shape[0] if score.ndim > 0 else 0))
    n_draws = int(score.sizes.get("draw", score.shape[-1] if score.ndim > 0 else 0))
    return score, n_chains, n_draws


#  Generic helpers
def _tf_pair(tf_pair: tuple[str, str]) -> Tuple[str, str]:
    if len(tf_pair) != 2:
        raise ValueError("Need exactly two TFs for pairwise plots")
    return tuple(tf_pair)


def _flatten_axes(obj: object) -> list[plt.Axes]:
    flat: list[plt.Axes] = []
    if isinstance(obj, plt.Axes):
        return [obj]
    if isinstance(obj, (list, tuple, np.ndarray)):
        for sub in obj:
            flat.extend(_flatten_axes(sub))
    return flat


def _plot_trace_only(score_values: np.ndarray, out: Path) -> None:
    values = np.asarray(score_values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    fig, ax = plt.subplots(figsize=(8, 3))
    for idx, chain in enumerate(values):
        ax.plot(chain, alpha=0.7, label=f"chain {idx + 1}")
    ax.set_title("Trace of Score (all chains)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    if values.shape[0] > 1:
        ax.legend(loc="best", frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trace(idata: az.InferenceData, out_dir: Path) -> None:
    """
    Overlay all chains' score-traces + posterior density → trace_score.png.

    • Left panel: Posterior density of “score”
    • Right panel: Trace of “score” versus iteration
    """
    out = out_dir / "diag__trace_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    score, _, _ = _score_stats(idata)
    if score is None:
        logger.warning("Skipping trace plot: 'score' not found in trace.")
        return
    values = np.asarray(score.values, dtype=float)
    if values.size == 0:
        logger.warning("Skipping trace plot: empty score array.")
        return
    with np.errstate(all="ignore"):
        spread = np.nanmax(values) - np.nanmin(values)
    if not np.isfinite(spread) or spread <= 0:
        logger.warning("Trace plot: score is constant or non-finite; rendering trace-only plot.")
        _plot_trace_only(values, out)
        return

    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    with az.style.context("arviz-doc"):
        try:
            axes = az.plot_trace(idata, var_names=["score"], combined=False)
        except Exception as exc:
            logger.warning("Trace plot failed (%s); rendering trace-only plot.", exc)
            _plot_trace_only(values, out)
            return

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
    out = out_dir / "diag__autocorr_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    score, _, n_draws = _score_stats(idata)
    if score is None:
        logger.warning("Skipping autocorr plot: 'score' not found in trace.")
        return
    if n_draws < 2:
        logger.warning("Skipping autocorr plot: need at least 2 draws (found %d).", n_draws)
        return
    max_lag = min(max_lag, max(1, n_draws - 1))

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
    out = out_dir / "diag__convergence.txt"
    out_dir.mkdir(exist_ok=True, parents=True)

    score, n_chains, n_draws = _score_stats(idata)
    if score is None:
        logger.warning("Skipping convergence metrics: 'score' not found in trace.")
        with out.open("w") as fh:
            fh.write("rhat: n/a\n")
            fh.write("ess:  n/a\n")
        return
    if n_chains < 2 or n_draws < 4:
        logger.warning(
            "Skipping convergence metrics: need >=2 chains and >=4 draws (got chains=%d draws=%d).",
            n_chains,
            n_draws,
        )
        with out.open("w") as fh:
            fh.write("rhat: n/a\n")
            fh.write("ess:  n/a\n")
        return

    rhat = az.rhat(score)["score"].item()
    ess = az.ess(score)["score"].item()

    with out.open("w") as fh:
        fh.write(f"rhat: {rhat:.3f}\n")
        fh.write(f"ess:  {ess:.1f}\n")


def plot_rank_diagnostic(idata: az.InferenceData, out_dir: Path) -> None:
    """
    Draw a rank-plot for “score” across chains → rank_plot_score.png.
    """
    out = out_dir / "diag__rank_plot_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    score, n_chains, n_draws = _score_stats(idata)
    if score is None:
        logger.warning("Skipping rank plot: 'score' not found in trace.")
        return
    if n_chains < 2 or n_draws < 4:
        logger.warning(
            "Skipping rank plot: need >=2 chains and >=4 draws (got chains=%d draws=%d).",
            n_chains,
            n_draws,
        )
        return

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
    out = out_dir / "diag__ess_evolution_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    score, n_chains, n_draws = _score_stats(idata)
    if score is None:
        logger.warning("Skipping ESS plot: 'score' not found in trace.")
        return
    if n_draws < 4:
        logger.warning("Skipping ESS plot: need >=4 draws (found %d).", n_draws)
        return

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
    out1 = out_dir / "diag__ess_local_score.png"
    out2 = out_dir / "diag__ess_quantile_score.png"
    out_dir.mkdir(exist_ok=True, parents=True)

    score, n_chains, n_draws = _score_stats(idata)
    if score is None:
        logger.warning("Skipping ESS diagnostics: 'score' not found in trace.")
        return
    if n_draws < 4:
        logger.warning("Skipping ESS diagnostics: need >=4 draws (found %d).", n_draws)
        return

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


def make_pair_idata(sample_dir: Path, tf_pair: tuple[str, str]) -> az.InferenceData:
    """
    Build an ArviZ InferenceData with two DataArrays: score_<TF1>, score_<TF2>.
    The TFs are taken from the provided tf_pair tuple.
    """
    x_tf, y_tf = _tf_pair(tf_pair)

    df = read_parquet(sequences_path(sample_dir))
    required = {"chain", "draw", f"score_{x_tf}", f"score_{y_tf}"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"sequences.parquet missing required columns: {missing}")
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"].copy()
    if df.empty:
        raise ValueError("sequences.parquet contains no draw rows")

    chains = sorted(df["chain"].unique())
    draws = sorted(df["draw"].unique())
    n_chains, n_draws = len(chains), len(draws)
    if n_chains == 0 or n_draws == 0:
        raise ValueError("sequences.parquet must include at least one chain and draw")

    arr_x = np.zeros((n_chains, n_draws))
    arr_y = np.zeros((n_chains, n_draws))

    for i, c in enumerate(chains):
        sub = df[df["chain"] == c].sort_values("draw")
        chain_draws = sub["draw"].tolist()
        if chain_draws != draws:
            raise ValueError(f"Inconsistent draws for chain {c}: expected draws {draws}, found {chain_draws}.")
        arr_x[i, :] = sub[f"score_{x_tf}"].to_numpy()
        arr_y[i, :] = sub[f"score_{y_tf}"].to_numpy()

    da_x = xr.DataArray(
        arr_x,
        dims=("chain", "draw"),
        coords={"chain": chains, "draw": draws},
        name=f"score_{x_tf}",
    )
    da_y = xr.DataArray(
        arr_y,
        dims=("chain", "draw"),
        coords={"chain": chains, "draw": draws},
        name=f"score_{y_tf}",
    )

    return az.InferenceData(posterior=da_x.to_dataset().merge(da_y.to_dataset()))


def plot_pair_pwm_scores(
    idata_pair: az.InferenceData,
    out_dir: Path,
    tf_pair: tuple[str, str],
) -> None:
    """
    2-D KDE + marginals for the selected TF pair.
    """
    x_tf, y_tf = _tf_pair(tf_pair)
    out = out_dir / "pwm__pair_scores.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("ticks", {"axes.grid": False})
    sns.set_palette("colorblind")

    with az.style.context("arviz-doc"):
        az.plot_pair(
            idata_pair,
            var_names=[f"score_{x_tf}", f"score_{y_tf}"],
            kind="kde",
            marginals=True,
            point_estimate=None,
        )
        plt.gcf().suptitle(f"Joint & Marginal KDE of PWM Scores ({x_tf} vs {y_tf})", fontsize=14)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()


def plot_parallel_pwm_scores(
    idata_pair: az.InferenceData,
    out_dir: Path,
    tf_pair: tuple[str, str],
) -> None:
    """
    Parallel-coordinates line plot for the selected TF pair.
    """

    x_tf, y_tf = _tf_pair(tf_pair)
    out = out_dir / "pwm__parallel_scores.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = idata_pair.posterior.to_dataframe().reset_index()

    chains = sorted(df["chain"].unique())
    palette = dict(zip(chains, sns.color_palette("colorblind", len(chains))))

    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(4, 6))

    for _, row in df.iterrows():
        ax.plot(
            [0, 1],
            [row[f"score_{x_tf}"], row[f"score_{y_tf}"]],
            color=palette[row["chain"]],
            alpha=0.3,
            lw=0.5,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([x_tf, y_tf], fontsize=10)
    ax.set_ylabel("Score value", fontsize=10)
    ax.set_title("Parallel Coordinates of PWM Scores", fontsize=14)
    sns.despine(ax=ax)

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
