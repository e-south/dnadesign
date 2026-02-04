"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/moves.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dnadesign.cruncher.analysis.plots._savefig import savefig

logger = logging.getLogger(__name__)


def _tune_end_from_phase(df: pd.DataFrame) -> int | None:
    if "phase" not in df.columns:
        return None
    tune_df = df[df["phase"] == "tune"]
    if tune_df.empty:
        return None
    return int(tune_df["sweep_idx"].max())


def plot_move_acceptance_time(
    move_df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    if move_df is None or move_df.empty:
        logger.warning("Skipping move acceptance plot: empty move stats.")
        return
    if not {"sweep_idx", "move_kind", "attempted", "accepted"}.issubset(move_df.columns):
        logger.warning("Skipping move acceptance plot: missing required columns.")
        return
    chain_values = None
    if "chain" in move_df.columns:
        chain_values = sorted(move_df["chain"].dropna().unique().tolist())
    if chain_values and len(chain_values) > 1:
        if len(chain_values) > 6:
            logger.warning("Too many chains for faceted move plot; aggregating across chains.")
            chain_values = None
    sns.set_style("ticks", {"axes.grid": False})
    if chain_values:
        ncols = 2
        nrows = int(np.ceil(len(chain_values) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 3.5 * nrows), sharey=True)
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, chain_id in zip(axes_list, chain_values):
            subset = move_df[move_df["chain"] == chain_id]
            grouped = subset.groupby(["sweep_idx", "move_kind"], as_index=False)[["attempted", "accepted"]].sum()
            grouped["accept_rate"] = grouped["accepted"] / grouped["attempted"].replace(0, np.nan)
            for move_kind, sub in grouped.groupby("move_kind"):
                ax.plot(sub["sweep_idx"], sub["accept_rate"], label=str(move_kind))
            tune_end = _tune_end_from_phase(subset)
            if tune_end is not None:
                ax.axvline(tune_end + 0.5, color="grey", linestyle="--", alpha=0.5)
            ax.set_title(f"Chain {chain_id}")
            ax.set_xlabel("Sweep index")
            ax.set_ylabel("Acceptance rate")
        for ax in axes_list[len(chain_values) :]:
            ax.axis("off")
        handles, labels = axes_list[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="upper center")
        fig.suptitle("Move acceptance rate over time", y=0.98)
        fig.tight_layout()
    else:
        grouped = move_df.groupby(["sweep_idx", "move_kind"], as_index=False)[["attempted", "accepted"]].sum()
        grouped["accept_rate"] = grouped["accepted"] / grouped["attempted"].replace(0, np.nan)
        fig, ax = plt.subplots(figsize=(8, 4))
        for move_kind, sub in grouped.groupby("move_kind"):
            ax.plot(sub["sweep_idx"], sub["accept_rate"], label=str(move_kind))
        tune_end = _tune_end_from_phase(move_df)
        if tune_end is not None:
            ax.axvline(tune_end + 0.5, color="grey", linestyle="--", alpha=0.5, label="tune/draw")
        ax.set_title("Move acceptance rate over time")
        ax.set_xlabel("Sweep index")
        ax.set_ylabel("Acceptance rate")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)


def plot_move_usage_time(
    move_df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    if move_df is None or move_df.empty:
        logger.warning("Skipping move usage plot: empty move stats.")
        return
    if not {"sweep_idx", "move_kind", "attempted"}.issubset(move_df.columns):
        logger.warning("Skipping move usage plot: missing required columns.")
        return
    chain_values = None
    if "chain" in move_df.columns:
        chain_values = sorted(move_df["chain"].dropna().unique().tolist())
    if chain_values and len(chain_values) > 1:
        if len(chain_values) > 6:
            logger.warning("Too many chains for faceted move usage plot; aggregating across chains.")
            chain_values = None
    sns.set_style("ticks", {"axes.grid": False})
    if chain_values:
        ncols = 2
        nrows = int(np.ceil(len(chain_values) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 3.5 * nrows), sharey=True)
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, chain_id in zip(axes_list, chain_values):
            subset = move_df[move_df["chain"] == chain_id]
            grouped = subset.groupby(["sweep_idx", "move_kind"], as_index=False)["attempted"].sum()
            totals = (
                grouped.groupby("sweep_idx", as_index=False)["attempted"].sum().rename(columns={"attempted": "total"})
            )
            merged = grouped.merge(totals, on="sweep_idx", how="left")
            merged["usage_frac"] = merged["attempted"] / merged["total"].replace(0, np.nan)
            for move_kind, sub in merged.groupby("move_kind"):
                ax.plot(sub["sweep_idx"], sub["usage_frac"], label=str(move_kind))
            tune_end = _tune_end_from_phase(subset)
            if tune_end is not None:
                ax.axvline(tune_end + 0.5, color="grey", linestyle="--", alpha=0.5)
            ax.set_title(f"Chain {chain_id}")
            ax.set_xlabel("Sweep index")
            ax.set_ylabel("Usage fraction")
        for ax in axes_list[len(chain_values) :]:
            ax.axis("off")
        handles, labels = axes_list[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="upper center")
        fig.suptitle("Move usage fraction over time", y=0.98)
        fig.tight_layout()
    else:
        grouped = move_df.groupby(["sweep_idx", "move_kind"], as_index=False)["attempted"].sum()
        totals = grouped.groupby("sweep_idx", as_index=False)["attempted"].sum().rename(columns={"attempted": "total"})
        merged = grouped.merge(totals, on="sweep_idx", how="left")
        merged["usage_frac"] = merged["attempted"] / merged["total"].replace(0, np.nan)
        fig, ax = plt.subplots(figsize=(8, 4))
        for move_kind, sub in merged.groupby("move_kind"):
            ax.plot(sub["sweep_idx"], sub["usage_frac"], label=str(move_kind))
        tune_end = _tune_end_from_phase(move_df)
        if tune_end is not None:
            ax.axvline(tune_end + 0.5, color="grey", linestyle="--", alpha=0.5, label="tune/draw")
        ax.set_title("Move usage fraction over time")
        ax.set_xlabel("Sweep index")
        ax.set_ylabel("Usage fraction")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)


def plot_pt_swap_by_pair(
    pt_df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
    png_compress_level: int,
) -> None:
    if pt_df is None or pt_df.empty:
        logger.warning("Skipping PT swap plot: empty swap table.")
        return
    if not {"pair_index", "acceptance_rate"}.issubset(pt_df.columns):
        logger.warning("Skipping PT swap plot: missing required columns.")
        return
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(pt_df["pair_index"].astype(str), pt_df["acceptance_rate"], color="#4c78a8", alpha=0.85)
    ax.set_title("PT swap acceptance by adjacent pair")
    ax.set_xlabel("Pair index (coldest last)")
    ax.set_ylabel("Swap acceptance rate")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
