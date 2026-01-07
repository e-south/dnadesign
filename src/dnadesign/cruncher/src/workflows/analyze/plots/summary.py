"""Multi-TF summary tables and plots for analysis runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dnadesign.cruncher.utils.parquet import read_parquet


def _score_columns(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def load_score_frame(seq_path: Path, tf_names: list[str]) -> pd.DataFrame:
    df = read_parquet(seq_path)
    if "phase" in df.columns:
        df = df[df["phase"] == "draw"].copy()
    cols = _score_columns(tf_names)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns in sequences.parquet: {missing}")
    return df[cols].copy()


def write_score_summary(score_df: pd.DataFrame, tf_names: list[str], out_path: Path) -> None:
    summary = score_df.agg(["mean", "median", "std", "min", "max"]).T
    summary.insert(0, "tf", [name.replace("score_", "") for name in summary.index])
    summary.reset_index(drop=True, inplace=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)


def write_elite_topk(elites_df: pd.DataFrame, tf_names: list[str], out_path: Path, top_k: int) -> None:
    if "sequence" not in elites_df.columns:
        raise ValueError("Elites parquet missing required 'sequence' column.")
    cols = _score_columns(tf_names)
    missing = [col for col in cols if col not in elites_df.columns]
    if missing:
        raise ValueError(f"Missing score columns in elites parquet: {missing}")
    df = elites_df.copy()
    if "rank" in df.columns:
        df = df.nsmallest(top_k, "rank")
    elif "norm_sum" in df.columns:
        df = df.nlargest(top_k, "norm_sum")
    else:
        df = df.head(top_k)
    keep_cols = ["sequence"] + [c for c in ("rank", "norm_sum") if c in df.columns] + cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[keep_cols].to_csv(out_path, index=False)


def plot_score_hist(score_df: pd.DataFrame, tf_names: list[str], out_path: Path) -> None:
    sns.set_style("ticks", {"axes.grid": False})
    n = len(tf_names)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3 * nrows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, tf in enumerate(tf_names):
        ax = axes_list[i]
        sns.histplot(score_df[f"score_{tf}"], bins=30, kde=True, ax=ax)
        ax.set_title(f"{tf} score distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
    for j in range(i + 1, len(axes_list)):
        axes_list[j].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_box(score_df: pd.DataFrame, tf_names: list[str], out_path: Path) -> None:
    melted = score_df.melt(var_name="tf", value_name="score")
    melted["tf"] = melted["tf"].str.replace("score_", "", regex=False)
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(max(6, len(tf_names) * 1.5), 4))
    sns.boxplot(data=melted, x="tf", y="score", ax=ax)
    ax.set_title("Per-TF score distribution")
    ax.set_xlabel("TF")
    ax.set_ylabel("Score")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(score_df: pd.DataFrame, tf_names: list[str], out_path: Path) -> None:
    corr = score_df.corr()
    corr.index = [name.replace("score_", "") for name in corr.index]
    corr.columns = [name.replace("score_", "") for name in corr.columns]
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, cmap="vlag", center=0.0, square=True, ax=ax)
    ax.set_title("TF score correlation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_parallel_coords(elites_df: pd.DataFrame, tf_names: list[str], out_path: Path) -> None:
    cols = _score_columns(tf_names)
    missing = [col for col in cols if col not in elites_df.columns]
    if missing:
        raise ValueError(f"Missing score columns in elites parquet: {missing}")
    df = elites_df.copy()
    if "rank" in df.columns:
        df = df.nsmallest(min(50, len(df)), "rank")
    elif "norm_sum" in df.columns:
        df = df.nlargest(min(50, len(df)), "norm_sum")
    else:
        df = df.head(min(50, len(df)))
    scores = df[cols].copy()
    scores = (scores - scores.min()) / (scores.max() - scores.min()).replace(0, 1)
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(max(6, len(tf_names) * 1.5), 4))
    xs = list(range(len(cols)))
    for _, row in scores.iterrows():
        ax.plot(xs, row.values, color="steelblue", alpha=0.3, linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("score_", "") for c in cols], rotation=30, ha="right")
    ax.set_ylabel("Scaled score (0-1)")
    ax.set_title("Parallel coordinates (top elites)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
