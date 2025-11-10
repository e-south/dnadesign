"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/ranked_variants.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _series_for_metric(
    df: pd.DataFrame, metric_id: Optional[str]
) -> Tuple[pd.Series, str]:
    if not metric_id:
        raise RuntimeError("ranked_variants: metric_id is required")
    col = f"permuter__metric__{metric_id}"
    if col not in df.columns:
        raise RuntimeError(f"ranked_variants: metric column not found: {col}")
    return df[col].astype("float64"), str(metric_id)


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
    ref_sequence: Optional[str] = None,  # unused
    metric_id: Optional[str] = None,
    evaluators: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    font_scale: Optional[float] = None,
) -> None:
    df = all_df.copy()
    y, y_label = _series_for_metric(df, metric_id)
    df = df.assign(_y=y).dropna(subset=["_y"]).copy()

    # Prefer round-2 variants for ranking (combinations), but fall back to all if absent
    if "permuter__round" in df.columns and (df["permuter__round"] == 2).any():
        df2 = df[df["permuter__round"] == 2].copy()
    else:
        df2 = df

    # Mut count if available
    mut_col = "permuter__mut_count" if "permuter__mut_count" in df2.columns else None

    df2 = df2.sort_values("_y", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    N = 100  # default top-N
    df_top = df2.head(min(N, len(df2))).copy()
    df_top["rank"] = np.arange(1, len(df_top) + 1)

    # Title scaffold
    fs = float(font_scale) if font_scale else 1.0
    width, height = figsize if figsize else (9.0, 4.5)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.90", linewidth=0.7)

    if mut_col:
        # Discrete palette by mut_count
        mut_levels = sorted(df_top[mut_col].fillna(-1).astype(int).unique().tolist())
        cmap = plt.get_cmap("tab10")
        color_by = {lvl: cmap(i % 10) for i, lvl in enumerate(mut_levels)}
        for lvl in mut_levels:
            sub = df_top[df_top[mut_col].fillna(-1).astype(int) == lvl]
            ax.plot(
                sub["rank"],
                sub["_y"],
                marker="o",
                linestyle="-",
                alpha=0.9,
                label=f"k={lvl}",
                linewidth=1.2,
                color=color_by[lvl],
            )
    else:
        ax.plot(
            df_top["rank"],
            df_top["_y"],
            marker="o",
            linestyle="-",
            alpha=0.95,
            linewidth=1.2,
            label=None,
        )

    # Optional xtick labels with aa_combo_str if present and small N
    if "permuter__aa_combo_str" in df_top.columns and len(df_top) <= 40:
        ax.set_xticks(df_top["rank"].tolist())
        ax.set_xticklabels(
            df_top["permuter__aa_combo_str"].tolist(),
            rotation=70,
            ha="right",
            fontsize=int(round(8.5 * fs)),
        )
    else:
        ax.set_xticks(df_top["rank"].tolist())
        ax.set_xticklabels(df_top["rank"].tolist(), fontsize=int(round(9.5 * fs)))

    ax.set_xlabel("Variant rank", fontsize=int(round(11 * fs)))
    ax.set_ylabel(y_label, fontsize=int(round(11 * fs)))
    ax.tick_params(axis="y", labelsize=int(round(10 * fs)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if mut_col:
        ax.legend(
            title="Mutations (k)",
            fontsize=int(round(9 * fs)),
            title_fontsize=int(round(10 * fs)),
            frameon=False,
        )

    # Titles/subtitle
    ref_name = (
        df_top["permuter__ref"].iloc[0]
        if "permuter__ref" in df_top.columns and not df_top.empty
        else ""
    )
    title = f"{job_name}{f' ({ref_name})' if ref_name else ''}"
    fig.suptitle(title, fontsize=int(round(13 * fs)))
    if evaluators:
        fig.text(
            0.5,
            0.96,
            evaluators,
            ha="center",
            va="top",
            fontsize=int(round(9.5 * fs)),
            alpha=0.75,
        )

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
