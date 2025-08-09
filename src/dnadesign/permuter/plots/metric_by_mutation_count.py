"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/plots/metric_by_mutation_count.py

Scatter of each variant's score (y) vs. variant rank (x),
colored by number of mutations. Works with objective_score or
falls back to normalized metric, then legacy 'score' if present.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _choose_y_series(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    if "objective_score" in df.columns:
        return df["objective_score"], "Objective score"

    if "norm_metrics" in df.columns and not df["norm_metrics"].isna().all():
        key: Optional[str] = None
        for d in df["norm_metrics"]:
            if isinstance(d, dict) and d:
                key = next(iter(d.keys()))
                break
        if key:
            return (
                df["norm_metrics"].apply(lambda d: (d or {}).get(key, None)),
                f"Norm {key}",
            )

    if "score" in df.columns:
        return df["score"], "Score"

    raise RuntimeError("No objective_score, norm_metrics, or score available to plot.")


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
    ref_sequence: Optional[str] = None,  # accepted for API parity; unused
) -> None:
    """
    Scatter of every variant's score (y) vs. variant rank (x),
    colored by number of mutations (categorical, no colorbar).
    """
    df = all_df.copy()

    # y-series selection with fallback
    y, y_label = _choose_y_series(df)
    df = df.assign(_y=y).dropna(subset=["_y"])

    # mutation counts
    def _count_mods(m: List[str] | object) -> int:
        if isinstance(m, list):
            return len(m)
        return 0

    df["mut_count"] = df["modifications"].apply(_count_mods)

    # rank by descending score
    df = df.sort_values("_y", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # discrete palette
    mut_levels = sorted(df["mut_count"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {lvl: cmap(i % 10) for i, lvl in enumerate(mut_levels)}

    fig, ax = plt.subplots(figsize=(6, 4))

    # plot highest mut_count first so lower counts draw on top
    for lvl in sorted(mut_levels, reverse=True):
        sub = df[df["mut_count"] == lvl]
        ax.scatter(
            sub["rank"],
            sub["_y"],
            color=color_map[lvl],
            s=18,
            alpha=0.55,
            edgecolors="none",
            label=f"{lvl} mut{'s' if lvl != 1 else ''}",
        )

    ref_name = (
        df["ref_name"].iloc[0] if "ref_name" in df.columns and not df.empty else ""
    )
    ax.set_xlabel("Variant rank (desc)", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(f"{job_name}{f' ({ref_name})' if ref_name else ''}", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linestyle="-", linewidth=0.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.legend(frameon=False, title="Mutation count", fontsize=8, title_fontsize=9)
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
