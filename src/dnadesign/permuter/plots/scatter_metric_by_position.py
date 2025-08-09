"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/plots/scatter_metric_by_position.py

Round-1 scatter of variant score vs. sequence position.
Uses objective_score (preferred) → normalized metric → legacy score.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib as mpl
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


def _extract_position(modifications: List[str] | str) -> int:
    mods = modifications if isinstance(modifications, list) else [modifications]
    for token in mods:
        digits = "".join(ch for ch in str(token) if ch.isdigit())
        if digits:
            return int(digits)
    return 0


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
    ref_sequence: Optional[str] = None,  # accepted for API parity; unused
) -> None:
    """
    Round-1 only: scatter of all scores per position (light gray),
    overlaid with mean±SD (dark gray), with no marker edges.
    """
    df = all_df[all_df["round"] == 1].copy()
    if df.empty:
        raise RuntimeError(f"{job_name}: no round-1 variants to plot")

    y, y_label = _choose_y_series(df)
    df = df.assign(_y=y).dropna(subset=["_y"])
    df["position"] = df["modifications"].apply(_extract_position)

    # stats per position
    stats = (
        df.groupby("position")["_y"]
        .agg(mean="mean", sd="std")
        .reset_index()
        .sort_values("position")
    )

    # grid style
    mpl.rcParams.update(
        {
            "axes.axisbelow": True,
            "grid.color": "0.9",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True)

    # all individual points
    ax.scatter(
        df["position"],
        df["_y"],
        color="lightgray",
        alpha=0.3,
        s=10,
        edgecolors="none",
        label="_nolegend_",
    )

    # mean ± SD
    ax.errorbar(
        stats["position"],
        stats["mean"],
        yerr=stats["sd"].fillna(0),
        fmt="o-",
        color="gray",
        alpha=0.6,
        markersize=4,
        capsize=2,
        elinewidth=0.8,
        markeredgecolor="none",
        label="mean ± SD",
    )

    ref_name = (
        df["ref_name"].iloc[0] if "ref_name" in df.columns and not df.empty else ""
    )
    ax.set_title(
        f"{job_name}{f' ({ref_name})' if ref_name else ''}", fontsize=12, pad=6
    )

    ax.set_xlabel("Sequence position")
    ax.set_ylabel(y_label)

    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
