"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/metric_by_mutation_count.py

Round-separated swarms with population indication (transparent violins).
X = round (R1, R2, ...), Y = metric/objective. Color encodes mutation count.

Single metric → axis label is that metric; multi → 'Objective'.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray

_METRIC_LABELS = {
    "ll": "log_likelihood",
    "llr": "log_likelihood_ratio",
    "emb": "embedding_distance",
}


def _pretty_metric(mid: str | None) -> str:
    if not mid:
        return "Objective"
    return _METRIC_LABELS.get(str(mid).strip(), str(mid).strip())


def _series_for_metric(
    df: pd.DataFrame, metric_id: Optional[str]
) -> Tuple[pd.Series, str]:
    if not metric_id:
        raise RuntimeError(
            "metric_id is required (expects a column permuter__observed__<id>)"
        )
    col = f"permuter__observed__{metric_id}"
    if col not in df.columns:
        raise RuntimeError(
            f"Observed metric column not found: {col}. "
            "Run `permuter evaluate` so observed metrics are present."
        )
    return df[col].astype("float64"), _pretty_metric(metric_id)


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
    df = df.assign(_y=y).dropna(subset=["_y"])

    def _count_mods(m: List[str] | object) -> int:
        # Accept list / tuple / ndarray; otherwise 0
        if isinstance(m, (list, tuple)):
            return len(m)
        return (
            int(isinstance(m, ndarray)) * int(len(m)) if isinstance(m, ndarray) else 0
        )

    # Prefer protocol-emitted namespaced mut_count; fall back to token count if absent
    if "permuter__mut_count" in df.columns:
        df["mut_count"] = df["permuter__mut_count"].astype(int)
    else:
        df["mut_count"] = df["permuter__modifications"].apply(_count_mods)

    if "permuter__round" not in df.columns:
        raise RuntimeError(f"{job_name}: variants missing 'permuter__round' field")
    rounds = sorted(df["permuter__round"].unique())
    x_pos = {r: i + 1 for i, r in enumerate(rounds)}

    # slightly tighter jitter
    width = 0.25
    rng = np.random.default_rng(0)

    # point palette per mutation count; soft per-round color for the violin
    mut_levels = sorted(df["mut_count"].unique())
    cmap = plt.get_cmap("tab10")
    color_by_mut = {lvl: cmap(i % 10) for i, lvl in enumerate(mut_levels)}
    round_cmap = plt.get_cmap("tab20")
    color_by_round = {r: round_cmap(i % 20) for i, r in enumerate(rounds)}

    fs = float(font_scale) if font_scale else 1.0
    fig, ax = plt.subplots(
        figsize=(figsize if figsize else (7.4, 4.0)), constrained_layout=True
    )

    # grid behind points
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linestyle="-", linewidth=0.6, zorder=0)

    for r in rounds:
        sub = df[df["permuter__round"] == r]
        x0 = x_pos[r]

        # single violin per round = population distribution
        if not sub.empty:
            vp = ax.violinplot(
                dataset=[sub["_y"].values],
                positions=[x0],
                widths=width * 1.6,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            body = vp["bodies"][0]
            body.set_facecolor(color_by_round[r])
            body.set_edgecolor(color_by_round[r])
            body.set_alpha(0.30)
            body.set_zorder(1.5)

        # one scatter per datum (no duplicate plotting)
        for lvl in mut_levels:
            ss = sub[sub["mut_count"] == lvl]
            if ss.empty:
                continue
            j = rng.uniform(-width, width, size=len(ss))
            ax.scatter(
                x0 + j,
                ss["_y"],
                s=16,
                alpha=0.25,
                edgecolors="none",
                color=color_by_mut[lvl],
                zorder=2.0,
            )

    ax.set_xlim(0.5, len(rounds) + 0.5)
    ax.set_xticks([x_pos[r] for r in rounds])
    ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=int(round(11 * fs)))
    ax.set_xlabel("Round", fontsize=int(round(11 * fs)))
    ax.set_ylabel(y_label, fontsize=int(round(11 * fs)))
    ax.tick_params(axis="y", labelsize=int(round(10 * fs)))

    # title + subtitle kept inside figure to avoid clipping when saving
    ref_name = (
        df["permuter__ref"].iloc[0]
        if "permuter__ref" in df.columns and not df.empty
        else ""
    )

    title = f"{job_name}{f' ({ref_name})' if ref_name else ''}"
    fig.suptitle(title, fontsize=int(round(12 * fs)), y=0.995)
    if evaluators:
        fig.text(
            0.5,
            0.968,
            evaluators,
            ha="center",
            va="top",
            fontsize=int(round(9 * fs)),
            alpha=0.50,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ensure titles are never clipped in saved PNG
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
