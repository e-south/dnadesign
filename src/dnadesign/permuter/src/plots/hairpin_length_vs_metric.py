"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/hairpin_length_vs_metric.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _series_for_metric(
    df: pd.DataFrame, metric_id: Optional[str]
) -> Tuple[pd.Series, str]:
    if not metric_id:
        raise RuntimeError(
            "metric_id is required (expects a column permuter__metric__<id>)"
        )
    col = f"permuter__metric__{metric_id}"
    if col not in df.columns:
        raise RuntimeError(f"Metric column not found: {col}")
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
    df = df.assign(_y=y).dropna(subset=["_y"])
    len_col = "permuter__hp_length_paired"
    cat_label_col = "permuter__hp_category_label"
    cat_id_col = "permuter__hp_category_id"
    order_col = "permuter__hp_category_order"
    if len_col not in df.columns:
        raise RuntimeError(
            "hairpin_length_vs_metric requires 'permuter__hp_length_paired' column"
        )
    if (cat_label_col not in df.columns) and (cat_id_col not in df.columns):
        raise RuntimeError(
            "hairpin_length_vs_metric requires hp category columns: "
            "'permuter__hp_category_label' or 'permuter__hp_category_id'"
        )

    if cat_label_col in df.columns:
        cat = df[cat_label_col].astype(str)
    else:
        cat = df[cat_id_col].astype(str)
    df = df.assign(_cat=cat.fillna("cat"))
    # deterministic category ordering if present
    if order_col in df.columns:
        order = df.groupby("_cat")[order_col].min().sort_values().index.tolist()
    else:
        order = sorted(df["_cat"].unique().tolist())

    fs = float(font_scale) if font_scale else 1.0
    # --- enforce a square figure so visual slopes are comparable across categories ---
    if figsize:
        w, h = float(figsize[0]), float(figsize[1])
        side = min(max(3.5, w), max(3.5, h))  # reasonable lower bound
    else:
        side = 6.0
    fig, ax = plt.subplots(figsize=(side, side), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, hspace=0.12, wspace=0.06)

    # seaborn-like background
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color="0.90", linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # palette
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(i % 10) for i, c in enumerate(order)}
    shapes = {
        c: m for c, m in zip(order, ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"])
    }

    # scatter cloud
    for c in order:
        sub = df[df["_cat"] == c]
        if sub.empty:
            continue
        ax.scatter(
            sub[len_col].astype(int),
            sub["_y"],
            s=14,
            alpha=0.14,
            color=colors[c],
            edgecolors="none",
            label=None,
        )

    # per-length means + line
    means = (
        df.groupby(["_cat", len_col])["_y"]
        .mean()
        .reset_index()
        .sort_values(["_cat", len_col])
    )
    for c in order:
        ss = means[means["_cat"] == c]
        if ss.empty:
            continue
        ax.plot(
            ss[len_col],
            ss["_y"],
            marker=shapes[c],
            linestyle="-",
            linewidth=1.4,
            markersize=4.5,
            color=colors[c],
            alpha=0.95,
            label=str(c),
        )

    # labels
    ax.set_xlabel("Hairpin stem length (paired nt)", fontsize=int(round(12 * fs)))
    ax.set_ylabel(y_label, fontsize=int(round(12 * fs)))
    ax.tick_params(labelsize=int(round(10.5 * fs)))

    # legend
    ax.legend(
        title="Category",
        fontsize=int(round(9.5 * fs)),
        title_fontsize=int(round(10.5 * fs)),
        frameon=False,
        loc="best",
    )

    # titles
    ref_name = (
        df["permuter__ref"].iloc[0]
        if "permuter__ref" in df.columns and not df.empty
        else ""
    )
    title = f"{job_name}{f' ({ref_name})' if ref_name else ''}"
    fig.suptitle(title, fontsize=int(round(14 * fs)), y=1.075)
    if evaluators:
        fig.text(
            0.5,
            1.02,
            evaluators,
            ha="center",
            va="top",
            fontsize=int(round(10.5 * fs)),
            alpha=0.80,
        )

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
