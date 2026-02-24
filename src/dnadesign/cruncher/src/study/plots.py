"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/plots.py

Render Study aggregate diagnostic plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb

_AXIS_LABEL_FONT_SIZE = 14
_TICK_LABEL_FONT_SIZE = 12
_LEGEND_FONT_SIZE = 11
_TITLE_FONT_SIZE = 13
_ERRORBAR_CAP_SIZE = 4
_ERRORBAR_CAP_THICKNESS = 1.1


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _resolve_baseline_x(df: pd.DataFrame, *, x_col: str) -> float | None:
    if "is_base_value" not in df.columns:
        return None
    mask = df["is_base_value"].fillna(False).astype(bool)
    if not bool(mask.any()):
        return None
    values = _as_float(df.loc[mask, x_col]).dropna().to_numpy(dtype=float)
    unique = sorted(set(float(item) for item in values.tolist()))
    if len(unique) != 1:
        raise ValueError(f"Baseline marker requires one unique {x_col} value, found {unique}")
    return float(unique[0])


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _lighter_color(color: str, amount: float = 0.35) -> str:
    r, g, b = to_rgb(color)
    mix = max(0.0, min(1.0, float(amount)))
    return str(to_hex((r + (1.0 - r) * mix, g + (1.0 - g) * mix, b + (1.0 - b) * mix)))


def _finalize_axes(
    *,
    fig,
    ax_score,
    ax_div,
    title: str,
    legend_handles: list[object],
    legend_labels: list[str],
) -> None:
    ax_score.tick_params(axis="both", labelsize=_TICK_LABEL_FONT_SIZE)
    ax_div.tick_params(axis="y", labelsize=_TICK_LABEL_FONT_SIZE)
    ax_score.set_title(title, fontsize=_TITLE_FONT_SIZE)
    ax_score.legend(
        legend_handles,
        legend_labels,
        fontsize=_LEGEND_FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=1,
        frameon=False,
    )
    fig.tight_layout()


def plot_mmr_diversity_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    required = {
        "series_label",
        "diversity",
        "score_mean",
        "score_sem",
        "diversity_metric_mean",
        "diversity_metric_sem",
        "diversity_metric_label",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"MMR tradeoff plot table missing columns: {missing}")
    if df.empty:
        raise ValueError("MMR tradeoff plot requires non-empty input table.")

    plt = _pyplot()
    fig, ax_score = plt.subplots(figsize=(6.5, 6.5))
    ax_div = ax_score.twinx()
    baseline_x = _resolve_baseline_x(df, x_col="diversity")
    # Okabe-Ito palette for colorblind-safe contrast.
    score_color = "#0072B2"
    div_color = "#E69F00"

    for series_label, group in df.groupby("series_label", sort=True):
        ordered = group.sort_values("diversity")
        x = _as_float(ordered["diversity"]).to_numpy()
        y_score = _as_float(ordered["score_mean"]).to_numpy()
        y_score_sem = _as_float(ordered["score_sem"]).fillna(0.0).to_numpy()
        y_div = _as_float(ordered["diversity_metric_mean"]).to_numpy()
        y_div_sem = _as_float(ordered["diversity_metric_sem"]).fillna(0.0).to_numpy()
        ax_score.errorbar(
            x,
            y_score,
            yerr=y_score_sem,
            marker="o",
            linewidth=1.5,
            color=score_color,
            ecolor=_lighter_color(score_color),
            capsize=_ERRORBAR_CAP_SIZE,
            capthick=_ERRORBAR_CAP_THICKNESS,
            alpha=0.80,
            label=f"Score: {series_label}",
        )
        ax_div.errorbar(
            x,
            y_div,
            yerr=y_div_sem,
            marker="s",
            linewidth=1.2,
            color=div_color,
            ecolor=_lighter_color(div_color),
            capsize=_ERRORBAR_CAP_SIZE,
            capthick=_ERRORBAR_CAP_THICKNESS,
            alpha=0.70,
            label=f"Diversity: {series_label}",
        )
        if baseline_x is not None:
            baseline_mask = (abs(x - baseline_x) <= 1e-9) & np.isfinite(x)
            if baseline_mask.any():
                ax_score.scatter(
                    x[baseline_mask],
                    y_score[baseline_mask],
                    marker="o",
                    s=55,
                    facecolors=score_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                )
                ax_div.scatter(
                    x[baseline_mask],
                    y_div[baseline_mask],
                    marker="s",
                    s=55,
                    facecolors=div_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                )

    if baseline_x is not None:
        ax_score.axvline(
            baseline_x,
            linestyle="--",
            linewidth=1.1,
            color="#595959",
            alpha=0.8,
            label="Core config value",
        )

    ax_score.set_xlabel("MMR diversity", fontsize=_AXIS_LABEL_FONT_SIZE)
    ax_score.set_ylabel("Median selected score", color=score_color, fontsize=_AXIS_LABEL_FONT_SIZE)
    metric_label = str(df["diversity_metric_label"].iloc[0])
    ax_div.set_ylabel(metric_label, color=div_color, fontsize=_AXIS_LABEL_FONT_SIZE)
    ax_score.grid(True, alpha=0.2)
    handles_a, labels_a = ax_score.get_legend_handles_labels()
    handles_b, labels_b = ax_div.get_legend_handles_labels()
    _finalize_axes(
        fig=fig,
        ax_score=ax_score,
        ax_div=ax_div,
        title="MMR diversity tradeoff: score vs sequence diversity",
        legend_handles=handles_a + handles_b,
        legend_labels=labels_a + labels_b,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sequence_length_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    required = {
        "series_label",
        "sequence_length",
        "score_mean",
        "score_sem",
        "diversity_metric_mean",
        "diversity_metric_sem",
        "diversity_metric_label",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Sequence-length tradeoff plot table missing columns: {missing}")
    if df.empty:
        raise ValueError("Sequence-length tradeoff plot requires non-empty input table.")

    plt = _pyplot()
    fig, ax_score = plt.subplots(figsize=(6.5, 6.5))
    ax_div = ax_score.twinx()
    baseline_x = _resolve_baseline_x(df, x_col="sequence_length")
    score_color = "#0072B2"
    div_color = "#E69F00"

    for series_label, group in df.groupby("series_label", sort=True):
        ordered = group.sort_values("sequence_length")
        x = _as_float(ordered["sequence_length"]).to_numpy()
        y_score = _as_float(ordered["score_mean"]).to_numpy()
        y_score_sem = _as_float(ordered["score_sem"]).fillna(0.0).to_numpy()
        y_div = _as_float(ordered["diversity_metric_mean"]).to_numpy()
        y_div_sem = _as_float(ordered["diversity_metric_sem"]).fillna(0.0).to_numpy()
        ax_score.errorbar(
            x,
            y_score,
            yerr=y_score_sem,
            marker="o",
            linewidth=1.5,
            color=score_color,
            ecolor=_lighter_color(score_color),
            capsize=_ERRORBAR_CAP_SIZE,
            capthick=_ERRORBAR_CAP_THICKNESS,
            alpha=0.85,
            label=f"Score: {series_label}",
        )
        ax_div.errorbar(
            x,
            y_div,
            yerr=y_div_sem,
            marker="s",
            linewidth=1.2,
            color=div_color,
            ecolor=_lighter_color(div_color),
            capsize=_ERRORBAR_CAP_SIZE,
            capthick=_ERRORBAR_CAP_THICKNESS,
            alpha=0.75,
            label=f"Diversity: {series_label}",
        )
        if baseline_x is not None:
            baseline_mask = (abs(x - baseline_x) <= 1e-9) & np.isfinite(x)
            if baseline_mask.any():
                ax_score.scatter(
                    x[baseline_mask],
                    y_score[baseline_mask],
                    marker="o",
                    s=55,
                    facecolors=score_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                )
                ax_div.scatter(
                    x[baseline_mask],
                    y_div[baseline_mask],
                    marker="s",
                    s=55,
                    facecolors=div_color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                )

    if baseline_x is not None:
        ax_score.axvline(
            baseline_x,
            linestyle="--",
            linewidth=1.1,
            color="#595959",
            alpha=0.8,
            label="Core config value",
        )

    ax_score.set_xlabel("Sequence length", fontsize=_AXIS_LABEL_FONT_SIZE)
    from matplotlib.ticker import MaxNLocator

    ax_score.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_score.set_ylabel("Median elite score", color=score_color, fontsize=_AXIS_LABEL_FONT_SIZE)
    metric_label = str(df["diversity_metric_label"].iloc[0])
    ax_div.set_ylabel(metric_label, color=div_color, fontsize=_AXIS_LABEL_FONT_SIZE)
    ax_score.grid(True, alpha=0.2)
    handles_a, labels_a = ax_score.get_legend_handles_labels()
    handles_b, labels_b = ax_div.get_legend_handles_labels()
    _finalize_axes(
        fig=fig,
        ax_score=ax_score,
        ax_div=ax_div,
        title="Sequence length tradeoff: score vs sequence diversity",
        legend_handles=handles_a + handles_b,
        legend_labels=labels_a + labels_b,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
