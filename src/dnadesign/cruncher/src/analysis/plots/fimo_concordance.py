"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/fimo_concordance.py

Plot descriptive concordance between Cruncher optimizer scores and FIMO scores.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.cruncher.analysis.plots._savefig import savefig
from dnadesign.cruncher.analysis.plots._style import apply_axes_style


def _safe_corr(x: pd.Series, y: pd.Series, *, method: str) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return None
    value = x.corr(y, method=method)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _binned_summary(x: pd.Series, y: pd.Series, *, n_bins: int = 24) -> pd.DataFrame:
    data = pd.DataFrame({"x": x.astype(float), "y": y.astype(float)})
    if data.empty:
        return pd.DataFrame(columns=["x", "y50", "y10", "y90", "n"])
    unique_x = int(data["x"].nunique(dropna=True))
    if unique_x < 2:
        return pd.DataFrame(columns=["x", "y50", "y10", "y90", "n"])
    bins = min(int(n_bins), unique_x)
    if bins < 2:
        return pd.DataFrame(columns=["x", "y50", "y10", "y90", "n"])
    try:
        groups = pd.qcut(data["x"], q=bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["x", "y50", "y10", "y90", "n"])
    summary = (
        data.groupby(groups, observed=True)
        .agg(
            x=("x", "median"),
            y50=("y", "median"),
            y10=("y", lambda s: float(s.quantile(0.10))),
            y90=("y", lambda s: float(s.quantile(0.90))),
            n=("y", "size"),
        )
        .reset_index(drop=True)
    )
    summary = summary[summary["n"] >= 3].copy()
    return summary


def plot_optimizer_vs_fimo(
    concordance_df: pd.DataFrame,
    out_path: Path,
    *,
    x_label: str = "Cruncher joint score (weakest TF best-window norm-LLR)",
    y_label: str = "FIMO weakest-TF score (-log10 sequence p-value)",
    title: str = "Cruncher optimizer vs FIMO weakest-TF score",
    dpi: int = 300,
    png_compress_level: int = 9,
) -> dict[str, object]:
    if concordance_df is None or concordance_df.empty:
        raise ValueError("FIMO concordance plot requires non-empty input data.")
    required = {"objective_scalar", "fimo_joint_weakest_score"}
    missing = sorted(required - set(concordance_df.columns))
    if missing:
        raise ValueError(f"FIMO concordance plot missing required columns: {missing}")

    x = pd.to_numeric(concordance_df["objective_scalar"], errors="coerce")
    y = pd.to_numeric(concordance_df["fimo_joint_weakest_score"], errors="coerce")
    valid = x.notna() & y.notna()
    if not bool(valid.any()):
        raise ValueError("FIMO concordance plot requires at least one finite x/y point.")
    x = x[valid].astype(float)
    y = y[valid].astype(float)

    pearson = _safe_corr(x, y, method="pearson")
    spearman = _safe_corr(x, y, method="spearman")
    low_score_fraction = float((y < 0.1).mean())
    trend_df = _binned_summary(x, y)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(x, y, s=12, alpha=0.20, c="#1f77b4", edgecolors="none", rasterized=True, zorder=1)
    if not trend_df.empty:
        ax.fill_between(
            trend_df["x"],
            trend_df["y10"],
            trend_df["y90"],
            color="#ff7f0e",
            alpha=0.15,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            trend_df["x"],
            trend_df["y50"],
            color="#d95f02",
            linewidth=1.8,
            zorder=3,
        )
    fig.suptitle(title)
    spearman_label = f"{float(spearman):.3f}" if spearman is not None else "NA"
    subtitle = f"N={len(x):,} | Spearman rho={spearman_label} | FIMO weakest < 0.1: {100.0 * low_score_fraction:.1f}%"
    ax.set_title(subtitle, fontsize=9, color="#4d4d4d", pad=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    apply_axes_style(ax, ygrid=True, xgrid=True)
    y_min = float(y.min())
    y_max = float(y.max())
    y_span = max(y_max - y_min, 1e-9)
    ax.set_ylim(y_min - 0.02 * y_span, y_max + 0.03 * y_span)

    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
    return {
        "n_points": int(len(x)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "low_score_fraction": low_score_fraction,
        "trend_points": int(len(trend_df)),
        "title": title,
    }
