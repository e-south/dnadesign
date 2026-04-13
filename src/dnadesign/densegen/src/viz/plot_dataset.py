"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_dataset.py

Dataset-native plotting for DenseGen shared source records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker as mticker

from .plot_common import _apply_style, _palette, _save_figure, _style

_MISSING_LABEL = "(missing)"
_OTHER_LABEL = "(other)"


def _normalize_metadata_value(value: object) -> str:
    if pd.isna(value):
        return _MISSING_LABEL
    token = str(value).strip()
    return token or _MISSING_LABEL


def _prepare_metadata_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([_MISSING_LABEL] * int(len(frame)), index=frame.index, dtype="object")
    return frame[column].map(_normalize_metadata_value)


def _top_categories(series: pd.Series, *, limit: int) -> tuple[pd.Series, list[str]]:
    if int(limit) <= 0:
        raise ValueError("top-category limit must be > 0")
    normalized = series.map(_normalize_metadata_value)
    counts = normalized.value_counts(dropna=False)
    if counts.empty:
        return normalized, [_MISSING_LABEL]
    if len(counts) <= int(limit):
        return normalized, [str(item) for item in counts.index.tolist()]
    keep = [str(item) for item in counts.head(int(limit)).index.tolist()]
    collapsed = normalized.where(normalized.isin(set(keep)), other=_OTHER_LABEL)
    return collapsed, [*keep, _OTHER_LABEL]


def _reduced_crosstab(
    row_series: pd.Series,
    col_series: pd.Series,
    *,
    max_rows: int,
    max_cols: int,
) -> pd.DataFrame:
    row_values, row_order = _top_categories(row_series, limit=max_rows)
    col_values, col_order = _top_categories(col_series, limit=max_cols)
    table = pd.crosstab(row_values, col_values)
    return table.reindex(index=row_order, columns=col_order, fill_value=0)


def plot_dataset_source_inventory(
    df: pd.DataFrame,
    out_path: Path,
    *,
    style: dict | None = None,
    max_sources: int = 24,
) -> list[Path]:
    if df is None or df.empty:
        raise ValueError("dataset_source_inventory requires output records.")
    if "source" not in df.columns:
        raise ValueError("dataset_source_inventory requires `source` in output records.")

    style_cfg = _style(style)
    style_cfg.setdefault("figsize", (12.0, 5.8))
    total_rows = int(len(df))
    source_series = _prepare_metadata_series(df, "source")
    display_sources, source_order = _top_categories(source_series, limit=max_sources)
    source_counts = display_sources.value_counts().reindex(source_order, fill_value=0).sort_values(ascending=True)

    metrics = [
        ("source", total_rows, int(source_series.nunique(dropna=False))),
        (
            "densegen__plan",
            int(df["densegen__plan"].notna().sum()) if "densegen__plan" in df.columns else 0,
            int(df["densegen__plan"].dropna().astype(str).nunique()) if "densegen__plan" in df.columns else 0,
        ),
        (
            "densegen__input_name",
            int(df["densegen__input_name"].notna().sum()) if "densegen__input_name" in df.columns else 0,
            int(df["densegen__input_name"].dropna().astype(str).nunique())
            if "densegen__input_name" in df.columns
            else 0,
        ),
    ]

    fig, (ax_counts, ax_coverage) = plt.subplots(
        1,
        2,
        figsize=tuple(style_cfg.get("figsize", (12.0, 5.8))),
        gridspec_kw={"width_ratios": [1.9, 1.1]},
        constrained_layout=False,
    )

    colors = list(reversed(_palette(style_cfg, len(source_counts))))
    ax_counts.barh(source_counts.index.tolist(), source_counts.values.tolist(), color=colors)
    ax_counts.set_title("Rows by source")
    ax_counts.set_xlabel("Rows")
    for idx, value in enumerate(source_counts.values.tolist()):
        ax_counts.text(float(value), idx, f" {int(value):,}", va="center", ha="left", fontsize=10)
    _apply_style(ax_counts, style_cfg)

    metric_labels = [label for label, _present, _unique in metrics][::-1]
    metric_fractions = [present / total_rows if total_rows else 0.0 for _label, present, _unique in metrics][::-1]
    coverage_colors = list(reversed(_palette(style_cfg, len(metric_labels))))
    ax_coverage.barh(metric_labels, metric_fractions, color=coverage_colors)
    ax_coverage.set_xlim(0.0, 1.05)
    ax_coverage.set_title("Metadata coverage")
    ax_coverage.set_xlabel("Fraction of rows")
    ax_coverage.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    for idx, (label, present, unique_values) in enumerate(metrics[::-1]):
        fraction = present / total_rows if total_rows else 0.0
        ax_coverage.text(
            min(1.02, fraction + 0.02),
            idx,
            f"{present:,}/{total_rows:,} rows | {unique_values:,} unique",
            va="center",
            ha="left",
            fontsize=10,
        )
    _apply_style(ax_coverage, style_cfg)

    fig.suptitle("DenseGen dataset source inventory", fontsize=float(style_cfg.get("title_size", 14)))
    fig.text(
        0.5,
        0.01,
        f"Total rows: {total_rows:,}. `{_MISSING_LABEL}` indicates records without explicit DenseGen metadata; "
        f"`{_OTHER_LABEL}` aggregates categories outside the top-{int(max_sources)} display budget.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.18, wspace=0.35)

    out = out_path.parent / "dataset" / out_path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out, style=style_cfg)
    plt.close(fig)
    return [out]


def plot_dataset_metadata_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    *,
    style: dict | None = None,
    max_sources: int = 24,
    max_plans: int = 24,
    max_inputs: int = 24,
) -> list[Path]:
    if df is None or df.empty:
        raise ValueError("dataset_metadata_heatmap requires output records.")
    if "source" not in df.columns:
        raise ValueError("dataset_metadata_heatmap requires `source` in output records.")

    style_cfg = _style(style)
    style_cfg.setdefault("figsize", (15.5, 6.2))
    heatmap_style = dict(style_cfg)
    heatmap_style["grid"] = False

    source_series = _prepare_metadata_series(df, "source")
    plan_series = _prepare_metadata_series(df, "densegen__plan")
    input_series = _prepare_metadata_series(df, "densegen__input_name")

    tables = [
        (
            _reduced_crosstab(source_series, plan_series, max_rows=max_sources, max_cols=max_plans),
            "Source -> densegen__plan",
            "densegen__plan",
        ),
        (
            _reduced_crosstab(source_series, input_series, max_rows=max_sources, max_cols=max_inputs),
            "Source -> densegen__input_name",
            "densegen__input_name",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=tuple(style_cfg.get("figsize", (15.5, 6.2))),
        constrained_layout=False,
    )

    for ax, (table, title, x_label) in zip(axes, tables):
        image = ax.imshow(np.log1p(table.to_numpy(dtype=float)), aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("source")
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels(table.columns.tolist(), rotation=70, ha="right")
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels(table.index.tolist())
        _apply_style(ax, heatmap_style)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label("log1p(row count)")
        if table.shape[0] * table.shape[1] <= 144:
            for row_idx in range(table.shape[0]):
                for col_idx in range(table.shape[1]):
                    value = int(table.iat[row_idx, col_idx])
                    if value <= 0:
                        continue
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{value:,}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if value >= table.to_numpy().max() * 0.45 else "#111111",
                    )

    fig.suptitle("DenseGen dataset metadata heatmaps", fontsize=float(style_cfg.get("title_size", 14)))
    fig.text(
        0.5,
        0.01,
        f"`{_MISSING_LABEL}` marks rows without explicit DenseGen metadata. "
        f"`{_OTHER_LABEL}` aggregates categories outside the display budget.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.42)

    out = out_path.parent / "dataset" / out_path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out, style=style_cfg)
    plt.close(fig)
    return [out]
