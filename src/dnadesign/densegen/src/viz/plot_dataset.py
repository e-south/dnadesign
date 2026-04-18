"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_dataset.py

Dataset-native plotting for DenseGen shared source records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_common import _apply_style, _save_figure, _style

_MISSING_LABEL = "(missing)"
_OTHER_LABEL = "(other)"
_KNOWN_LABELS = {
    "usr": "USR",
    "dna": "DNA",
    "tfbs": "TFBS",
    "metadata": "Metadata",
}
_SOURCE_DESCRIPTOR_LABELS = {
    "background_only": "Background",
    "ethanol": "EtOH",
    "ciprofloxacin": "Cipro",
    "ethanol_ciprofloxacin": "EtOH + Cipro",
}
_SOURCE_PLAN_COLORS = {
    "background_only": ("#bed2f2", "#7ea0d8"),
    "ethanol": ("#bfe4c8", "#5f9b70"),
    "ciprofloxacin": ("#f8d1a6", "#d28a4b"),
    "ethanol_ciprofloxacin": ("#f1bfd8", "#c77aa7"),
}
_SOURCE_VARIANT_RE = re.compile(r"^([A-Za-z]+[0-9]*)_(.+)$")
_SOURCE_VARIANT_PREFIXES = ("sig", "variant", "rep", "batch")


def _normalize_metadata_value(value: object) -> str:
    if pd.isna(value):
        return _MISSING_LABEL
    token = str(value).strip()
    return token or _MISSING_LABEL


def _prepare_metadata_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([_MISSING_LABEL] * int(len(frame)), index=frame.index, dtype="object")
    return frame[column].map(_normalize_metadata_value)


def _format_source_label(value: object) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    if token in {_MISSING_LABEL, _OTHER_LABEL}:
        return token
    scoped_parts = [part for part in token.split("__") if part]
    if scoped_parts and scoped_parts[0].lower().replace("-", "_") == "plan_pool":
        scoped_parts = scoped_parts[1:]
    if scoped_parts:
        descriptor_parts: list[str] = []
        variant_parts: list[str] = []
        for part in scoped_parts:
            lowered = part.lower().replace("-", "_")
            descriptor_label = _SOURCE_DESCRIPTOR_LABELS.get(lowered)
            if descriptor_label is not None:
                descriptor_parts.append(descriptor_label)
                continue
            variant_match = _SOURCE_VARIANT_RE.fullmatch(part)
            if (
                variant_match
                and not lowered.startswith("plan_pool")
                and variant_match.group(1).lower().startswith(_SOURCE_VARIANT_PREFIXES)
            ):
                variant_key = str(variant_match.group(1)).strip().lower()
                variant_value = str(variant_match.group(2)).strip()
                if variant_key in {"sig35", "sigma70"}:
                    variant_parts.append(f"σ70 {variant_value}")
                else:
                    variant_parts.append(f"{variant_match.group(1)} {variant_value}")
                continue
            descriptor_parts.append(part)
        descriptor = " | ".join(
            " ".join(
                _KNOWN_LABELS.get(word.lower(), word[:1].upper() + word[1:])
                for word in part.replace("-", "_").split("_")
                if word
            )
            for part in descriptor_parts
            if part
        ).strip()
        variant = ", ".join(part for part in variant_parts if part)
        if descriptor and variant:
            return f"{descriptor} [{variant}]"
        if descriptor:
            return descriptor
        if variant:
            return f"[{variant}]"
    parts = [part for part in re.split(r"[/_:.\-\s]+", token) if part]
    if not parts:
        return token
    words: list[str] = []
    for part in parts:
        lowered = part.lower()
        if lowered in _KNOWN_LABELS:
            words.append(_KNOWN_LABELS[lowered])
        else:
            words.append(part[:1].upper() + part[1:])
    return " ".join(words)


def _source_descriptor_key(value: object) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if "__background_only__" in token:
        return "background_only"
    if "__ethanol_ciprofloxacin__" in token:
        return "ethanol_ciprofloxacin"
    if "__ciprofloxacin__" in token:
        return "ciprofloxacin"
    if "__ethanol__" in token:
        return "ethanol"
    return "background_only"


def _compact_source_axis_label(value: object) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    if token in {_MISSING_LABEL, _OTHER_LABEL}:
        return token
    scoped_parts = [part for part in token.split("__") if part]
    if scoped_parts and scoped_parts[0].lower().replace("-", "_") == "plan_pool":
        scoped_parts = scoped_parts[1:]
    descriptor_key = _source_descriptor_key(token)
    descriptor_label = {
        "background_only": "Bg",
        "ethanol": "EtOH",
        "ciprofloxacin": "Cipro",
        "ethanol_ciprofloxacin": "EtOH+Cipro",
    }.get(descriptor_key, _format_source_label(value))
    variant_tokens: list[str] = []
    for part in scoped_parts:
        lowered = part.lower().replace("-", "_")
        variant_match = _SOURCE_VARIANT_RE.fullmatch(part)
        if (
            variant_match
            and not lowered.startswith("plan_pool")
            and variant_match.group(1).lower().startswith(_SOURCE_VARIANT_PREFIXES)
        ):
            variant_key = str(variant_match.group(1)).strip().lower()
            variant_value = str(variant_match.group(2)).strip()
            if variant_key in {"sig35", "sigma70"}:
                variant_tokens.append(f"σ70{variant_value}")
            else:
                variant_tokens.append(f"{variant_match.group(1)}{variant_value}")
    if not variant_tokens:
        return descriptor_label
    return f"{descriptor_label} {', '.join(variant_tokens)}"


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


def _single_hue_heatmap_cmap() -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "densegen_dataset_seagreen",
        ["#ffffff", "#dff3ea", "#9fd6bf", "#4ca786", "#0f6b58"],
    )


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
    source_series = _prepare_metadata_series(df, "source")
    display_sources, source_order = _top_categories(source_series, limit=max_sources)
    source_counts = display_sources.value_counts().reindex(source_order, fill_value=0).sort_values(ascending=True)
    source_labels = [_compact_source_axis_label(label) for label in source_counts.index.tolist()]
    max_label_chars = max((len(label) for label in source_labels), default=12)
    fig_side = max(8.6, min(13.4, 6.4 + 0.5 * float(len(source_labels))))
    fig_height = fig_side * 1.25
    fig_width = max(7.1, (fig_side + max(1.6, min(4.2, 0.06 * float(max_label_chars)))) * 0.75)
    label_font_size = max(21.0, min(27.0, 27.0 - 0.18 * max(0, len(source_labels) - 5)))
    annotation_font_size = label_font_size
    title_font_size = annotation_font_size
    axis_font_size = annotation_font_size
    style_cfg.setdefault("tick_size", label_font_size)
    style_cfg.setdefault("label_size", axis_font_size)
    style_cfg.setdefault("title_size", title_font_size)
    fig, ax_counts = plt.subplots(
        1,
        1,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
    )

    y_positions = np.arange(len(source_labels), dtype=float)
    x_values = source_counts.values.astype(float)
    bar_colors = [
        _SOURCE_PLAN_COLORS.get(_source_descriptor_key(raw_label), _SOURCE_PLAN_COLORS["background_only"])[0]
        for raw_label in source_counts.index.tolist()
    ]
    edge_colors = [
        _SOURCE_PLAN_COLORS.get(_source_descriptor_key(raw_label), _SOURCE_PLAN_COLORS["background_only"])[1]
        for raw_label in source_counts.index.tolist()
    ]
    ax_counts.barh(
        y_positions,
        x_values,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=0.8,
        alpha=0.96,
        height=0.68,
        zorder=2.0,
    )
    ax_counts.set_yticks(y_positions)
    ax_counts.set_yticklabels(source_labels)
    _apply_style(ax_counts, style_cfg)
    ax_counts.set_title(
        "Dense arrays broken down by part-type composition",
        fontsize=title_font_size,
        pad=12.0,
    )
    ax_counts.set_xlabel("Dense array counts", fontsize=axis_font_size, labelpad=8.0)
    ax_counts.set_box_aspect(1.4)
    ax_counts.tick_params(axis="y", labelsize=label_font_size, pad=12.0)
    ax_counts.tick_params(axis="x", labelsize=axis_font_size, pad=6.0)
    for tick, raw_label in zip(ax_counts.get_yticklabels(), source_counts.index.tolist(), strict=True):
        tick.set_color(
            _SOURCE_PLAN_COLORS.get(_source_descriptor_key(raw_label), _SOURCE_PLAN_COLORS["background_only"])[1]
        )
    annotation_dx = max(2.5, float(np.max(x_values) * 0.022)) if x_values.size else 2.5
    for idx, value in enumerate(source_counts.values.tolist()):
        ax_counts.text(
            float(value) + annotation_dx,
            y_positions[idx],
            f"{int(value):,}",
            va="center",
            ha="left",
            fontsize=annotation_font_size,
            color="#5d6670",
        )
    ax_counts.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.22)

    left_margin = max(0.24, min(0.38, 0.14 + 0.0092 * float(max_label_chars)))
    fig.subplots_adjust(left=left_margin, right=0.98, bottom=0.12, top=0.92)

    out = out_path.parent / "dataset" / out_path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out, style=style_cfg)
    plt.close(fig)
    return [out]


def plot_source_cohort_concentration(
    df: pd.DataFrame,
    out_path: Path,
    *,
    style: dict | None = None,
    max_sources: int = 24,
) -> list[Path]:
    return plot_dataset_source_inventory(
        df,
        out_path,
        style=style,
        max_sources=max_sources,
    )


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
    style_cfg.setdefault("figsize", (15.2, 7.2))
    heatmap_style = dict(style_cfg)
    heatmap_style["grid"] = False

    source_series = _prepare_metadata_series(df, "source")
    plan_series = _prepare_metadata_series(df, "densegen__plan")
    input_series = _prepare_metadata_series(df, "densegen__input_name")
    display_sources, source_order = _top_categories(source_series, limit=max_sources)
    plan_values, plan_order = _top_categories(plan_series, limit=max_plans)
    input_values, input_order = _top_categories(input_series, limit=max_inputs)
    table_plan = pd.crosstab(display_sources, plan_values).reindex(index=source_order, columns=plan_order, fill_value=0)
    table_input = pd.crosstab(display_sources, input_values).reindex(
        index=source_order, columns=input_order, fill_value=0
    )

    tables = [
        (
            table_plan,
            "DenseGen plans draw from a narrow set of source cohorts",
            "DenseGen plan",
        ),
        (
            table_input,
            "DenseGen inputs mostly mirror that same source structure",
            "DenseGen input name",
        ),
    ]
    log_tables = [np.log1p(table.to_numpy(dtype=float)) for table, _title, _x_label in tables]
    vmax = max(float(np.nanmax(values)) for values in log_tables) if log_tables else 1.0
    vmax = max(1.0, vmax)
    max_source_chars = max((len(_format_source_label(label)) for label in source_order), default=12)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=tuple(style_cfg.get("figsize", (15.5, 6.2))),
        sharey=True,
        constrained_layout=False,
    )

    for idx, (ax, (table, title, x_label)) in enumerate(zip(axes, tables, strict=True)):
        image = ax.imshow(
            np.log1p(table.to_numpy(dtype=float)),
            aspect="auto",
            cmap=_single_hue_heatmap_cmap(),
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Source cohort" if idx == 0 else "")
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels(table.columns.tolist(), rotation=55, ha="right")
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels([_format_source_label(label) for label in table.index.tolist()])
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False, left=False)
        _apply_style(ax, heatmap_style)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.065, pad=0.028)
        colorbar.set_label(
            "log1p(record count)",
            size=float(style_cfg.get("label_size", style_cfg.get("font_size", 13))),
        )
        colorbar.ax.tick_params(labelsize=float(style_cfg.get("tick_size", style_cfg.get("font_size", 13))))
        if table.shape[0] * table.shape[1] <= 144:
            max_value = max(1, int(table.to_numpy(dtype=int).max()))
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
                        fontsize=8.4,
                        color="white" if value >= max_value * 0.45 else "#0b1f17",
                    )

    recovery_count = 0
    if "densegen__metadata_inferred_from_source" in df.columns:
        recovery_count = int(pd.Series(df["densegen__metadata_inferred_from_source"]).fillna(False).astype(bool).sum())
    footer_lines = []
    if recovery_count > 0:
        footer_lines.append(
            f"`densegen__plan`/`densegen__input_name` were recovered from `source` for {recovery_count:,} rows."
        )
    footer_lines.append(
        f"`{_MISSING_LABEL}` marks rows without explicit DenseGen metadata. "
        f"`{_OTHER_LABEL}` aggregates categories outside the display budget."
    )
    fig.suptitle(
        "Source cohorts mostly map onto the same DenseGen plan and input structure",
        fontsize=float(style_cfg.get("title_size", 14)),
    )
    fig.text(
        0.5,
        0.01,
        "\n".join(footer_lines),
        ha="center",
        va="bottom",
        fontsize=10,
    )
    left_margin = max(0.2, min(0.38, 0.12 + 0.0065 * float(max_source_chars)))
    fig.subplots_adjust(left=left_margin, right=0.98, bottom=0.25, top=0.86, wspace=0.26)

    out = out_path.parent / "dataset" / out_path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out, style=style_cfg)
    plt.close(fig)
    return [out]


def plot_source_plan_input_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    *,
    style: dict | None = None,
    max_sources: int = 24,
    max_plans: int = 24,
    max_inputs: int = 24,
) -> list[Path]:
    return plot_dataset_metadata_heatmap(
        df,
        out_path,
        style=style,
        max_sources=max_sources,
        max_plans=max_plans,
        max_inputs=max_inputs,
    )
