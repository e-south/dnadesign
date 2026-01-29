"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_common.py

Shared plotting helpers and styling utilities for DenseGen plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Embed TrueType fonts for clean text in vector exports
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _dg(col: str) -> str:
    return col if col.startswith("densegen__") else f"densegen__{col}"


def _style(style: Optional[dict]) -> dict:
    s = (style or {}).copy()
    s.setdefault("seaborn_style", True)
    s.setdefault("despine", True)
    s.setdefault("legend_frame", False)
    s.setdefault("figsize", (8, 4))
    s.setdefault("palette", "okabe_ito")
    s.setdefault("font_size", 13)
    return s


def _format_plot_path(path: Path, run_root: Path, absolute: bool) -> str:
    if absolute:
        return str(path)
    try:
        return str(path.relative_to(run_root))
    except ValueError:
        return str(path)


def _format_source_label(label: str, run_root: Path, absolute: bool) -> str:
    if ":" not in label:
        return label
    prefix, raw = label.split(":", 1)
    raw = raw.strip()
    if not raw:
        return label
    try:
        path = Path(raw)
    except Exception:
        return label
    return f"{prefix}:{_format_plot_path(path, run_root, absolute)}"


def _apply_style(ax, style: dict):
    if style.get("seaborn_style", True):
        applied = False
        for name in ("seaborn-v0_8-ticks", "seaborn-ticks"):
            try:
                plt.style.use(name)
                applied = True
                break
            except Exception:
                continue
        if not applied:
            raise ValueError(
                "seaborn_style is true but no seaborn style is available. "
                "Install matplotlib styles that include seaborn or set seaborn_style: false."
            )
    if style.get("despine", True):
        if "top" in ax.spines:
            ax.spines["top"].set_visible(False)
        if "right" in ax.spines:
            ax.spines["right"].set_visible(False)
    fs = float(style.get("font_size", 13))
    ax.tick_params(axis="both", labelsize=float(style.get("tick_size", fs * 0.9)))
    ax.xaxis.label.set_size(float(style.get("label_size", fs)))
    ax.yaxis.label.set_size(float(style.get("label_size", fs)))
    ax.title.set_size(float(style.get("title_size", fs * 1.1)))
    lg = ax.get_legend()
    if lg is not None:
        lg.set_frame_on(bool(style.get("legend_frame", False)))


def _fig_ax(style: dict):
    w, h = style.get("figsize", (8, 4))
    return plt.subplots(figsize=(float(w), float(h)))


def _safe_filename(text: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", str(text).strip())
    return cleaned or "densegen"


def _format_percent(value: float) -> str:
    if value >= 0.10:
        return f"{value * 100:.0f}%"
    if value >= 0.01:
        return f"{value * 100:.1f}%"
    return f"{value * 100:.2f}%"


def _add_anchored_box(
    ax: mpl.axes.Axes,
    lines: list[str],
    *,
    loc: str = "upper right",
    fontsize: float = 9.0,
    alpha: float = 0.9,
    edgecolor: str | None = "#dddddd",
) -> AnchoredText | None:
    if not lines:
        return None
    text = "\n".join(lines)
    box = AnchoredText(text, loc=loc, prop={"size": fontsize}, frameon=True, pad=0.35, borderpad=0.4)
    box.patch.set_alpha(alpha)
    if edgecolor is None or str(edgecolor).lower() in {"none", "transparent"}:
        box.patch.set_edgecolor("none")
        box.patch.set_linewidth(0.0)
    else:
        box.patch.set_edgecolor(edgecolor)
    box.patch.set_facecolor("white")
    ax.add_artist(box)
    return box


def _draw_tier_markers(
    ax: mpl.axes.Axes,
    thresholds: list[tuple[str, float | None, str | None]],
    *,
    ymax_fraction: float = 0.58,
    label_mode: str = "box",
    loc: str = "upper right",
    fontsize: float | None = None,
) -> None:
    cleaned: list[tuple[str, float, str | None]] = []
    for label, value, label_text in thresholds:
        if value is None:
            continue
        cleaned.append((label, float(value), label_text))
    if not cleaned:
        return
    for _, value, _ in cleaned:
        ax.axvline(value, ymin=0.0, ymax=ymax_fraction, linestyle="--", linewidth=1, alpha=0.85, color="#222222")
    if label_mode == "box":
        lines = []
        for label, value, label_text in cleaned:
            value_text = f"{value:.2f}"
            if label_text:
                lines.append(f"{label}: {value_text} (n={label_text})")
            else:
                lines.append(f"{label}: {value_text}")
        y_min, y_max = ax.get_ylim()
        y_top = y_min + (y_max - y_min) * ymax_fraction
        for label, value, _ in cleaned:
            ax.scatter([value], [y_top], s=16, color="#222222", edgecolors="none", zorder=4)
            ax.annotate(
                label,
                (value, y_top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=(fontsize or float(ax.yaxis.label.get_size()) * 0.74) * 0.9,
                color="#222222",
            )
        box_size = fontsize or float(ax.yaxis.label.get_size()) * 0.74
        _add_anchored_box(ax, lines, loc=loc, fontsize=box_size, alpha=0.9, edgecolor="none")


def _shared_axis_cleanup(axes: list[mpl.axes.Axes]) -> None:
    if len(axes) < 2:
        return
    for ax in axes[:-1]:
        ax.label_outer()
        ax.tick_params(labelbottom=False)


def _shared_x_cleanup(axes: list[mpl.axes.Axes]) -> None:
    if len(axes) < 2:
        return
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
