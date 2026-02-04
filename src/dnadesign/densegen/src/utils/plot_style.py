"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/densegen/utils/plot_style.py

Plot styling helpers for Stage-A summary figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

_IUPAC_RE = re.compile(r"^[ACGTRYKMSWBDHVN]+$", re.IGNORECASE)


def stage_a_rcparams(style: dict | None = None) -> dict[str, Any]:
    settings = (style or {}).copy()
    font_size = float(settings.get("font_size", 12.0))
    tick_size = float(settings.get("tick_size", font_size * 0.9))
    label_size = float(settings.get("label_size", font_size))
    title_size = float(settings.get("title_size", font_size * 1.15))
    line_width = float(settings.get("line_width", 1.2))
    marker_size = float(settings.get("marker_size", 20.0))
    grid_alpha = float(settings.get("grid_alpha", 0.2))
    grid_width = float(settings.get("grid_linewidth", 0.6))
    dpi = float(settings.get("dpi", 160))
    return {
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "lines.linewidth": line_width,
        "lines.markersize": marker_size,
        "axes.grid": False,
        "grid.alpha": grid_alpha,
        "grid.linewidth": grid_width,
        "figure.dpi": dpi,
    }


def format_regulator_label(tf_id: str, wrap_width: int = 20) -> str:
    text = str(tf_id or "").strip()
    if not text:
        return ""
    if "_" in text:
        head, tail = text.split("_", 1)
        if tail and _IUPAC_RE.fullmatch(tail):
            return f"{head}\n{tail}"
    cleaned = text.replace("_", " ")
    if wrap_width and len(cleaned) > wrap_width:
        return "\n".join(textwrap.wrap(cleaned, width=wrap_width))
    return cleaned
