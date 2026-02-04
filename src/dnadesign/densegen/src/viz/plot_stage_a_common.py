"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_a_common.py

Shared helpers for Stage-A summary plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.colors import to_rgba

from .plot_common import _palette


def _pastelize_color(color: str, amount: float = 0.6) -> tuple[float, float, float, float]:
    base = to_rgba(color)
    return (
        base[0] + (1.0 - base[0]) * amount,
        base[1] + (1.0 - base[1]) * amount,
        base[2] + (1.0 - base[2]) * amount,
        base[3],
    )


def _stage_a_text_sizes(style: dict) -> dict[str, float]:
    font_size = float(style.get("font_size", 12.0))
    label_size = float(style.get("label_size", font_size))
    panel_title = float(style.get("title_size", font_size * 1.15))
    fig_title = float(style.get("fig_title_size", panel_title * 1.15))
    regulator_label = float(style.get("regulator_label_size", label_size * 0.95))
    sublabel = float(style.get("sublabel_size", label_size * 0.8))
    annotation = float(style.get("annotation_size", label_size * 0.72))
    return {
        "fig_title": fig_title,
        "panel_title": panel_title,
        "regulator_label": regulator_label,
        "sublabel": sublabel,
        "annotation": annotation,
    }


def _stage_a_regulator_colors(regulators: list[str], style: dict) -> dict[str, str]:
    base = _palette(style, max(len(regulators), 6), no_repeat=False)
    special = {"lexa": "#0072B2", "cpxr": "#009E73"}
    color_by_reg: dict[str, str] = {}
    used: set[str] = set()
    for reg in regulators:
        lowered = str(reg).strip().lower()
        if lowered.startswith("lexa"):
            color_by_reg[reg] = special["lexa"]
            used.add(special["lexa"])
        elif lowered.startswith("cpxr"):
            color_by_reg[reg] = special["cpxr"]
            used.add(special["cpxr"])
    available = [color for color in base if color not in used]
    if not available:
        available = list(base)
    idx = 0
    for reg in regulators:
        if reg in color_by_reg:
            continue
        color_by_reg[reg] = available[idx % len(available)]
        idx += 1
    return color_by_reg
