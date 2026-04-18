"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_a_common.py

Shared helpers for Stage-A summary plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def _is_background_regulator(label: str) -> bool:
    norm = str(label).strip().lower().replace("-", "_")
    if not norm:
        return False
    if norm in {"background", "background_pool", "neutral_bg"}:
        return True
    return norm.startswith("background_")


def _stage_a_pool_regulator_column(pool_df: pd.DataFrame, *, input_name: str) -> str:
    if "regulator_id" in pool_df.columns:
        return "regulator_id"
    if "tf" in pool_df.columns:
        return "tf"
    raise ValueError(f"Stage-A pool missing regulator_id or tf column for input '{input_name}'.")


def _stage_a_pool_tfbs_column(pool_df: pd.DataFrame, *, input_name: str) -> str:
    if "tfbs_sequence" in pool_df.columns:
        return "tfbs_sequence"
    if "tfbs" in pool_df.columns:
        return "tfbs"
    raise ValueError(f"Stage-A pool missing tfbs_sequence or tfbs column for input '{input_name}'.")


def _stage_a_non_background_sampling_rows(input_name: str, sampling: dict) -> list[dict]:
    eligible_hist = sampling.get("eligible_score_hist")
    if not isinstance(eligible_hist, list) or not eligible_hist:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
    rows: list[dict] = []
    for row in eligible_hist:
        if not isinstance(row, dict):
            raise ValueError(f"Stage-A sampling has invalid eligible score entry for input '{input_name}'.")
        regulator = row.get("regulator")
        if regulator is None:
            raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
        if _is_background_regulator(str(regulator)):
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"Stage-A sampling missing non-background regulator labels for input '{input_name}'.")
    return rows


def _stage_a_regulator_order(input_name: str, sampling: dict) -> list[str]:
    return [str(row["regulator"]) for row in _stage_a_non_background_sampling_rows(input_name, sampling)]


def _stage_a_retained_tfbs_lengths_by_regulator(
    pool_df: pd.DataFrame,
    *,
    input_name: str,
    regulators: list[str],
) -> dict[str, list[int]]:
    tf_col = _stage_a_pool_regulator_column(pool_df, input_name=input_name)
    tfbs_col = _stage_a_pool_tfbs_column(pool_df, input_name=input_name)
    allowed = set(regulators)
    lengths_by_reg = {reg: [] for reg in regulators}
    for regulator, sequence in pool_df[[tf_col, tfbs_col]].itertuples(index=False):
        reg = str(regulator)
        if reg not in allowed or pd.isna(sequence):
            continue
        lengths_by_reg[reg].append(len(str(sequence)))
    return lengths_by_reg


def _stage_a_hist_centers(edges: list[float] | np.ndarray) -> np.ndarray:
    edges_arr = np.asarray(edges, dtype=float)
    if edges_arr.ndim != 1 or edges_arr.size < 2:
        raise ValueError("Stage-A histogram edges must be one-dimensional with at least two values.")
    return (edges_arr[:-1] + edges_arr[1:]) / 2.0
