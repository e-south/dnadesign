"""Shared helpers for Stage B realized-review plot renderers."""

from __future__ import annotations

from textwrap import fill

import pandas as pd

from ....plot_style import REVIEW_LEGEND_FONTSIZE
from .display_text import role_display_label

_SUBTITLE_Y = 0.850


def plot_title(text: str) -> str:
    return fill(str(text), width=48, break_long_words=False)


def title_x(layout: dict[str, float]) -> float:
    return (float(layout["left"]) + float(layout["right"])) / 2.0


def subtitle_y_for_title(title: str) -> float:
    wrapped_lines = max(1, str(title).count("\n") + 1)
    return _SUBTITLE_Y - 0.034 * max(0, wrapped_lines - 1)


def single_nonempty(values: object) -> str:
    series = pd.Series(values, dtype="object")
    clean = sorted({str(value) for value in series.tolist() if str(value) not in {"", "nan", "None"}})
    return clean[0] if len(clean) == 1 else ""


def control_role_for_label(frame: pd.DataFrame) -> str:
    if "null_control_role" not in frame.columns:
        return ""
    if "oracle_role" not in frame.columns:
        return single_nonempty(frame["null_control_role"])
    matched_null = frame.loc[frame["oracle_role"].astype(str) == "matched_null", "null_control_role"]
    role = single_nonempty(matched_null)
    if role:
        return role
    return single_nonempty(frame["null_control_role"])


def legend_role_label(role: object, *, label_name: object, control_role: object | None = None) -> str:
    role_text = str(role)
    if role_text == "matched_null" and str(control_role or "") == "count_fixed_shuffled_slot_negative_control":
        return "Slot-shuffled control"
    return role_display_label(role, label_name=label_name, control_role=control_role)


def legend_below_figure(fig: object, ax: object, *, ncols: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    column_count = min(int(ncols), max(1, len(by_label)), 2)
    fig.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        fontsize=REVIEW_LEGEND_FONTSIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.060),
        ncols=column_count,
        columnspacing=0.6,
        handlelength=1.0,
        handletextpad=0.35,
    )


def same_batch_top_k_lift(frame: pd.DataFrame) -> float:
    """Return the plotted same-batch top-label lift for a label trajectory."""

    positive = frame.loc[frame["oracle_role"].astype(str) == "positive", "same_batch_top_lift_ratio"]
    source = positive if not positive.empty else frame["same_batch_top_lift_ratio"]
    values = pd.to_numeric(source, errors="raise").dropna()
    if values.empty:
        raise ValueError("Stage B lift trajectory has no same_batch_top_lift_ratio values")
    return float(values.mean())
