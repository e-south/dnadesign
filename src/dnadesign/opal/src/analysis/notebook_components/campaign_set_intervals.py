"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/campaign_set_intervals.py

Notebook component builders for campaign set intervals OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def center_label(*, summary: str, interval_kind: str) -> str:
    if interval_kind == "student_t_mean_ci" or str(summary).strip().lower() == "mean":
        return "mean"
    return "median"


def aggregate_center(values: list[float], *, center: str) -> float:
    import numpy as np

    if center == "mean":
        return float(np.mean(values))
    return float(np.median(values))


def student_t_mean_ci(values: list[float], *, confidence_level: float) -> tuple[float, float]:
    import numpy as np
    from scipy import stats

    if len(values) < 2:
        return float("nan"), float("nan")
    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = float(stats.sem(arr))
    half_width = float(stats.t.ppf((1.0 + level) / 2.0, len(arr) - 1) * sem)
    return mean - half_width, mean + half_width


def interval_sentence(
    *,
    interval_kind: str,
    interval_unit: str,
    rounds_with_interval: int,
    confidence_level: float,
) -> str:
    if interval_kind == "none":
        return " No interval band is requested for this comparison view."
    if rounds_with_interval <= 0:
        return " No interval band is drawn when fewer than two comparison units contribute per group/round."
    if interval_kind == "student_t_mean_ci":
        percent = round(float(confidence_level) * 100.0)
        return (
            f" Shaded bands are {percent}% Student-t mean confidence intervals across {interval_unit} "
            "where at least two units contribute."
        )
    return (
        f" Shaded bands are IQR across {interval_unit} where at least two units contribute; "
        "they are not statistical confidence intervals."
    )
