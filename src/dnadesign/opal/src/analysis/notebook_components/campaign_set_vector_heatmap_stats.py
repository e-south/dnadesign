"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/campaign_set_vector_heatmap_stats.py

Notebook component builders for campaign set vector heatmap stats OPAL analysis.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from typing import Any, Iterable, Mapping

from ...plots._mpl_utils import pretty_label, pretty_title
from .campaign_set_intervals import aggregate_center
from .campaign_set_sources import finite_number


def heatmap_values(
    rows: Iterable[Mapping[str, Any]],
    *,
    groups: list[str],
    rounds: list[int],
    channels: list[str],
) -> dict[tuple[str, int, str], float]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    allowed = {(group, round_index, channel) for group in groups for round_index in rounds for channel in channels}
    for row in rows:
        key = (str(row.get("group") or "not recorded"), int(finite_number(row.get("round"))), str(row.get("channel")))
        value = finite_number(row.get("value"))
        if key in allowed and value is not None:
            grouped[key].append(float(value))
    return {key: aggregate_center(values, center="mean") for key, values in grouped.items()}


def target_mse_values(
    rows: Iterable[Mapping[str, Any]],
    *,
    reference: Mapping[str, float],
    channels: list[str],
) -> dict[tuple[str, int], list[float]]:
    by_unit: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = finite_number(row.get("value"))
        round_value = finite_number(row.get("round"))
        if value is None or round_value is None:
            continue
        group = str(row.get("group") or "not recorded")
        unit = str(row.get("comparison_unit_key") or row.get("campaign") or "")
        channel = str(row.get("channel") or "")
        if channel in channels:
            by_unit[(group, int(round_value), unit)][channel] = float(value)
    out: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (group, round_index, _unit), values_by_channel in by_unit.items():
        if any(channel not in values_by_channel for channel in channels):
            continue
        mse = sum((values_by_channel[channel] - reference[channel]) ** 2 for channel in channels) / len(channels)
        out[(group, round_index)].append(float(mse))
    return out


def reference_vector(rows: Iterable[Mapping[str, Any]], *, channels: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        channel = str(row.get("channel") or "")
        value = finite_number(row.get("value"))
        if channel in channels and value is not None and channel not in out:
            out[channel] = float(value)
    return out


def reference_mse_axis_fields(params: Mapping[str, Any]) -> dict[str, Any]:
    low, high = axis_limits(params.get("reference_mse_y_limits", params.get("reference_mse_limits")))
    first_reference = first_reference_line(params.get("reference_mse_reference_lines"))
    return {
        "axis_scale_class": str(params.get("reference_mse_scale_class") or "reference_mse").strip(),
        "y_axis_min": low,
        "y_axis_max": high,
        "y_axis_reference_value": first_reference.get("value"),
        "y_axis_reference_label": first_reference.get("label", ""),
        "y_axis_include_zero_tick": bool(params.get("reference_mse_include_zero_tick", True)),
    }


def vector_axis_scale(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scale_class = common_text(rows, "axis_scale_class")
    low = common_number(rows, "y_axis_min")
    high = common_number(rows, "y_axis_max")
    reference_value = common_number(rows, "y_axis_reference_value")
    reference_label = common_text(rows, "y_axis_reference_label")
    reference_lines = []
    if reference_value is not None:
        reference_lines.append({"value": reference_value, "label": reference_label})
    return {
        "scale_class": scale_class,
        "limits": [low, high] if low is not None or high is not None else None,
        "reference_lines": reference_lines,
        "include_zero_tick": {bool(row.get("y_axis_include_zero_tick")) for row in rows} == {True},
    }


def axis_limits(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    return finite_number(value[0]), finite_number(value[1])


def first_reference_line(value: Any) -> dict[str, Any]:
    if not isinstance(value, list) or not value:
        return {}
    first = value[0]
    if not isinstance(first, Mapping):
        return {}
    reference_value = finite_number(first.get("value"))
    if reference_value is None:
        return {}
    return {
        "value": reference_value,
        "label": str(first.get("label") or ""),
    }


def heatmap_domain(rows: Iterable[Mapping[str, Any]], reference_values: Iterable[float]) -> tuple[float, float]:
    values = [
        float(value)
        for value in [*(finite_number(row.get("value")) for row in rows), *reference_values]
        if value is not None
    ]
    if not values:
        return 0.0, 1.0
    return min(0.0, min(values)), max(1.0, max(values))


def ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def common_text(rows: Iterable[Mapping[str, Any]], key: str) -> str:
    values = {str(row.get(key) or "").strip() for row in rows if str(row.get(key) or "").strip()}
    return sorted(values)[0] if len(values) == 1 else ""


def common_number(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = {float(value) for row in rows if (value := finite_number(row.get(key))) is not None}
    return sorted(values)[0] if len(values) == 1 else None


def sentence_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " " + text.rstrip(".") + "."


def role_sort_key(value: str) -> tuple[int, str]:
    normalized = str(value).lower()
    if normalized == "positive":
        return (0, normalized)
    if normalized == "null":
        return (1, normalized)
    return (2, normalized)


def mse_formula(metric_expression: str) -> str:
    expression = str(metric_expression or "").strip()
    if not expression:
        return "MSE = mean_c((mean selected y_hat_c - target_c)^2)"
    return expression.split(";", maxsplit=1)[0].strip()


def target_vector_label(reference: Mapping[str, float], *, channels: list[str]) -> str:
    parts = []
    for channel in channels:
        value = float(reference[channel])
        token = str(int(value)) if value.is_integer() else f"{value:g}"
        parts.append(f"{channel}={token}")
    return "target vector [" + ", ".join(parts) + "]"


def plot_question(rows: list[Mapping[str, Any]], *, target_vector_label: str) -> str:
    target = common_text(rows, "metadata__target_label") or common_text(rows, "metadata__target")
    label_family = common_text(rows, "metadata__label_family_label") or common_text(rows, "metadata__label_family_id")
    target_text = pretty_title(target) if target else "Vector target"
    family_text = pretty_label(label_family) if label_family else "vector objective"
    if target_vector_label:
        return f"{target_text} {family_text}: positive vs null selected-vector convergence"
    return f"{target_text} {family_text}: do positive-oracle selections outperform null?"


def wrap_no_ellipsis(value: str, *, width: int) -> str:
    lines = textwrap.wrap(
        str(value or "").strip(),
        width=max(int(width), 12),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else ""
