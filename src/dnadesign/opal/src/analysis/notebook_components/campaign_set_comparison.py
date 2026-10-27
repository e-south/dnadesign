"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/campaign_set_comparison.py

Notebook component builders for campaign set comparison OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import io
from typing import Any, Iterable, Mapping

from ...plots._mpl_utils import pretty_label
from .campaign_set_intervals import aggregate_center, center_label, interval_sentence, student_t_mean_ci
from .campaign_set_metric_rows import finite_number


def render_notebook_campaign_set_metric_comparison_image(
    rows: Iterable[Mapping[str, Any]],
    *,
    title: str,
    group_key: str,
    interval_kind: str = "iqr",
    confidence_level: float | None = None,
    interpretation_note: str = "",
    dpi: int = 180,
) -> dict[str, Any] | None:
    """Render a compact PNG comparison from campaign-set metric rows."""

    data = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and finite_number(row.get("value")) is not None
        and finite_number(row.get("round")) is not None
    ]
    if not data:
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    from ...plots._mpl_utils import (
        DEFAULT_SQUARE_FIGSIZE,
        apply_notebook_axes_style,
        apply_plot_style,
        apply_y_axis_scale,
        categorical_color,
        categorical_linestyle,
        categorical_marker,
        legend_below_single_row,
        wrap_plot_title,
    )

    grouped: dict[str, dict[int, list[tuple[float, str]]]] = {}
    counts_by_group_round: dict[tuple[str, int], list[float]] = {}
    metric = str(data[0].get("metric") or "value")
    cohort = str(data[0].get("cohort") or "")
    summary = str(data[0].get("summary") or "median")
    metric_label = _common_text(data, "metric_label") or _metric_axis_label(metric=metric, summary=summary)
    legend_metric_label = _common_text(data, "legend_metric_label") or metric_label
    metric_expression = _common_text(data, "metric_expression")
    collection_visual_label = _common_text(data, "collection_visual_label")
    axis_scale = _axis_scale(data)
    font_size = _common_number(data, "font_size") or 13.0
    question_title = _campaign_set_question_title(data, title=title, metric=metric)
    group_axis_label = pretty_label(group_key)
    requested_interval_kind = str(interval_kind or "none").strip()
    if requested_interval_kind not in {"none", "iqr", "student_t_mean_ci"}:
        raise ValueError(f"Unsupported campaign-set interval_kind: {requested_interval_kind!r}.")
    center = center_label(summary=summary, interval_kind=requested_interval_kind)
    pairs = {(str(row.get("metric") or ""), str(row.get("cohort") or "")) for row in data}
    if len(pairs) != 1:
        rendered = ", ".join(f"{metric_name}/{cohort_name}" for metric_name, cohort_name in sorted(pairs))
        raise ValueError(f"Campaign-set metric comparison requires exactly one metric/cohort pair; got {rendered}.")
    for row in data:
        group = str(row.get("group") or "not recorded")
        round_index = int(finite_number(row.get("round")))
        value = float(finite_number(row.get("value")))
        unit_key = str(row.get("comparison_unit_key") or row.get("campaign") or f"row-{len(data)}")
        grouped.setdefault(group, {}).setdefault(round_index, []).append((value, unit_key))
        count_value = finite_number(row.get("cohort_count"))
        if count_value is not None:
            counts_by_group_round.setdefault((group, round_index), []).append(float(count_value))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=DEFAULT_SQUARE_FIGSIZE)
    apply_notebook_axes_style(ax, square=True)
    group_labels = sorted(grouped, key=_group_sort_key)
    rounds_with_interval = 0
    interval_unit_counts: list[int] = []
    for index, group in enumerate(group_labels):
        by_round = grouped[group]
        xs = sorted(by_round)
        ys = [aggregate_center([value for value, _unit in by_round[round_index]], center=center) for round_index in xs]
        lows: list[float] = []
        highs: list[float] = []
        for round_index in xs:
            values = [value for value, _unit in by_round[round_index]]
            unit_count = len({_unit for _value, _unit in by_round[round_index]})
            if requested_interval_kind == "iqr" and len(values) >= 2 and unit_count >= 2:
                lows.append(float(np.quantile(values, 0.25)))
                highs.append(float(np.quantile(values, 0.75)))
                rounds_with_interval += 1
                interval_unit_counts.append(unit_count)
            elif requested_interval_kind == "student_t_mean_ci" and len(values) >= 2 and unit_count >= 2:
                low, high = student_t_mean_ci(values, confidence_level=confidence_level or 0.95)
                lows.append(low)
                highs.append(high)
                rounds_with_interval += 1
                interval_unit_counts.append(unit_count)
            else:
                lows.append(float("nan"))
                highs.append(float("nan"))
        mask = np.isfinite(lows) & np.isfinite(highs)
        color = categorical_color(index)
        if bool(np.any(mask)):
            x_arr = np.asarray(xs, dtype=float)
            ax.fill_between(
                x_arr[mask],
                np.asarray(lows, dtype=float)[mask],
                np.asarray(highs, dtype=float)[mask],
                color=color,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            xs,
            ys,
            color=color,
            marker=categorical_marker(index),
            linestyle=categorical_linestyle(index),
            linewidth=2.4,
            markersize=7,
            zorder=2,
            label=pretty_label(group),
        )

    ax.set_xlabel("Round", fontsize=font_size)
    y_axis_label = metric_label
    ax.set_ylabel(y_axis_label, fontsize=font_size)
    apply_y_axis_scale(
        ax,
        limits=axis_scale.get("limits"),
        reference_lines=axis_scale.get("reference_lines"),
        include_zero_tick=bool(axis_scale.get("include_zero_tick")),
    )
    rendered_title = wrap_plot_title(question_title, width=54)
    if cohort and str(cohort).lower() != "selected" and pretty_label(cohort).lower() not in rendered_title.lower():
        rendered_title = f"{rendered_title} ({pretty_label(cohort)})"
    count_text = _campaign_set_count_text(counts_by_group_round, cohort=cohort)
    if count_text:
        rendered_title = f"{rendered_title}\n{count_text}"
    ax.set_title(rendered_title, fontsize=font_size, fontweight="semibold")
    ax.set_xticks(sorted({int(finite_number(row.get("round"))) for row in data}))
    ax.tick_params(axis="both", labelsize=font_size)
    used_bottom_legend = legend_below_single_row(fig, ax, bottom=0.10)
    if not used_bottom_legend:
        fig.tight_layout(pad=0.35)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), facecolor="white")
    plt.close(fig)
    group_text = ", ".join(pretty_label(group) for group in group_labels)
    relationship_mode = any(str(row.get("pair_key") or "").strip() for row in data)
    interval_unit = "relationship pairs" if relationship_mode else "campaigns"
    interval = {
        "kind": requested_interval_kind,
        "unit": interval_unit,
        "rounds_with_interval": rounds_with_interval,
        "min_unit_count": min(interval_unit_counts) if interval_unit_counts else 0,
        "max_unit_count": max(interval_unit_counts) if interval_unit_counts else 0,
        "is_confidence_interval": requested_interval_kind == "student_t_mean_ci",
    }
    if requested_interval_kind == "student_t_mean_ci":
        interval["confidence_level"] = float(confidence_level or 0.95)
    if any(str(row.get("replicate_on") or "").strip() for row in data):
        interval["replicate_on"] = sorted({str(row.get("replicate_on")) for row in data if row.get("replicate_on")})
    interval_sentence_text = interval_sentence(
        interval_kind=requested_interval_kind,
        interval_unit=interval_unit,
        rounds_with_interval=rounds_with_interval,
        confidence_level=confidence_level or 0.95,
    )
    interpretation_sentence_text = _sentence_text(interpretation_note)
    metric_sentence_text = "" if interpretation_sentence_text else _metric_interpretation_sentence(metric)
    expression_sentence_text = _sentence_text(
        f"Configured score/loss expression: {metric_expression}" if metric_expression else ""
    )
    axis_sentence_text = _axis_scale_sentence(axis_scale)
    axis_summary = center if center == summary else f"{center} {summary}"
    rendered_caption = (
        f"{question_title}. Campaign-set comparison grouped by {group_axis_label}. "
        f"Values are {center} {metric_label} across {interval_unit} per group/round."
        f"{interval_sentence_text}{metric_sentence_text}{axis_sentence_text}"
        f"{interpretation_sentence_text}{expression_sentence_text}"
    )
    rendered_alt_text = (
        f"Campaign-set comparison for {rendered_title}. X axis is OPAL round; "
        f"Y axis is {axis_summary} {legend_metric_label} ({y_axis_label}); color, marker, and line style encode "
        f"{group_axis_label}. Groups shown: {group_text}."
        f"{interval_sentence_text}{metric_sentence_text}{axis_sentence_text}"
        f"{interpretation_sentence_text}"
    )
    return {
        "image_bytes": buffer.getvalue(),
        "alt_text": rendered_alt_text,
        "caption": rendered_caption,
        "label": collection_visual_label,
        "title": rendered_title,
        "metric_label": metric_label,
        "legend_metric_label": legend_metric_label,
        "metric_expression": metric_expression,
        "axis_scale": axis_scale,
        "group_count": len(group_labels),
        "row_count": len(data),
        "interval": interval,
    }


def _common_text(rows: Iterable[Mapping[str, Any]], key: str) -> str:
    values = {str(row.get(key) or "").strip() for row in rows if str(row.get(key) or "").strip()}
    return sorted(values)[0] if len(values) == 1 else ""


def _axis_scale(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scale_class = _common_text(rows, "axis_scale_class")
    low = _common_number(rows, "y_axis_min")
    high = _common_number(rows, "y_axis_max")
    reference_value = _common_number(rows, "y_axis_reference_value")
    reference_label = _common_text(rows, "y_axis_reference_label")
    include_zero_values = {bool(row.get("y_axis_include_zero_tick")) for row in rows}
    reference_lines = []
    if reference_value is not None:
        reference_lines.append({"value": reference_value, "label": reference_label})
    return {
        "scale_class": scale_class,
        "limits": [low, high] if low is not None or high is not None else None,
        "reference_lines": reference_lines,
        "include_zero_tick": include_zero_values == {True},
    }


def _common_number(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = {float(value) for row in rows if (value := finite_number(row.get(key))) is not None}
    return sorted(values)[0] if len(values) == 1 else None


def _campaign_set_question_title(rows: Iterable[Mapping[str, Any]], *, title: str, metric: str) -> str:
    prefix = _semantic_prefix(rows)
    phrase = _metric_title_phrase(metric=metric, fallback=title)
    return f"{prefix}: {phrase}" if prefix else phrase


def _semantic_prefix(rows: Iterable[Mapping[str, Any]]) -> str:
    target = _common_text(rows, "metadata__target_label") or _target_label(_common_text(rows, "metadata__target"))
    family = _common_text(rows, "metadata__label_family_label") or pretty_label(
        _common_text(rows, "metadata__label_family_id")
    )
    return " ".join(part for part in (target, family) if part)


def _target_label(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    known = {"cipro": "Cipro", "ethanol": "Ethanol", "dual": "Ethanol + Cipro"}
    return known.get(token.lower(), pretty_label(token))


def _metric_title_phrase(*, metric: str, fallback: str) -> str:
    if metric == "reference_mse":
        return "target-vector MSE positive/null trajectory"
    if metric == "view__selection_score":
        return "objective score positive/null trajectory"
    return str(fallback or "campaign-set comparison")


def _group_sort_key(group: str) -> tuple[int, str]:
    lowered = str(group or "").strip().lower()
    order = {"positive": 0, "null": 1, "negative": 2, "control": 3}
    return (order.get(lowered, 20), lowered)


def _campaign_set_count_text(counts_by_group_round: Mapping[tuple[str, int], list[float]], *, cohort: str) -> str:
    if not counts_by_group_round:
        return ""
    values = [
        int(round(float(value)))
        for values in counts_by_group_round.values()
        for value in values
        if finite_number(value) is not None
    ]
    if not values:
        return ""
    unique = sorted(set(values))
    cohort_text = pretty_label(cohort or "cohort")
    if len(unique) == 1:
        return f"{cohort_text} n={unique[0]} per campaign-round"
    return f"{cohort_text} n={min(unique)}-{max(unique)} per campaign-round"


def _metric_axis_label(*, metric: str, summary: str) -> str:
    if metric == "view__selection_score":
        return f"{pretty_label(summary)} objective score (campaign scale)"
    if metric == "reference_mse":
        return "MSE of selected mean vector to reference"
    return f"{pretty_label(summary)} {pretty_label(metric)}"


def _metric_interpretation_sentence(metric: str) -> str:
    if metric == "view__selection_score":
        return (
            " Selected score is on each campaign's configured objective scale; compare it within a compatible "
            "campaign set, not as a cross-family effect size."
        )
    if metric == "reference_mse":
        return (
            " Reference MSE is computed after vector aggregation: it is the MSE between the selected cohort mean "
            "vector and the declared reference vector; lower is better."
        )
    return ""


def _axis_scale_sentence(axis_scale: Mapping[str, Any]) -> str:
    limits = axis_scale.get("limits")
    references = axis_scale.get("reference_lines")
    scale_class = str(axis_scale.get("scale_class") or "").strip()
    if not limits and not references and not scale_class:
        return ""
    parts = []
    if scale_class:
        parts.append(f"axis scale class {pretty_label(scale_class)}")
    if isinstance(limits, list) and len(limits) == 2:
        low = "-inf" if limits[0] is None else f"{float(limits[0]):g}"
        high = "+inf" if limits[1] is None else f"{float(limits[1]):g}"
        parts.append(f"y-limits [{low}, {high}]")
    if references:
        labels = []
        for reference in references:
            label = str(reference.get("label") or "").strip()
            value = finite_number(reference.get("value"))
            if value is not None:
                labels.append(f"{label or 'reference'} at {value:g}")
        if labels:
            parts.append("reference line " + ", ".join(labels))
    return " Axis context: " + "; ".join(parts) + "."


def _sentence_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    clean = text.rstrip(".")
    return f" {clean}."
