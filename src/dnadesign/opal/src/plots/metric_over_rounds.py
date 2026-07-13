"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/metric_over_rounds.py

Generic scalar-over-rounds plot primitive for OPAL ledger predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ..registries.plots import PlotMeta, register_plot
from ._cohort_utils import positive_ranks, selected_mask
from ._events_util import load_events, resolve_outputs_dir
from ._mpl_utils import (
    DEFAULT_SQUARE_FIGSIZE,
    apply_notebook_axes_style,
    apply_plot_style,
    apply_y_axis_scale,
    categorical_color,
    categorical_style,
    ensure_mpl_config_dir,
    legend_below_single_row,
    plot_metric_label,
    plot_metric_short_label,
    pretty_label,
    pretty_title,
    save_notebook_square_figure,
)
from ._param_utils import get_str, normalize_metric_field
from ._round_overlay import resolve_highlight_round


@register_plot(
    "metric_over_rounds",
    meta=PlotMeta(
        summary="Generic scalar metric summaries over rounds by cohort.",
        params={
            "metric": "Numeric ledger field to summarize (default view__selection_score).",
            "cohort": "Cohort or list of cohorts: selected, top_k, all_pool (default selected).",
            "top_k": "Rank cutoff for top_k cohort (default 10).",
            "summaries": "Summary or list: mean, median, count, q10, q25, q75, q90.",
            "band": "Optional uncertainty band for matching summaries: none|iqr (default none).",
            "band_alpha": "Transparency for band='iqr' (default 0.18).",
            "threshold": "Optional horizontal threshold/reference line.",
            "highlight_round": "Optional point marker: latest, true, false, or an integer.",
        },
        requires=["as_of_round", "run_id", "view__selection_score"],
        notes=["Reads outputs/ledger/predictions and writes tidy scalar summaries when save_data is enabled."],
        data_shape="scalar over rounds",
        tidy_schema=["round", "cohort", "metric", "summary", "value"],
        objective_family="generic",
        data_layer="predictions",
        round_scope="round_history",
        failure_modes=[
            "missing metric column",
            "metric is not numeric",
            "selected/top_k cohort columns missing",
            "no rows match the requested round/run scope",
        ],
    ),
)
def render(context, params: dict) -> None:
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style()
    from matplotlib.ticker import MaxNLocator

    metric = normalize_metric_field(get_str(params, ["metric", "metric_field", "field"], "view__selection_score"))
    if not metric:
        raise ValueError("metric_over_rounds requires a metric field.")
    cohorts = _list_param(params.get("cohort", params.get("cohorts", "selected")))
    allowed = {"selected", "top_k", "all_pool"}
    bad = sorted(set(cohorts) - allowed)
    if bad:
        raise ValueError(f"Unknown cohort(s) for metric_over_rounds: {bad}. Allowed: {sorted(allowed)}")
    summary_param = params.get("summaries", params.get("summary", ["mean"]))
    summaries = [summary.lower() for summary in _list_param(summary_param)]
    band = _band_param(params.get("band", params.get("interval", "none")))
    band_alpha = float(params.get("band_alpha", 0.18))
    if not (0.0 <= band_alpha <= 1.0):
        raise ValueError("band_alpha must be between 0 and 1.")
    if band == "iqr":
        for summary in ("q25", "q75"):
            if summary not in summaries:
                summaries.append(summary)
    top_k = int(params.get("top_k", 10))
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    need = {"as_of_round", "run_id", metric}
    row_filters = []
    if "selected" in cohorts:
        need.add("view__is_selected")
    if "top_k" in cohorts:
        need.add("view__rank_competition")
    if cohorts == ["selected"]:
        row_filters.append({"column": "view__is_selected", "op": "eq", "value": True})
    elif cohorts == ["top_k"]:
        row_filters.append({"column": "view__rank_competition", "op": "lte", "value": top_k})
    df = load_events(
        resolve_outputs_dir(context),
        need,
        round_selector=context.rounds,
        selection_view_id=context.selection_view_id,
        run_id=context.run_id,
        row_filters=row_filters,
    )
    if df.empty:
        raise ValueError("metric_over_rounds had zero rows after round/run filtering.")
    if metric not in df.columns:
        raise ValueError(f"metric_over_rounds missing metric column: {metric}")
    df[metric] = pd.to_numeric(df[metric], errors="raise")
    if not np.isfinite(df[metric].to_numpy(dtype=float)).all():
        raise ValueError(f"metric_over_rounds metric contains non-finite values: {metric}")

    rows = []
    for cohort in cohorts:
        sub = _cohort_frame(df, cohort=cohort, top_k=top_k)
        if sub.empty:
            raise ValueError(f"metric_over_rounds cohort {cohort!r} has no rows.")
        for round_index, round_df in sub.groupby("as_of_round"):
            values = round_df[metric].astype(float)
            for summary in summaries:
                rows.append(
                    {
                        "round": int(round_index),
                        "cohort": cohort,
                        "metric": metric,
                        "summary": summary,
                        "value": _summary_value(values, summary),
                    }
                )
    tidy = pd.DataFrame(rows).sort_values(["cohort", "summary", "round"]).reset_index(drop=True)
    plotted_summaries = [
        summary for summary in summaries if summary != "count" and not (band == "iqr" and summary in {"q25", "q75"})
    ]
    metric_label = plot_metric_label(params, metric)
    legend_metric_label = plot_metric_short_label(params, metric)

    figsize = tuple(params.get("figsize_in", DEFAULT_SQUARE_FIGSIZE))
    fig, ax = plt.subplots(figsize=figsize)
    apply_notebook_axes_style(ax, square=True)
    highlight_round = resolve_highlight_round(
        params.get("highlight_round", params.get("overlay_round")),
        tidy["round"].unique().tolist(),
    )
    if band == "iqr":
        cohort_order = {cohort: index for index, cohort in enumerate(sorted(tidy["cohort"].unique()))}
        for cohort, cohort_tidy in tidy.groupby("cohort"):
            low = cohort_tidy[cohort_tidy["summary"] == "q25"].sort_values("round")
            high = cohort_tidy[cohort_tidy["summary"] == "q75"].sort_values("round")
            if low.empty or high.empty:
                continue
            band_df = low[["round", "value"]].merge(
                high[["round", "value"]],
                on="round",
                suffixes=("_low", "_high"),
            )
            if band_df.empty:
                continue
            ax.fill_between(
                band_df["round"].to_numpy(dtype=float),
                band_df["value_low"].to_numpy(dtype=float),
                band_df["value_high"].to_numpy(dtype=float),
                alpha=band_alpha,
                color=categorical_color(cohort_order[str(cohort)]),
                linewidth=0,
                label=_band_legend_label(
                    cohort=str(cohort),
                    metric_label=legend_metric_label,
                    cohort_count=len(cohort_order),
                ),
                zorder=1,
            )
    line_index = 0
    for (cohort, summary), sub in tidy.groupby(["cohort", "summary"]):
        if summary == "count" and len(set(summaries)) > 1:
            continue
        if band == "iqr" and summary in {"q25", "q75"}:
            continue
        style = categorical_style(line_index)
        label = _line_legend_label(
            cohort=str(cohort),
            summary=str(summary),
            metric_label=legend_metric_label,
            cohort_count=len(set(tidy["cohort"])),
            summary_count=len(plotted_summaries),
        )
        line = ax.plot(
            sub["round"],
            sub["value"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=6,
            label=label,
            zorder=3,
        )[0]
        line_index += 1
        if highlight_round is not None:
            hi = sub[sub["round"] == int(highlight_round)]
            if not hi.empty:
                ax.scatter(
                    hi["round"],
                    hi["value"],
                    s=84,
                    marker="o",
                    facecolors="none",
                    edgecolors=line.get_color(),
                    linewidths=1.8,
                    zorder=5,
                )
    threshold = params.get("threshold", params.get("reference_line"))
    if threshold is not None:
        ax.axhline(float(threshold), color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    y_axis = params.get("y_axis") if isinstance(params.get("y_axis"), dict) else {}
    apply_y_axis_scale(
        ax,
        limits=params.get("y_limits", y_axis.get("limits")),
        reference_lines=params.get("y_reference_lines", y_axis.get("reference_lines")),
        include_zero_tick=bool(params.get("include_zero_tick", y_axis.get("include_zero_tick", False))),
    )
    ax.set_xlabel("Round")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(metric_label)
    title = pretty_title(params.get("title", f"{metric_label} over rounds"))
    count_text = _cohort_count_text(tidy)
    if count_text:
        title = f"{title}\n{count_text}"
    ax.set_title(title)
    if not legend_below_single_row(fig, ax):
        fig.tight_layout(pad=0.35)
    out = context.output_dir / context.filename
    save_notebook_square_figure(fig, out, dpi=context.dpi, tight=False)
    plt.close(fig)

    if context.save_data:
        context.save_df(tidy)


def _list_param(value: object) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _band_param(value: object) -> str:
    band = str(value or "none").strip().lower()
    if band in {"", "none", "false", "off"}:
        return "none"
    if band in {"iqr", "quartile", "quartiles"}:
        return "iqr"
    raise ValueError("band must be one of: none, iqr.")


def _cohort_frame(df, *, cohort: str, top_k: int):
    if cohort == "all_pool":
        return df.copy()
    if cohort == "selected":
        if "view__is_selected" not in df.columns:
            raise ValueError("selected cohort requires view__is_selected.")
        return df[selected_mask(df["view__is_selected"])].copy()
    if cohort == "top_k":
        if "view__rank_competition" not in df.columns:
            raise ValueError("top_k cohort requires view__rank_competition.")
        return df[positive_ranks(df["view__rank_competition"]) <= int(top_k)].copy()
    raise ValueError(f"Unknown cohort: {cohort}")


def _summary_value(values, summary: str) -> float:
    summary = str(summary).lower()
    if summary == "mean":
        return float(values.mean())
    if summary == "median":
        return float(values.median())
    if summary == "count":
        return float(values.count())
    if summary.startswith("q"):
        return float(values.quantile(float(summary.removeprefix("q")) / 100.0))
    raise ValueError(f"Unknown summary for metric_over_rounds: {summary}")


def _cohort_count_text(tidy: pd.DataFrame) -> str:
    counts = tidy.loc[tidy["summary"] == "count", "value"]
    if counts.empty:
        return ""
    values = sorted({int(round(float(value))) for value in counts.tolist()})
    if not values:
        return ""
    cohorts = sorted(set(tidy.loc[tidy["summary"] == "count", "cohort"].astype(str)))
    cohort_text = pretty_label(cohorts[0]) if len(cohorts) == 1 else "Cohort"
    if len(values) == 1:
        return f"{cohort_text} n={values[0]} per round"
    return f"{cohort_text} n={values[0]}-{values[-1]} per round"


def _line_legend_label(
    *,
    cohort: str,
    summary: str,
    metric_label: str,
    cohort_count: int,
    summary_count: int,
) -> str:
    summary_label = pretty_label(summary)
    if cohort_count <= 1:
        if summary_count <= 1:
            return f"{summary_label} {metric_label}"
        return f"{summary_label} {metric_label}"
    return f"{pretty_label(cohort)} {summary_label} {metric_label}"


def _band_legend_label(*, cohort: str, metric_label: str, cohort_count: int) -> str:
    if cohort_count <= 1:
        return f"IQR {metric_label}"
    return f"{pretty_label(cohort)} IQR {metric_label}"
