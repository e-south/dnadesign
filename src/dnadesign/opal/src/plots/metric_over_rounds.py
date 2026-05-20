"""
--------------------------------------------------------------------------------
<dnadesign project>
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
from ._mpl_utils import ensure_mpl_config_dir
from ._param_utils import get_str, normalize_metric_field


@register_plot(
    "metric_over_rounds",
    meta=PlotMeta(
        summary="Generic scalar metric summaries over rounds by cohort.",
        params={
            "metric": "Numeric ledger field to summarize (default pred__score_selected).",
            "cohort": "Cohort or list of cohorts: selected, top_k, all_pool (default selected).",
            "top_k": "Rank cutoff for top_k cohort (default 10).",
            "summaries": "Summary or list: mean, median, count, q10, q25, q75, q90.",
            "threshold": "Optional horizontal threshold/reference line.",
        },
        requires=["as_of_round", "run_id", "pred__score_selected"],
        notes=["Reads outputs/ledger/predictions and writes tidy scalar summaries when save_data is enabled."],
        data_shape="scalar over rounds",
        tidy_schema=["round", "cohort", "metric", "summary", "value"],
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

    metric = normalize_metric_field(get_str(params, ["metric", "metric_field", "field"], "pred__score_selected"))
    if not metric:
        raise ValueError("metric_over_rounds requires a metric field.")
    cohorts = _list_param(params.get("cohort", params.get("cohorts", "selected")))
    allowed = {"selected", "top_k", "all_pool"}
    bad = sorted(set(cohorts) - allowed)
    if bad:
        raise ValueError(f"Unknown cohort(s) for metric_over_rounds: {bad}. Allowed: {sorted(allowed)}")
    summaries = _list_param(params.get("summaries", params.get("summary", ["mean", "median"])))
    top_k = int(params.get("top_k", 10))
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    need = {"as_of_round", "run_id", metric}
    if "selected" in cohorts:
        need.add("sel__is_selected")
    if "top_k" in cohorts:
        need.add("sel__rank_competition")
    df = load_events(resolve_outputs_dir(context), need, round_selector=context.rounds, run_id=context.run_id)
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

    figsize = tuple(params.get("figsize_in", (8.0, 4.8)))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for (cohort, summary), sub in tidy.groupby(["cohort", "summary"]):
        if summary == "count" and len(set(summaries)) > 1:
            continue
        label = f"{cohort}:{summary}"
        ax.plot(sub["round"], sub["value"], marker="o", linewidth=1.8, label=label)
    threshold = params.get("threshold", params.get("reference_line"))
    if threshold is not None:
        ax.axhline(float(threshold), color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Round")
    ax.set_ylabel(metric)
    ax.set_title(str(params.get("title", "Metric over rounds")))
    ax.legend(frameon=False, fontsize=9)
    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)

    if context.save_data:
        context.save_df(tidy)


def _list_param(value: object) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _cohort_frame(df, *, cohort: str, top_k: int):
    if cohort == "all_pool":
        return df.copy()
    if cohort == "selected":
        if "sel__is_selected" not in df.columns:
            raise ValueError("selected cohort requires sel__is_selected.")
        return df[selected_mask(df["sel__is_selected"])].copy()
    if cohort == "top_k":
        if "sel__rank_competition" not in df.columns:
            raise ValueError("top_k cohort requires sel__rank_competition.")
        return df[positive_ranks(df["sel__rank_competition"]) <= int(top_k)].copy()
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
