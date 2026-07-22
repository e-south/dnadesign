"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/percent_high_activity_over_rounds.py

Plots percent high-activity designs over rounds from ledger predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..registries.plots import PlotMeta, register_plot
from ._events_util import load_events, resolve_outputs_dir
from ._mpl_utils import (
    DEFAULT_SQUARE_FIGSIZE,
    annotate_plot_meta,
    apply_notebook_axes_style,
    apply_plot_style,
    categorical_color,
    categorical_style,
    ensure_mpl_config_dir,
    legend_below_single_row,
    plot_metric_label,
    pretty_title,
    save_notebook_square_figure,
    scale_to_sizes,
    swarm_smart,
)
from ._param_utils import event_columns_for, get_str, normalize_metric_field
from ._round_overlay import resolve_highlight_round


@register_plot(
    "percent_high_activity_over_rounds",
    meta=PlotMeta(
        summary="Percent of candidates above a configured metric threshold across rounds.",
        params={
            "metric": "Numeric ledger field to threshold (default view__selection_score).",
            "threshold": "Scalar cutoff for 'high' (default 0.8).",
            "threshold_quantile": "Optional quantile-derived cutoff in [0, 1]; mutually exclusive with threshold.",
            "mode": "line|violin|both (default both).",
            "hue_field": "Optional obj__/pred__/sel__ field for swarm color.",
            "size_by": "Optional obj__/pred__/sel__ field for swarm size.",
            "highlight_round": "Optional round overlay marker: latest, true, false, or an integer.",
            "show_meta": "Draw small diagnostic text inside the axes (default false).",
            "percent_ylim": "Percent axis range: auto or full (default auto).",
        },
        requires=["as_of_round", "view__selection_score"],
        notes=["Reads outputs/ledger/predictions."],
        data_shape="thresholded scalar over rounds",
        tidy_schema=["as_of_round", "total", "high", "percent_high"],
        objective_family="generic",
        data_layer="predictions_selection",
        round_scope="round_history",
        failure_modes=[
            "missing configured metric column",
            "no rows match the requested round/run scope",
            "non-finite metric values",
            "insufficient finite points for violin mode",
        ],
    ),
)
def render(context, params: dict) -> None:
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style()
    threshold_param = params.get("threshold")
    threshold_quantile_param = params.get("threshold_quantile", params.get("quantile"))
    if threshold_param is not None and threshold_quantile_param is not None:
        raise ValueError("Use either threshold or threshold_quantile, not both.")
    threshold_quantile = None if threshold_quantile_param is None else float(threshold_quantile_param)
    if threshold_quantile is not None and not (0.0 <= threshold_quantile <= 1.0):
        raise ValueError("threshold_quantile must be between 0 and 1.")
    mode = str(params.get("mode", "both")).lower()  # "line" | "violin" | "both"
    if mode not in {"line", "violin", "both"}:
        raise ValueError("mode must be 'line', 'violin', or 'both'.")
    draw_distribution = mode in {"violin", "both"}
    violin_alpha = float(params.get("violin_alpha", 0.5))
    violin_width = params.get("violin_width", None)
    swarm = bool(params.get("swarm", True))
    swarm_max_points = int(params.get("swarm_max_points", 3000))
    swarm_jitter = float(params.get("swarm_jitter", 0.08))
    swarm_alpha = float(params.get("swarm_alpha", 0.25))
    swarm_size = float(params.get("swarm_size", 9.0))
    rasterize_at = params.get("rasterize_at", None)
    if rasterize_at is not None:
        rasterize_at = int(rasterize_at)
    show_meta = bool(params.get("show_meta", False))
    percent_ylim = str(params.get("percent_ylim", "auto")).strip().lower()
    if percent_ylim not in {"auto", "full"}:
        raise ValueError("percent_ylim must be 'auto' or 'full'.")
    metric_field = normalize_metric_field(
        get_str(params, ["metric", "score_field", "metric_field", "field"], "view__selection_score")
    )
    if not metric_field:
        raise ValueError("percent_high_activity_over_rounds requires a metric field.")
    metric_label = plot_metric_label(params, metric_field)
    # Optional hue/size (applied to swarm points only)
    hue_field = normalize_metric_field(get_str(params, ["hue_field", "hue", "color", "color_by", "colour_by"], None))
    cmap = get_str(params, ["cmap"], "viridis")
    size_by = normalize_metric_field(get_str(params, ["size_by", "size", "size_field", "point_size_by"], None))
    draw_swarm = bool(draw_distribution and swarm)
    # Assert: requested keys must resolve
    if any(k in params for k in ("hue_field", "hue", "color", "color_by", "colour_by")) and not hue_field:
        raise ValueError(
            "A hue parameter was provided but could not be parsed. Use an obj__/pred__/sel__ column or alias."
        )
    if any(k in params for k in ("size_by", "size", "size_field", "point_size_by")) and not size_by:
        raise ValueError(
            "A size_by parameter was provided but could not be parsed. Use an obj__/pred__/sel__ column or alias."
        )
    if not draw_swarm and hue_field:
        raise ValueError("hue/color parameters require a swarm point layer; use mode='both' or remove hue.")
    if not draw_swarm and size_by:
        raise ValueError("size_by parameters require a swarm point layer; use mode='both' or remove size_by.")

    size_min = float(params.get("size_min", 10.0))
    size_max = float(params.get("size_max", 60.0))
    outputs_dir = resolve_outputs_dir(context)
    # Always read from typed sinks (predictions + runs).
    need = {"as_of_round", metric_field}
    if draw_swarm:
        need |= event_columns_for(hue_field, size_by)
    df = load_events(
        outputs_dir,
        need,
        round_selector=context.rounds,
        selection_view_id=context.selection_view_id,
        run_id=context.run_id,
    )
    if df.empty:
        raise ValueError("Ledger predictions contained zero rows after projection.")
    if metric_field not in df.columns:
        raise ValueError(f"metric field '{metric_field}' not present in predictions.")

    rsel = context.rounds
    if rsel in ("unspecified", "latest"):
        latest = int(df["as_of_round"].max())
        df = df[df["as_of_round"] == latest]
    elif rsel != "all":
        df = df[df["as_of_round"].isin(rsel if isinstance(rsel, list) else [rsel])]
    if df.empty:
        raise ValueError("No rows matched the requested round selector.")
    metric_values = df[metric_field].astype(float).to_numpy()
    if not np.isfinite(metric_values).all():
        raise ValueError(f"metric field '{metric_field}' contains non-finite values.")
    if threshold_quantile is not None:
        threshold = float(np.quantile(metric_values, threshold_quantile))
        quantile_pct = 100.0 * threshold_quantile
        threshold_label = _threshold_quantile_label(metric_field, quantile_pct, threshold, metric_label=metric_label)
    else:
        threshold = float(0.8 if threshold_param is None else threshold_param)
        threshold_label = f"{threshold:.2f}"

    grp = df.groupby("as_of_round").agg(
        total=(metric_field, "size"),
        high=(metric_field, lambda s: (s >= threshold).sum()),
    )
    grp["percent_high"] = (grp["high"] / grp["total"]) * 100.0
    grp = grp.reset_index().sort_values("as_of_round")

    figsize = tuple(params.get("figsize_in", DEFAULT_SQUARE_FIGSIZE))
    fig, ax = plt.subplots(figsize=figsize)
    apply_notebook_axes_style(ax, square=True)
    rounds = grp["as_of_round"].tolist()
    highlight_round = resolve_highlight_round(
        params.get("highlight_round", params.get("overlay_round")),
        rounds,
    )
    if violin_width is None:
        violin_width = 0.45 if len(rounds) <= 1 else 0.9
    violin_width = float(violin_width)
    # Build per-round arrays only for distribution modes; line mode only needs grouped counts.
    series = []
    hues = [] if hue_field else None
    sizes = [] if size_by else None
    if draw_distribution:
        for r in rounds:
            sub = df.loc[df["as_of_round"] == r]
            y = sub[metric_field].astype(float).to_numpy()
            series.append(y)
            if hue_field:
                if hue_field not in sub.columns:
                    raise ValueError(f"hue field '{hue_field}' not present for round {r}")
                hues.append(sub[hue_field].astype(float).to_numpy())
            if size_by:
                if size_by not in sub.columns:
                    raise ValueError(f"size_by field '{size_by}' not present for round {r}")
                sv_all = sub[size_by].astype(float).to_numpy()
                sv = scale_to_sizes(sv_all, s_min=size_min, s_max=size_max)
                sizes.append(sv)

    # Violin + optional swarm on primary y-axis (0..1)
    if draw_distribution:
        # Assert: per-round series must be finite, have ≥3 points, and non-zero variance
        for rr, yy in zip(rounds, series):
            if yy.size < 3:
                raise ValueError(f"Cannot draw violin: round {rr} has <3 finite points.")
            if float(np.nanmax(yy)) <= float(np.nanmin(yy)):
                raise ValueError(
                    f"Cannot draw violin: round {rr} has zero variance in '{metric_field}' after filtering."
                )
        parts = ax.violinplot(
            series,
            positions=rounds,
            widths=violin_width,
            showmeans=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(categorical_color(0))
            pc.set_alpha(violin_alpha)
            pc.set_edgecolor("#444444")
            pc.set_linewidth(0.7)
        parts["cmeans"].set_alpha(min(1.0, violin_alpha + 0.2))
        if draw_swarm:
            swarm_smart(
                ax,
                rounds,
                series,
                jitter=swarm_jitter,
                max_points_per_group=swarm_max_points,
                s=swarm_size,
                sizes_by_group=sizes,
                hue_by_group=hues,
                cmap=cmap,
                alpha=swarm_alpha,
                rasterize_at=rasterize_at,
            )
        ax.set_ylabel(metric_label)
    ax.set_xlabel("Round")
    ax.set_title(pretty_title(params.get("title", f"High-activity rate for {metric_label}")))
    ax.set_xticks(rounds)

    # Percent-high line on a twin axis when asked
    if mode in {"line", "both"}:
        ax2 = ax if mode == "line" else ax.twinx()
        if mode == "both":
            apply_notebook_axes_style(ax2, grid=False)
        style = categorical_style(1 if mode == "both" else 0)
        ax2.plot(
            grp["as_of_round"],
            grp["percent_high"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=6,
            label=f"At/above {threshold_label}",
        )
        ax2.set_ylabel("Candidates at/above threshold (%)")
        if percent_ylim == "full":
            ax2.set_ylim(0, 100)
        else:
            upper = _auto_percent_upper(float(grp["percent_high"].max()))
            ax2.set_ylim(0, upper)
        if highlight_round is not None:
            hi = grp.loc[grp["as_of_round"] == int(highlight_round)]
            if not hi.empty:
                ax2.scatter(
                    hi["as_of_round"],
                    hi["percent_high"],
                    s=84,
                    facecolors="none",
                    edgecolors="#202020",
                    linewidths=1.8,
                    zorder=5,
                )

    # Log + annotate
    total_points = int(sum(len(s) for s in series)) if draw_distribution else int(len(df))
    raster = bool(draw_distribution and rasterize_at is not None and total_points >= rasterize_at)
    context.logger.info(
        "params percent_high_activity: mode=%s threshold=%.3f rounds=%s swarm=%s swarm_max=%d points=%d",
        mode,
        threshold,
        rounds,
        draw_swarm,
        swarm_max_points,
        total_points,
    )
    if show_meta:
        annotate_plot_meta(
            ax,
            hue=hue_field,
            size_by=size_by,
            alpha=violin_alpha if mode in {"violin", "both"} else None,
            rasterized=raster,
            extras={"metric": metric_field, "mode": mode, "threshold": threshold_label},
        )

    legend_ax = ax2 if mode in {"line", "both"} else ax
    if not legend_below_single_row(fig, legend_ax):
        fig.tight_layout(pad=0.35)
    out = context.output_dir / context.filename
    save_notebook_square_figure(fig, out, dpi=context.dpi, tight=False)
    plt.close(fig)
    if context.save_data:
        grp["threshold"] = threshold
        grp["threshold_label"] = threshold_label
        if threshold_quantile is not None:
            grp["threshold_quantile"] = threshold_quantile
        context.save_df(grp)


def _auto_percent_upper(max_percent: float) -> float:
    import math

    if max_percent <= 0:
        return 5.0
    return min(100.0, max(5.0, math.ceil((max_percent * 1.15) / 5.0) * 5.0))


def _threshold_quantile_label(metric_field: str, quantile_pct: float, threshold: float, *, metric_label: str) -> str:
    if metric_field == "view__selection_score":
        return f"P{quantile_pct:g} of {metric_label} ({threshold:.3g})"
    return f"P{quantile_pct:g} {metric_label} cutoff ({threshold:.3g})"
