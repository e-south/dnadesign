"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/scatter_score_vs_rank.py

Plots objective score vs selection rank from ledger predictions. Reads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from ..registries.plots import PlotMeta, register_plot
from ._events_util import load_events, resolve_outputs_dir
from ._mpl_utils import (
    DEFAULT_SQUARE_FIGSIZE,
    add_flush_colorbar,
    annotate_plot_meta,
    apply_notebook_axes_style,
    apply_plot_style,
    apply_y_axis_scale,
    categorical_style,
    ensure_mpl_config_dir,
    legend_below_single_row,
    plot_metric_label,
    pretty_label,
    pretty_title,
    save_notebook_square_figure,
    scale_to_sizes,
    scatter_smart,
    sequential_colormap,
)
from ._param_utils import (
    event_columns_for,
    get_bool,
    get_float,
    get_str,
    normalize_metric_field,
)


@register_plot(
    "scatter_score_vs_rank",
    meta=PlotMeta(
        summary="Objective score by selection rank; selected candidates are outlined.",
        premise="Selection rank must preserve descending objective-score order.",
        decision_value="Confirms the operative ranking direction and exposes score compression or ties near top-K.",
        rationale="A direct score-rank curve verifies that the selector preserves the declared objective order.",
        alt_text=(
            "Scatter and line plot of selected-objective score against selection rank. The rank axis is reversed so "
            "rank one appears at the right; selected candidates are outlined."
        ),
        non_claim_boundary="A correct rank transform does not validate the objective, predictor, or biology.",
        tier="diagnostic",
        params={
            "score_field": "Ledger field for y-axis (default view__selection_score).",
            "rank_mode": "sequential|competition (default sequential).",
            "rank_scale": "linear|log (default linear). Log scale resolves the top-ranked tail in large pools.",
            "rank_label": "Optional explicit x-axis label, including favorable direction.",
            "show_selection_view": "Append the active selection-view label to the title (default false).",
            "legend_location": "below|upper_left (default below).",
            "y_axis": "Optional mapping with limits, reference_lines, and include_zero_tick.",
            "y_limits": "Optional explicit two-item y-axis limits; overrides y_axis.limits.",
            "y_reference_lines": "Optional labeled y-axis reference lines; overrides y_axis.reference_lines.",
            "include_zero_tick": "Include a zero tick when zero is inside the visible y range.",
            "hue_field": "Optional obj__/pred__/sel__ field for color.",
            "size_by": "Optional obj__/pred__/sel__ field for size.",
            "alpha": "Point alpha (default 0.45).",
            "round_cmap": "Sequential colormap for multi-round plots when hue_field is not set.",
            "show_meta": "Draw small diagnostic text inside the axes (default false).",
        },
        requires=[
            "as_of_round",
            "run_id",
            "view__selection_score",
            "view__rank_competition",
            "view__is_selected",
        ],
        notes=["Reads outputs/ledger/predictions."],
        data_shape="selection behavior scatter",
        tidy_schema=["as_of_round", "id", "view__is_selected"],
        objective_family="generic",
        data_layer="predictions_selection",
        round_scope="single_or_round_history",
        failure_modes=[
            "missing score or rank columns",
            "score/hue/size fields are nonnumeric",
            "no rows match the requested round/run scope",
            "ambiguous run_id for selected rounds",
        ],
    ),
)
def render(context, params: dict) -> None:
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt

    apply_plot_style()
    from matplotlib.ticker import MaxNLocator

    outputs_dir = resolve_outputs_dir(context)

    score_field = get_str(params, ["score_field"], "view__selection_score")
    score_field = normalize_metric_field(score_field) or "view__selection_score"
    score_label = plot_metric_label(params, score_field)
    rank_mode = (get_str(params, ["rank_mode"], "sequential") or "sequential").lower()
    rank_scale = (get_str(params, ["rank_scale"], "linear") or "linear").lower()
    rank_label = get_str(params, ["rank_label"], None)
    show_selection_view = get_bool(params, ["show_selection_view"], False)
    legend_location = (get_str(params, ["legend_location"], "below") or "below").lower()
    # "sequential" | "competition"
    alpha = get_float(params, ["alpha"], 0.45)
    hue_field = normalize_metric_field(get_str(params, ["hue_field", "hue", "color", "color_by", "colour_by"], None))
    cmap = get_str(params, ["cmap"], "viridis")
    round_cmap = get_str(params, ["round_cmap"], "round_progression")
    size_by = normalize_metric_field(get_str(params, ["size_by", "size", "size_field", "point_size_by"], None))
    # Assert: if user supplied hue/size keys but normalization yielded none → misconfiguration
    if any(k in params for k in ("hue_field", "hue", "color", "color_by", "colour_by")) and not hue_field:
        raise ValueError(
            "A hue parameter was provided but could not be parsed. "
            "Use an obj__/pred__/sel__ column or alias (e.g., 'effect_scaled', 'score')."
        )
    if any(k in params for k in ("size_by", "size", "size_field", "point_size_by")) and not size_by:
        raise ValueError(
            "A size_by parameter was provided but could not be parsed. "
            "Use an obj__/pred__/sel__ column or alias (e.g., 'logic_fidelity')."
        )
    s_min = get_float(params, ["size_min"], 10.0)
    s_max = get_float(params, ["size_max"], 60.0)
    # None (default) = do not rasterize; set a positive integer in YAML to enable.
    rasterize_at = params.get("rasterize_at", None)
    if rasterize_at is not None:
        rasterize_at = int(rasterize_at)
    rasterize_at_log = int(rasterize_at) if rasterize_at is not None else 0
    show_meta = bool(params.get("show_meta", False))
    manual_layout = False
    if rank_scale not in {"linear", "log"}:
        raise ValueError("rank_scale must be 'linear' or 'log'.")
    if legend_location not in {"below", "upper_left"}:
        raise ValueError("legend_location must be 'below' or 'upper_left'.")

    # Pull from predictions (full schema) and always join setpoint
    need = {
        "as_of_round",
        "run_id",
        "id",
        "view__rank_competition",
        "view__is_selected",
        score_field,
    }
    # Ensure optional hue/size columns are loaded if they refer to ledger columns
    need |= event_columns_for(hue_field, size_by)
    df = load_events(
        outputs_dir,
        need,
        round_selector=context.rounds,
        selection_view_id=context.selection_view_id,
        run_id=context.run_id,
    )
    if df.empty:
        raise ValueError("outputs/ledger/predictions had zero rows for requested columns.")

    rsel = context.rounds
    if rsel in ("unspecified", "latest"):
        latest = int(df["as_of_round"].max())
        df = df[df["as_of_round"] == latest]
    elif rsel == "all":
        pass
    else:
        df = df[df["as_of_round"].isin(rsel)]
    if df.empty:
        raise ValueError("No rows matched the requested round selector.")

    # Prepare ranks
    if rank_mode not in {"sequential", "competition"}:
        raise ValueError("rank_mode must be 'sequential' or 'competition'")
    if rank_mode == "sequential":
        # One contiguous 1..N per round, ordered by score (desc)
        df = df.sort_values(["as_of_round", score_field], ascending=[True, False])
        df["rank__sequential"] = df.groupby("as_of_round").cumcount() + 1
        x_field = "rank__sequential"
    else:
        if "view__rank_competition" not in df.columns:
            raise ValueError("view__rank_competition not present for competition ranking.")
        x_field = "view__rank_competition"

    # Hue/size arrays
    hue_vals = None
    if hue_field:
        if hue_field not in df.columns:
            raise ValueError(f"hue field '{hue_field}' not present in predictions.")
        hue_vals = df[hue_field].to_numpy(dtype=float)

    if "view__rank_competition" not in df.columns:
        df = df.sort_values(["as_of_round", score_field], ascending=[True, False]).assign(
            view__rank_competition=lambda x: x.groupby("as_of_round").cumcount() + 1
        )

    rounds: List[int] = sorted(df["as_of_round"].unique().tolist())
    if len(rounds) == 1:
        r = rounds[0]
        sub = df[df["as_of_round"] == r].sort_values(x_field, ascending=True)

        figsize = tuple(params.get("figsize_in", DEFAULT_SQUARE_FIGSIZE))
        fig, ax = plt.subplots(figsize=figsize)
        apply_notebook_axes_style(ax, square=True)
        x = sub[x_field].to_numpy()
        y = sub[score_field].to_numpy(dtype=float)
        # optional size mapping
        if size_by:
            if size_by not in sub.columns:
                raise ValueError(f"size/size_by field '{size_by}' not present in dataframe.")

            sizes = scale_to_sizes(sub[size_by].to_numpy(dtype=float), s_min=s_min, s_max=s_max)
        else:
            sizes = s_min
        # line for shape, then scatter for density
        style = categorical_style(0)
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            alpha=min(0.9, alpha + 0.2),
            label=f"Prediction pool · round {r}",
            zorder=2,
        )
        color_kw = {}
        if hue_vals is not None:
            color_kw = {"c": sub[hue_field].to_numpy(dtype=float), "cmap": cmap}
        rasterized = False if rasterize_at is None else x.size >= rasterize_at
        scatter_smart(
            ax,
            x,
            y,
            s=sizes,
            alpha=alpha,
            marker=style["marker"],
            **color_kw,
            rasterize_at=rasterize_at,
        )
        if "view__is_selected" in sub.columns:
            sel_mask = sub["view__is_selected"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            if sel_mask.any():
                scatter_smart(
                    ax,
                    x[sel_mask],
                    y[sel_mask],
                    s=max(s_min, 1.4 * s_min),
                    alpha=min(1.0, alpha + 0.25),
                    marker="D",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.0,
                    rasterize_at=rasterize_at,
                    label="Selected",
                )
        ax.set_xlabel(_rank_axis_label(x_field=x_field, rank_mode=rank_mode, rank_label=rank_label))
        ax.set_ylabel(score_label)
        ax.set_title(
            _score_rank_title(
                params.get("title", f"{score_label} vs rank, round {r}"),
                context=context,
                show_selection_view=show_selection_view,
            ),
            loc="left",
            fontweight="semibold",
            fontsize=10.5,
            pad=8,
            linespacing=1.35,
        )
        _set_rank_axis(ax, float(sub[x_field].max()), scale=rank_scale)
        if rank_scale == "linear":
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        # On-plot meta + log
        context.logger.info(
            "params score_vs_rank: round=%s rank_mode=%s hue=%s size_by=%s alpha=%.2f rasterize_at=%d points=%d",
            r,
            rank_mode,
            hue_field or "—",
            size_by or "—",
            alpha,
            rasterize_at_log,
            int(x.size),
        )
        if show_meta:
            annotate_plot_meta(
                ax,
                hue=hue_field,
                size_by=size_by,
                alpha=alpha,
                rasterized=rasterized,
                extras={"rank": rank_mode},
            )
    else:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        default_multi = DEFAULT_SQUARE_FIGSIZE
        figsize = tuple(params.get("figsize_in", default_multi))
        fig, ax = plt.subplots(figsize=figsize)
        apply_notebook_axes_style(ax, square=True)
        round_norm = Normalize(vmin=min(rounds), vmax=max(rounds))
        round_colors = sequential_colormap(round_cmap)
        point_alpha = float(params.get("multi_round_alpha", min(alpha, 0.28)))
        for line_index, (r, sub) in enumerate(df.groupby("as_of_round")):
            sub = sub.sort_values(x_field)
            style = categorical_style(line_index)
            round_color = round_colors(round_norm(float(r)))
            ax.plot(
                sub[x_field],
                sub[score_field],
                color=round_color,
                linestyle=style["linestyle"],
                linewidth=1.4,
                alpha=0.45,
                zorder=2,
            )
            color_kw = {}
            if hue_field and hue_field in sub.columns:
                color_kw = {"c": sub[hue_field], "cmap": cmap}
            else:
                color_kw = {"color": round_color}
            scatter_smart(
                ax,
                sub[x_field],
                sub[score_field],
                s=(
                    scale_to_sizes(sub[size_by], s_min=s_min, s_max=s_max)
                    if size_by and size_by in sub.columns
                    else s_min
                ),
                alpha=point_alpha,
                marker=style["marker"],
                rasterize_at=rasterize_at,
                **color_kw,
            )
            if "view__is_selected" in sub.columns:
                sel_mask = sub["view__is_selected"].astype("boolean").fillna(False).to_numpy(dtype=bool)
                if sel_mask.any():
                    sx = sub.loc[sel_mask, x_field].to_numpy()
                    sy = sub.loc[sel_mask, score_field].to_numpy(dtype=float)
                    scatter_smart(
                        ax,
                        sx,
                        sy,
                        s=max(s_min, 1.7 * s_min),
                        alpha=min(1.0, alpha + 0.25),
                        marker=style["marker"],
                        facecolors="none",
                        edgecolors=round_color,
                        linewidths=1.2,
                        rasterize_at=rasterize_at,
                        label="_nolegend_",
                    )
        ax.set_xlabel(_rank_axis_label(x_field=x_field, rank_mode=rank_mode, rank_label=rank_label))
        ax.set_ylabel(score_label)
        ax.set_title(
            _score_rank_title(
                params.get("title", f"{score_label} vs rank by round"),
                context=context,
                show_selection_view=show_selection_view,
            ),
            loc="left",
            fontweight="semibold",
            fontsize=10.5,
            pad=8,
            linespacing=1.35,
        )
        _set_rank_axis(ax, float(df[x_field].max()), scale=rank_scale)
        if rank_scale == "linear":
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        if hue_field is None:
            fig.subplots_adjust(left=0.14, right=0.80, bottom=0.16, top=0.86)
            mappable = ScalarMappable(norm=round_norm, cmap=round_colors)
            mappable.set_array([])
            cbar = add_flush_colorbar(fig, ax, mappable, label="Round", pad=0.04)
            cbar.set_ticks(rounds if len(rounds) <= 12 else [min(rounds), max(rounds)])
            manual_layout = True
        context.logger.info(
            "params score_vs_rank multi-rounds: rounds=%s rank_mode=%s hue=%s size_by=%s alpha=%.2f rasterize_at=%d",
            rounds,
            rank_mode,
            hue_field or "—",
            size_by or "—",
            alpha,
            rasterize_at_log,
        )
        rasterized_multi = False if rasterize_at is None else len(df) >= rasterize_at
        if show_meta:
            annotate_plot_meta(
                ax,
                hue=hue_field,
                size_by=size_by,
                alpha=alpha,
                rasterized=rasterized_multi,
                extras={"rank": rank_mode, "rounds": f"{len(rounds)}"},
            )

    y_axis_raw = params.get("y_axis")
    if y_axis_raw is not None and not isinstance(y_axis_raw, dict):
        raise ValueError("y_axis must be a mapping when provided.")
    y_axis = y_axis_raw or {}
    y_limits = params.get("y_limits", y_axis.get("limits"))
    y_reference_lines = params.get("y_reference_lines", y_axis.get("reference_lines"))
    has_y_reference_lines = y_reference_lines not in (None, "", False, [])
    apply_y_axis_scale(
        ax,
        limits=y_limits,
        reference_lines=y_reference_lines,
        include_zero_tick=bool(params.get("include_zero_tick", y_axis.get("include_zero_tick", False))),
    )
    if has_y_reference_lines and y_limits in (None, "", False):
        ax.margins(y=0.03)

    if legend_location == "upper_left":
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.0, 0.94 if has_y_reference_lines else 1.0),
                fontsize=8,
                ncol=min(2, len(handles)),
                frameon=False,
                handletextpad=0.45,
                columnspacing=0.8,
            )
        if not manual_layout:
            fig.tight_layout(pad=0.35)
    elif manual_layout:
        pass
    elif not legend_below_single_row(fig, ax):
        fig.tight_layout(pad=0.35)
    out = context.output_dir / context.filename
    save_notebook_square_figure(fig, out, dpi=context.dpi, tight=False)
    plt.close(fig)

    if context.save_data:
        keep = [
            "as_of_round",
            "id",
            x_field,
            "view__is_selected",
            score_field,
        ]
        context.save_df(df[keep])


def _set_rank_axis(ax, max_rank: float, *, scale: str) -> None:
    from matplotlib.ticker import FuncFormatter

    maximum = float(max_rank)
    if not maximum >= 1.0:
        raise ValueError("Rank axis requires max_rank >= 1.")
    if scale == "log":
        ax.set_xscale("log")
        ax.set_xlim(maximum * 1.08, 0.9)
    elif scale == "linear":
        ax.set_xlim(maximum * 1.01, 0.5)
    else:
        raise ValueError("rank_scale must be 'linear' or 'log'.")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{int(value):,}" if value >= 1 else ""))


def _score_rank_title(value: object, *, context: object, show_selection_view: bool) -> str:
    title = pretty_title(value)
    if not show_selection_view:
        return title
    selection_view_id = str(getattr(context, "selection_view_id", "") or "").strip()
    if not selection_view_id:
        raise ValueError("show_selection_view requires an active selection_view_id.")
    return f"{title}\nSelection view: {pretty_label(selection_view_id)}"


def _rank_axis_label(*, x_field: str, rank_mode: str, rank_label: str | None) -> str:
    if rank_label is not None:
        label = str(rank_label).strip()
        if not label:
            raise ValueError("rank_label must be non-empty when provided.")
        return label
    return f"{pretty_label(x_field)} ({rank_mode})"
