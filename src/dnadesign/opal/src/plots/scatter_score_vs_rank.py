"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/scatter_score_vs_rank.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt

from ..registries.plot import register_plot
from ._events_util import load_events_with_setpoint, resolve_outputs_dir
from ._mpl_utils import annotate_plot_meta, scale_to_sizes, scatter_smart
from ._param_utils import (
    event_columns_for,
    get_float,
    get_str,
    normalize_metric_field,
)


@register_plot("scatter_score_vs_rank")
def render(context, params: dict) -> None:
    outputs_dir = resolve_outputs_dir(context)

    score_field = get_str(params, ["score_field"], "pred__y_obj_scalar")
    score_field = normalize_metric_field(score_field) or "pred__y_obj_scalar"
    rank_mode = (get_str(params, ["rank_mode"], "sequential") or "sequential").lower()
    # "sequential" | "competition"
    alpha = get_float(params, ["alpha"], 0.45)
    hue_field = normalize_metric_field(get_str(params, ["hue_field", "hue", "color", "color_by", "colour_by"], None))
    cmap = get_str(params, ["cmap"], "viridis")
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

    # Pull from predictions (full schema) and always join setpoint
    need = {
        "as_of_round",
        "run_id",
        "id",
        "sel__rank_competition",
        "sel__is_selected",
        score_field,
    }
    # Ensure optional hue/size columns are loaded if they refer to ledger columns
    need |= event_columns_for(hue_field, size_by)
    df = load_events_with_setpoint(outputs_dir, need, round_selector=context.rounds)
    if df.empty:
        raise ValueError("ledger.predictions had zero rows for requested columns.")

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
        if "sel__rank_competition" not in df.columns:
            raise ValueError("sel__rank_competition not present for competition ranking.")
        x_field = "sel__rank_competition"

    # Hue/size arrays
    hue_vals = None
    if hue_field:
        if hue_field not in df.columns:
            raise ValueError(f"hue field '{hue_field}' not present in predictions.")
        hue_vals = df[hue_field].to_numpy(dtype=float)

    if "sel__rank_competition" not in df.columns:
        df = df.sort_values(["as_of_round", score_field], ascending=[True, False]).assign(
            sel__rank_competition=lambda x: x.groupby("as_of_round").cumcount() + 1
        )

    rounds: List[int] = sorted(df["as_of_round"].unique().tolist())
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    if len(rounds) == 1:
        r = rounds[0]
        sub = df[df["as_of_round"] == r].sort_values(x_field, ascending=True)

        figsize = tuple(params.get("figsize_in", (8.5, 5.0)))
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
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
        ax.plot(x, y, linewidth=1.2, alpha=min(0.9, alpha + 0.2))
        color_kw = {}
        if hue_vals is not None:
            color_kw = {"c": sub[hue_field].to_numpy(dtype=float), "cmap": cmap}
        rasterized = x.size >= rasterize_at
        scatter_smart(
            ax,
            x,
            y,
            s=sizes,
            alpha=alpha,
            **color_kw,
            rasterize_at=rasterize_at,
        )
        if "sel__is_selected" in sub.columns:
            sel_mask = sub["sel__is_selected"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            if sel_mask.any():
                scatter_smart(
                    ax,
                    x[sel_mask],
                    y[sel_mask],
                    s=max(s_min, 1.4 * s_min),
                    alpha=min(1.0, alpha + 0.25),
                    edgecolors="black",
                    rasterize_at=rasterize_at,
                )
        ax.set_xlabel(f"Rank ({'sequential' if rank_mode == 'sequential' else 'competition'})")
        ax.set_ylabel("Objective score")
        ax.set_title(f"Score vs Rank — round {r}")
        ax.set_xlim(left=sub[x_field].max(), right=1)
        # On-plot meta + log
        context.logger.info(
            "params score_vs_rank: round=%s rank_mode=%s hue=%s size_by=%s alpha=%.2f rasterize_at=%d points=%d",
            r,
            rank_mode,
            hue_field or "—",
            size_by or "—",
            alpha,
            rasterize_at,
            int(x.size),
        )
        annotate_plot_meta(
            ax,
            hue=hue_field,
            size_by=size_by,
            alpha=alpha,
            rasterized=rasterized,
            extras={"rank": rank_mode},
        )
    else:
        default_multi = (9.5, 6.0)
        figsize = tuple(params.get("figsize_in", default_multi))
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for r, sub in df.groupby("as_of_round"):
            sub = sub.sort_values(x_field)
            ax.plot(
                sub[x_field],
                sub[score_field],
                label=f"r{r}",
                linewidth=1.5,
                alpha=0.8,
            )
            color_kw = {}
            if hue_field and hue_field in sub.columns:
                color_kw = {"c": sub[hue_field], "cmap": cmap}
            scatter_smart(
                ax,
                sub[x_field],
                sub[score_field],
                s=(
                    scale_to_sizes(sub[size_by], s_min=s_min, s_max=s_max)
                    if size_by and size_by in sub.columns
                    else s_min
                ),
                alpha=alpha,
                **color_kw,
            )
        ax.legend(title="round", frameon=False)
        ax.set_xlabel(f"Rank ({'sequential' if rank_mode == 'sequential' else 'competition'})")
        ax.set_ylabel("Objective score")
        ax.set_title("Score vs Rank by Round")
        ax.set_xlim(left=df[x_field].max(), right=1)
        context.logger.info(
            "params score_vs_rank multi-rounds: rounds=%s rank_mode=%s hue=%s size_by=%s alpha=%.2f rasterize_at=%d",
            rounds,
            rank_mode,
            hue_field or "—",
            size_by or "—",
            alpha,
            rasterize_at,
        )
        annotate_plot_meta(
            ax,
            hue=hue_field,
            size_by=size_by,
            alpha=alpha,
            rasterized=(len(df) >= rasterize_at),
            extras={"rank": rank_mode, "rounds": f"{len(rounds)}"},
        )

    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)

    if context.save_data:
        keep = [
            "as_of_round",
            "id",
            x_field,
            "sel__is_selected",
            score_field,
        ]
        context.save_df(df[keep])
