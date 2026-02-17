"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/percent_high_activity_over_rounds.py

Plots percent high-activity designs over rounds from ledger predictions.
Consumes outputs/ledger/predictions for per-round summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..registries.plots import PlotMeta, register_plot
from ._events_util import load_events, resolve_outputs_dir
from ._mpl_utils import (
    annotate_plot_meta,
    ensure_mpl_config_dir,
    scale_to_sizes,
    swarm_smart,
)
from ._param_utils import event_columns_for, get_str, normalize_metric_field


@register_plot(
    "percent_high_activity_over_rounds",
    meta=PlotMeta(
        summary="Percent of candidates above a score threshold across rounds.",
        params={
            "threshold": "Scalar cutoff for 'high' (default 0.8).",
            "mode": "line|violin|both (default both).",
            "hue_field": "Optional obj__/pred__/sel__ field for swarm color.",
            "size_by": "Optional obj__/pred__/sel__ field for swarm size.",
        },
        requires=["as_of_round", "pred__score_selected"],
        notes=["Reads outputs/ledger/predictions."],
    ),
)
def render(context, params: dict) -> None:
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    import numpy as np

    threshold = float(params.get("threshold", 0.8))
    mode = str(params.get("mode", "both")).lower()  # "line" | "violin" | "both"
    if mode not in {"line", "violin", "both"}:
        raise ValueError("mode must be 'line', 'violin', or 'both'.")
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
    # Optional hue/size (applied to swarm points only)
    hue_field = normalize_metric_field(get_str(params, ["hue_field", "hue", "color", "color_by", "colour_by"], None))
    cmap = get_str(params, ["cmap"], "viridis")
    size_by = normalize_metric_field(get_str(params, ["size_by", "size", "size_field", "point_size_by"], None))
    # Assert: requested keys must resolve
    if any(k in params for k in ("hue_field", "hue", "color", "color_by", "colour_by")) and not hue_field:
        raise ValueError(
            "A hue parameter was provided but could not be parsed. Use an obj__/pred__/sel__ column or alias."
        )
    if any(k in params for k in ("size_by", "size", "size_field", "point_size_by")) and not size_by:
        raise ValueError(
            "A size_by parameter was provided but could not be parsed. Use an obj__/pred__/sel__ column or alias."
        )

    size_min = float(params.get("size_min", 10.0))
    size_max = float(params.get("size_max", 60.0))
    outputs_dir = resolve_outputs_dir(context)
    # Always read from typed sinks (predictions + runs).
    need = {"as_of_round", "pred__score_selected"}
    need |= event_columns_for(hue_field, size_by)
    df = load_events(outputs_dir, need, round_selector=context.rounds, run_id=context.run_id)
    if df.empty:
        raise ValueError("Ledger predictions contained zero rows after projection.")

    rsel = context.rounds
    if rsel in ("unspecified", "latest"):
        latest = int(df["as_of_round"].max())
        df = df[df["as_of_round"] == latest]
    elif rsel != "all":
        df = df[df["as_of_round"].isin(rsel if isinstance(rsel, list) else [rsel])]
    if df.empty:
        raise ValueError("No rows matched the requested round selector.")

    grp = df.groupby("as_of_round").agg(
        total=("pred__score_selected", "size"),
        high=("pred__score_selected", lambda s: (s >= threshold).sum()),
    )
    grp["percent_high"] = (grp["high"] / grp["total"]) * 100.0
    grp = grp.reset_index().sort_values("as_of_round")

    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    figsize = tuple(params.get("figsize_in", (7.5, 4.8)))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    rounds = grp["as_of_round"].tolist()
    if violin_width is None:
        violin_width = 0.45 if len(rounds) <= 1 else 0.9
    violin_width = float(violin_width)
    # Build per-round arrays (objective scalar always; optional hue/size aligned by index)
    series = []
    hues = [] if hue_field else None
    sizes = [] if size_by else None
    for r in rounds:
        sub = df.loc[df["as_of_round"] == r]
        y_all = sub["pred__score_selected"].astype(float).to_numpy()
        mask = np.isfinite(y_all)
        y = y_all[mask]
        series.append(y)
        if hue_field:
            if hue_field not in sub.columns:
                raise ValueError(f"hue field '{hue_field}' not present for round {r}")
            hv = sub[hue_field].astype(float).to_numpy()
            hues.append(hv[mask])
        if size_by:
            if size_by not in sub.columns:
                raise ValueError(f"size_by field '{size_by}' not present for round {r}")
            sv_all = sub[size_by].astype(float).to_numpy()
            sv = scale_to_sizes(sv_all, s_min=size_min, s_max=size_max)
            sizes.append(sv[mask])

    # Violin + optional swarm on primary y-axis (0..1)
    if mode in {"violin", "both"}:
        # Assert: per-round series must be finite, have ≥3 points, and non-zero variance
        for rr, yy in zip(rounds, series):
            if yy.size < 3:
                raise ValueError(f"Cannot draw violin: round {rr} has <3 finite points.")
            if float(np.nanmax(yy)) <= float(np.nanmin(yy)):
                raise ValueError(
                    f"Cannot draw violin: round {rr} has zero variance in 'pred__score_selected' after filtering."
                )
        parts = ax.violinplot(
            series,
            positions=rounds,
            widths=violin_width,
            showmeans=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(violin_alpha)
        parts["cmeans"].set_alpha(min(1.0, violin_alpha + 0.2))
        if swarm:
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
        ax.set_ylabel("Objective scalar")
    ax.set_xlabel("Round")
    ax.set_title("Objective Scalar Over Rounds")
    ax.set_xticks(rounds)

    # Percent-high line on a twin axis when asked
    if mode in {"line", "both"}:
        ax2 = ax if mode == "line" else ax.twinx()
        ax2.plot(
            grp["as_of_round"],
            grp["percent_high"],
            marker="o",
            linewidth=2.0,
            label=f"% ≥ {threshold:.2f}",
        )
        ax2.set_ylabel("% high (≥ threshold)")
        ax2.set_ylim(0, 100)
        if mode == "both":
            ax2.legend(frameon=False, loc="upper right")

    # Log + annotate
    total_points = int(sum(len(s) for s in series))
    raster = False if rasterize_at is None else total_points >= rasterize_at
    context.logger.info(
        "params percent_high_activity: mode=%s threshold=%.3f rounds=%s swarm=%s swarm_max=%d points=%d",
        mode,
        threshold,
        rounds,
        swarm,
        swarm_max_points,
        total_points,
    )
    annotate_plot_meta(
        ax,
        hue=hue_field,
        size_by=size_by,
        alpha=violin_alpha if mode in {"violin", "both"} else None,
        rasterized=raster,
        extras={"mode": mode, "threshold": f"{threshold:.2f}"},
    )

    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)
    if context.save_data:
        context.save_df(grp)
