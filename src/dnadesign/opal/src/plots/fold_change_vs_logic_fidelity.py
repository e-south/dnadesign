"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/fold_change_vs_logic_fidelity.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..registries.plot import register_plot
from ._events_util import load_events_with_setpoint, resolve_events_path
from ._mpl_utils import annotate_plot_meta, scale_to_sizes, scatter_smart
from ._param_utils import (
    event_columns_for,
    get_float,
    get_str,
    normalize_metric_field,
)


@register_plot("fold_change_vs_logic_fidelity")
def render(context, params: dict) -> None:
    events_path = resolve_events_path(context)

    delta = get_float(params, ["intensity_log2_offset_delta"], 0.0)
    alpha = get_float(params, ["alpha"], 0.40)
    hue_field = normalize_metric_field(
        get_str(params, ["hue_field", "hue", "color", "color_by", "colour_by"], None)
    )
    cmap = get_str(params, ["cmap"], "viridis")
    cbar = bool(params.get("cbar", True))
    # Accept multiple synonyms for size
    size_by = normalize_metric_field(
        get_str(params, ["size_by", "size", "size_field", "point_size_by"], None)
    )
    # Assert: requested keys must resolve
    if (
        any(k in params for k in ("hue_field", "hue", "color", "color_by", "colour_by"))
        and not hue_field
    ):
        raise ValueError(
            "A hue parameter was provided but could not be parsed. "
            "Use an obj__/pred__/sel__ column or alias (e.g., 'logic_fidelity', 'fold_change', 'score')."
        )
    if (
        any(k in params for k in ("size_by", "size", "size_field", "point_size_by"))
        and not size_by
    ):
        raise ValueError(
            "A size_by parameter was provided but could not be parsed. "
            "Use an obj__/pred__/sel__ column or alias (e.g., 'logic_fidelity', 'fold_change')."
        )
    s_min = get_float(params, ["size_min"], 10.0)
    s_max = get_float(params, ["size_max"], 60.0)
    size_clip = params.get("size_clip")  # [lo, hi]
    # None (default) = do not rasterize; set a positive integer in YAML to enable.
    rasterize_at = params.get("rasterize_at", None)
    if rasterize_at is not None:
        rasterize_at = int(rasterize_at)

    # Ensure setpoint is available (backfill from run_meta if needed)
    need = {
        "as_of_round",
        "run_id",
        "pred__y_hat_model",
        "sel__is_selected",
        "obj__diag__setpoint",
        "pred__y_obj_scalar",
    }
    # If hue/size ask for objective/pred/sel columns, load them from predictions
    need |= event_columns_for(hue_field, size_by)
    df = load_events_with_setpoint(events_path, need, round_selector=context.rounds)

    # Round selection: single round (default latest)
    rsel = context.rounds
    if rsel in ("unspecified", "latest"):
        latest = int(df["as_of_round"].max())
        df = df[df["as_of_round"] == latest]
    elif rsel != "all":
        lst = rsel if isinstance(rsel, list) else [rsel]
        if len(lst) != 1:
            raise ValueError(
                "Select exactly one round or use --round latest for this plot."
            )
        df = df[df["as_of_round"].isin(lst)]
    if df.empty:
        raise ValueError("No rows matched the requested round selector.")

    # Split logic (0:4) and intensity (4:8)
    def _split(a):
        v = np.asarray(a, dtype=float).ravel()
        if v.size < 8:
            return np.full(4, np.nan), np.full(4, np.nan)
        return v[0:4], v[4:8]

    logic_list, star_list = zip(*df["pred__y_hat_model"].map(_split).tolist())
    logic = np.vstack(logic_list)
    ystar = np.vstack(star_list)
    ylin = np.maximum(0.0, np.power(2.0, ystar) - delta)
    fold_change = np.max(ylin, axis=1) - np.min(ylin, axis=1)

    # Logic fidelity vs setpoint
    sp = df["obj__diag__setpoint"].dropna()
    if sp.empty:
        raise ValueError("Need obj__diag__setpoint to compute logic fidelity.")
    setpoint = np.asarray(sp.iloc[0], dtype=float).ravel()
    D = np.sqrt(np.sum(np.maximum(setpoint**2, (1.0 - setpoint) ** 2)))
    dist = np.linalg.norm(logic - setpoint[None, :], axis=1)
    lf = np.clip(1.0 - (dist / (D if D > 0 else 1.0)), 0.0, 1.0)

    # Simple in-house styling (no seaborn dependency)
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    figsize = tuple(params.get("figsize_in", (7.8, 5.2)))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Point sizes (optional) — can use derived series
    if size_by is None:
        sizes = float(s_min)
    else:
        if size_by in df.columns:
            base = df[size_by].to_numpy(dtype=float)
        elif size_by == "logic_fidelity":
            base = lf
        elif size_by == "fold_change":
            base = fold_change
        else:
            raise ValueError(f"Unknown size_by field: {size_by!r}")
        sizes = scale_to_sizes(base, s_min=s_min, s_max=s_max, clip=size_clip)

    rasterized = lf.size >= rasterize_at
    # Optional hue/coloring
    color_kw = {}
    if hue_field:
        if hue_field in df.columns:
            color_kw = {"c": df[hue_field].to_numpy(dtype=float), "cmap": cmap}
        elif hue_field == "logic_fidelity":
            color_kw = {"c": lf, "cmap": cmap}
        elif hue_field == "fold_change":
            color_kw = {"c": fold_change, "cmap": cmap}
        else:
            raise ValueError(f"Unknown hue field: {hue_field!r}")

    sc = scatter_smart(
        ax,
        lf,
        fold_change,
        s=sizes,
        alpha=alpha,
        **color_kw,
        rasterize_at=rasterize_at,
    )
    if hue_field and cbar:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(
            hue_field.replace("obj__", "").replace("pred__", "").replace("sel__", "")
        )
    if "sel__is_selected" in df.columns:
        sel_mask = df["sel__is_selected"].fillna(False).astype(bool)
        if sel_mask.any():
            idx = df.index[sel_mask]
            scatter_smart(
                ax,
                lf[idx],
                fold_change[idx],
                s=max(s_min, 1.4 * s_min),
                alpha=min(1.0, alpha + 0.25),
                edgecolors="black",
                rasterize_at=rasterize_at,
            )

    ax.set_xlabel("Logic fidelity (0-1)")
    ax.set_ylabel("Fold change (max-min) in linear intensity")
    ax.set_title("Trade-off: Fold Change vs Logic Fidelity")

    # Log + annotate
    from ._mpl_utils import log_kv

    log_kv(
        context.logger,
        "fold_change_vs_logic",
        round_sel=context.rounds,
        hue=hue_field or "-",
        size_by=size_by or "-",
        alpha=float(alpha),
        rasterize_at=(rasterize_at if rasterize_at is not None else "off"),
        points=int(lf.size),
    )
    annotate_plot_meta(
        ax,
        hue=hue_field,
        size_by=size_by,
        alpha=alpha,
        rasterized=rasterized,
        extras={"delta": f"{delta:.3g}"},
    )

    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)

    if context.save_data:
        tidy = pd.DataFrame({"logic_fidelity": lf, "fold_change": fold_change})
        context.save_df(tidy)
