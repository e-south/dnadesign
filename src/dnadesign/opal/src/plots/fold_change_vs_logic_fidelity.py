"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/fold_change_vs_logic_fidelity.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..registries.plot import register_plot
from ._events_util import load_events_with_setpoint, resolve_events_path
from ._mpl_utils import annotate_plot_meta, scale_to_sizes, scatter_smart
from ._param_utils import event_columns_for, get_float, get_str, normalize_metric_field


@register_plot("fold_change_vs_logic_fidelity")
def render(context, params: dict) -> None:
    events_path = resolve_events_path(context)

    delta = get_float(params, ["intensity_log2_offset_delta"], 0.0)
    # Allow user to choose the Y axis: fold_change (default), effect_raw/scaled, or score.
    y_axis = get_str(params, ["y_axis", "y_field", "y"], "fold_change")
    alpha = get_float(params, ["alpha"], 0.40)
    hue_field = normalize_metric_field(
        get_str(params, ["hue_field", "hue", "color", "color_by", "colour_by"], None)
    )
    cmap = get_str(params, ["cmap"], "viridis")
    cbar = bool(params.get("cbar", True))
    # Selected-point styling (shape override instead of overlay)
    selected_marker = get_str(params, ["selected_marker"], "*")
    selected_size_scale = get_float(params, ["selected_size_scale"], 1.0)
    selected_alpha = min(
        1.0, get_float(params, ["selected_alpha"], float(alpha) + 0.10)
    )
    selected_edgecolor = get_str(params, ["selected_edgecolor"], "black")
    # Accept multiple synonyms for size
    size_by = normalize_metric_field(
        get_str(params, ["size_by", "size", "size_field", "point_size_by"], None)
    )
    # Axis control: by default, snap logic-fidelity x-axis to [0, 1]
    force_xlim_01 = bool(params.get("force_xlim_01", True))
    xticks_n = int(params.get("xticks_n", 6)) if force_xlim_01 else None

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
    # If hue/size/y ask for objective/pred/sel columns, load them from predictions
    # (fold_change is derived locally and won't be added here)
    need |= event_columns_for(hue_field, size_by, y_axis)
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

    # --------------------
    # Choose Y-axis series
    # --------------------
    _ya = (y_axis or "fold_change").replace(".", "__").strip().lower()
    if _ya in ("", "fold_change"):
        y_plot = fold_change
        y_field_label = "Fold change (max-min) in linear intensity"
        y_title_short = "Fold Change"
        tidy_col = "fold_change"
    elif _ya in ("effect_raw", "obj__effect_raw"):
        if "obj__effect_raw" not in df.columns:
            raise ValueError(
                "Requested y_axis=effect_raw but obj__effect_raw is missing."
            )
        y_plot = df["obj__effect_raw"].astype(float).to_numpy()
        y_field_label = "Objective effect (raw)"
        y_title_short = "Effect (raw)"
        tidy_col = "obj__effect_raw"
    elif _ya in ("effect_scaled", "obj__effect_scaled"):
        if "obj__effect_scaled" not in df.columns:
            raise ValueError(
                "Requested y_axis=effect_scaled but obj__effect_scaled is missing."
            )
        y_plot = df["obj__effect_scaled"].astype(float).to_numpy()
        y_field_label = "Objective effect (scaled)"
        y_title_short = "Effect (scaled)"
        tidy_col = "obj__effect_scaled"
    elif _ya in ("score", "pred__y_obj_scalar"):
        y_plot = df["pred__y_obj_scalar"].astype(float).to_numpy()
        y_field_label = "Objective score"
        y_title_short = "Score"
        tidy_col = "pred__y_obj_scalar"
    else:
        raise ValueError(f"Unknown y_axis: {y_axis!r}")

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

    # Rasterization status — safe when rasterize_at is None
    rasterized = (rasterize_at is not None) and (lf.size >= int(rasterize_at))

    # Optional hue/coloring (build full vector + global scale for consistent colorbar)
    hue_vals_all = None
    vmin = vmax = None
    if hue_field:
        if hue_field in df.columns:
            hue_vals_all = df[hue_field].to_numpy(dtype=float)
        elif hue_field == "logic_fidelity":
            hue_vals_all = lf
        elif hue_field == "fold_change":
            hue_vals_all = fold_change
        else:
            raise ValueError(f"Unknown hue field: {hue_field!r}")
        finite = np.isfinite(hue_vals_all)
        if finite.any():
            vmin = float(np.nanmin(hue_vals_all[finite]))
            vmax = float(np.nanmax(hue_vals_all[finite]))

    # Selection mask (if present)
    sel_mask = (
        df["sel__is_selected"].fillna(False).astype(bool).to_numpy()
        if "sel__is_selected" in df.columns
        else np.zeros(lf.shape[0], dtype=bool)
    )
    not_sel = ~sel_mask

    # Draw NON-selected first (base layer)
    base_sizes = sizes if np.isscalar(sizes) else (sizes[not_sel])
    base_color_kw = {}
    if hue_vals_all is not None:
        base_color_kw = {"c": hue_vals_all[not_sel], "cmap": cmap}
        if vmin is not None and vmax is not None:
            base_color_kw.update({"vmin": vmin, "vmax": vmax})
    scatter_smart(
        ax,
        lf[not_sel],
        y_plot[not_sel],
        s=base_sizes,
        alpha=alpha,
        rasterize_at=rasterize_at,
        **base_color_kw,
    )

    # Draw SELECTED as a different marker (override, not overlay)
    if sel_mask.any():
        sel_sizes = (
            sizes
            if np.isscalar(sizes)
            else (sizes[sel_mask] * float(selected_size_scale))
        )
        sel_color_kw = {}
        if hue_vals_all is not None:
            sel_color_kw = {"c": hue_vals_all[sel_mask], "cmap": cmap}
            if vmin is not None and vmax is not None:
                sel_color_kw.update({"vmin": vmin, "vmax": vmax})
        scatter_smart(
            ax,
            lf[sel_mask],
            y_plot[sel_mask],
            s=sel_sizes,
            alpha=selected_alpha,
            marker=selected_marker,
            edgecolors=(selected_edgecolor if selected_edgecolor else "none"),
            linewidths=0.6 if selected_edgecolor else 0.0,
            rasterize_at=rasterize_at,
            **sel_color_kw,
        )

    # Dedicated colorbar with stable limits across both scatters
    if hue_vals_all is not None and cbar and (vmin is not None) and (vmax is not None):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        cb = fig.colorbar(mappable, ax=ax)
        cb.set_label(
            hue_field.replace("obj__", "").replace("pred__", "").replace("sel__", "")
        )

    # Axes, labels, and enforced x-limits for logic fidelity
    ax.set_xlabel("Logic fidelity (0-1)")
    ax.set_ylabel(y_field_label)
    ax.set_title(f"Trade-off: {y_title_short} vs Logic Fidelity")
    if force_xlim_01:
        ax.set_xlim(0.0, 1.0)
        if xticks_n and xticks_n >= 2:
            ax.set_xticks(np.linspace(0.0, 1.0, xticks_n))

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
        extras={
            "delta": f"{delta:.3g}",
            "selected_marker": selected_marker,
            "selected_n": int(sel_mask.sum()),
            "xlim": "[0,1]" if force_xlim_01 else "auto",
        },
    )

    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)

    if context.save_data:
        # Save logic_fidelity and the actually-plotted Y series under a meaningful column name
        tidy = pd.DataFrame({"logic_fidelity": lf, tidy_col: y_plot})
        context.save_df(tidy)
