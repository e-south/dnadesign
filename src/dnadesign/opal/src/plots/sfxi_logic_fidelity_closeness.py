"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/sfxi_logic_fidelity_closeness.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable

from ..registries.plot import register_plot
from ._events_util import load_events_with_setpoint, resolve_events_path
from ._mpl_utils import annotate_plot_meta


@register_plot("sfxi_logic_fidelity_closeness")
def render(context, params: dict) -> None:
    # ---- Parameters (assertive, yet simple to change) ----
    events_path = resolve_events_path(context)
    top_percentile = params.get("top_percentile")
    if top_percentile is not None:
        top_percentile = float(top_percentile)
        if not (0.0 < top_percentile <= 100.0):
            raise ValueError("top_percentile must be in (0, 100].")

    cmap = str(params.get("cmap", "Greys"))
    # Geometry: keep both main panels square (1:1). Allow explicit figsize_in to tune fonts vs. plot area.
    panel_size_in = float(params.get("panel_size_in", 4.0))  # used if no figsize_in
    figsize_in = params.get("figsize_in")  # optional [W,H] in inches
    cbar_w_in = float(params.get("cbar_width_in", 0.30))
    cbar_pad_in = float(params.get("cbar_pad_in", 0.06))
    gap_in = float(params.get("gap_between_panels_in", 0.40))
    use_violin = bool(params.get("violin", True))
    violin_alpha = float(params.get("violin_alpha", 0.55))
    violin_width = float(params.get("violin_width", 0.9))  # noqa

    # ---- Data (minimal columns; target setpoint from run_meta) ----
    need = {
        "as_of_round",
        "run_id",
        "pred__y_hat_model",
        "obj__diag__setpoint",
    }
    df = load_events_with_setpoint(events_path, need, round_selector=context.rounds)

    # Round selector
    rsel = context.rounds
    if rsel in ("unspecified", "latest"):
        latest = int(df["as_of_round"].max())
        df = df[df["as_of_round"] == latest]
    elif rsel == "all":
        pass
    else:
        lst = rsel if isinstance(rsel, list) else [rsel]
        df = df[df["as_of_round"].isin(lst)]
    if df.empty:
        raise ValueError("No rows matched the requested round selector.")

    # Resolve the 4-vector setpoint
    sp = df["obj__diag__setpoint"].dropna()
    if sp.empty:
        raise ValueError("obj__diag__setpoint not available in events.")
    setpoint = np.asarray(sp.iloc[0], dtype=float).ravel()
    if setpoint.shape[0] != 4 or not np.all(np.isfinite(setpoint)):
        raise ValueError("obj__diag__setpoint must be a finite length-4 vector.")

    # Extract first 4 outputs as "logic" and aggregate means by round
    def _first4(a):
        arr = np.asarray(a, dtype=float).ravel()
        if arr.size < 4:
            return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
        return arr[0:4]

    df["logic_hat_4"] = df["pred__y_hat_model"].map(_first4)

    rows = sorted(df["as_of_round"].unique().astype(int).tolist())
    if not rows:
        raise ValueError("No rounds available after filtering.")

    # Compute mean logic per round (n_rounds x 4) and MSE series
    mean_logic = []
    mse_series = []
    for r in rows:
        sub = df.loc[df["as_of_round"] == r, "logic_hat_4"]
        M = np.vstack(sub.to_list())  # (n, 4) with NaNs possible
        mean_logic.append(np.nanmean(M, axis=0))
        mse_all = np.nanmean((M - setpoint[None, :]) ** 2, axis=1)
        if top_percentile is None:
            mse_use = float(np.nanmean(mse_all))
        else:
            k = max(1, int(np.ceil(len(mse_all) * (top_percentile / 100.0))))
            mse_use = float(np.sort(mse_all)[:k].mean())
        mse_series.append(mse_use)
    mean_logic = np.vstack(mean_logic)

    # Stack target (first row) + per-round means into a single heatmap
    labels_y = ["target"] + [f"r{r}" for r in rows]
    heat = np.vstack([setpoint[None, :], mean_logic])
    if heat.shape[1] != 4:
        raise ValueError("Expected 4 logic dimensions for SFXI plots.")

    # ---- Figure layout: two square panels + cbar. If figsize_in provided, derive panel size from it.
    if figsize_in is not None:
        fig_w, fig_h = float(figsize_in[0]), float(figsize_in[1])
        # Choose the largest square side that fits both panels + cbar + gap
        side = min(fig_h, (fig_w - gap_in - cbar_pad_in - cbar_w_in) / 2.0)
        side = max(0.5, side)
        left_block_w = side + cbar_pad_in + cbar_w_in
        right_block_w = side
        figsize = (fig_w, fig_h)
    else:
        left_block_w = panel_size_in + cbar_pad_in + cbar_w_in
        right_block_w = panel_size_in
        fig_w = left_block_w + gap_in + right_block_w
        fig_h = panel_size_in
        figsize = (fig_w, fig_h)

    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    # Build two axes (heatmap block | MSE). Attach the colorbar to heatmap with inch-precise pad.
    fig = plt.figure(figsize=figsize)  # explicit spacing control
    gs = fig.add_gridspec(1, 2, width_ratios=[left_block_w, right_block_w])
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_mse = fig.add_subplot(gs[0, 1])
    # Convert inch gap to fractional wspace
    avg_ax_w_in = 0.5 * (left_block_w + right_block_w)
    fig.subplots_adjust(wspace=gap_in / max(avg_ax_w_in, 1e-6))

    # Style: hide top/right spines
    for ax in (ax_hm, ax_mse):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Left: grayscale heatmap, square cells, shared bottom x-axis tick labels only
    im = ax_hm.imshow(
        heat,
        aspect="equal",  # square cells
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
    )
    # Make the left axes square independent of data aspect
    try:
        ax_hm.set_box_aspect(1.0)
    except Exception:
        ax_hm.set_aspect("equal", adjustable="box")
    ax_hm.set_yticks(np.arange(heat.shape[0]))
    ax_hm.set_yticklabels(labels_y)
    ax_hm.set_xticks(np.arange(4))
    ax_hm.set_xticklabels(["v00", "v10", "v01", "v11"])
    ax_hm.set_xlabel("Logic components")
    ax_hm.set_title("SFXI logic (0–1) — target + mean by round (grayscale)")

    # colorbar: append next to heatmap with an exact pad/width (inches)
    divider = make_axes_locatable(ax_hm)
    cax = divider.append_axes(
        "right",
        size=axes_size.Fixed(cbar_w_in),  # inches
        pad=axes_size.Fixed(cbar_pad_in),  # inches
    )
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.set_ylabel("logic value", rotation=90, va="center")

    # Right: closeness vs setpoint distributions (violin by default; mean line if not)
    try:
        ax_mse.set_box_aspect(1.0)
    except Exception:
        ax_mse.set_aspect("equal", adjustable="box")
    title_suffix = "" if top_percentile is None else f" (top {top_percentile:.0f}%)"
    if use_violin:
        # Build per-round arrays of MSE (respect top_percentile if requested)
        series = []
        for r in rows:
            sub = df.loc[df["as_of_round"] == r, "logic_hat_4"]
            M = np.vstack(sub.to_list())
            mse_all = np.nanmean((M - setpoint[None, :]) ** 2, axis=1)
            if top_percentile is not None and len(mse_all) > 0:
                k = max(1, int(np.ceil(len(mse_all) * (top_percentile / 100.0))))
                mse_all = np.sort(mse_all)[:k]
            mse_all = mse_all[np.isfinite(mse_all)]
            if mse_all.size == 0:
                raise ValueError(f"No finite MSE values for round {r}.")
            if (
                float(np.nanmax(mse_all)) <= float(np.nanmin(mse_all))
                or mse_all.size < 3
            ):
                raise ValueError(
                    f"Cannot draw violin: round {r} MSE distribution is degenerate "
                    f"(size={mse_all.size}, zero variance)."
                )
            series.append(mse_all)
        parts = ax_mse.violinplot(
            series, positions=rows, widths=0.9, showmeans=True, showextrema=False
        )
        for pc in parts["bodies"]:
            pc.set_alpha(violin_alpha)
        parts["cmeans"].set_alpha(min(1.0, violin_alpha + 0.2))
        ax_mse.set_ylabel("MSE vs setpoint")
        ax_mse.set_title("Pool closeness (violin)" + title_suffix)
    else:
        ax_mse.plot(rows, mse_series, marker="o", linewidth=2.0)
        ax_mse.set_ylabel("MSE vs setpoint")
        ax_mse.set_title("Pool closeness" + title_suffix)
    ax_mse.set_xlabel("Round")
    ax_mse.set_xticks(rows)

    # Annotate + log
    sp_str = "[" + ", ".join(f"{v:.2f}" for v in list(setpoint)) + "]"
    annotate_plot_meta(
        ax_hm,
        hue=None,
        size_by=None,
        alpha=None,
        rasterized=False,
        extras={
            "setpoint": sp_str,
            "top%": (f"{top_percentile:.0f}" if top_percentile else "all"),
        },
    )
    context.logger.info(
        "params sfxi_logic_fidelity_closeness: rounds=%s figsize=%s panel=%.2f top_percentile=%s",
        rows,
        (figsize if figsize_in is not None else "(auto)"),
        (right_block_w if figsize_in is not None else panel_size_in),
        (f"{top_percentile:.0f}" if top_percentile else "all"),
    )

    # Save
    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)

    if context.save_data:
        tidy = pd.DataFrame({"as_of_round": rows, "mse": mse_series})
        context.save_df(tidy)
