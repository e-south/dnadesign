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
    # Users can either give figsize or derive from per-cell geometry (square cells).
    figsize_in = params.get("figsize_in", None)  # e.g., [12, 4.5]
    cell_size_in = float(params.get("cell_size_in", 0.90))  # per heatmap cell (in)
    mse_panel_w_in = float(params.get("mse_panel_width_in", 3.2))  # right column (in)
    cbar_w_in = float(params.get("cbar_width_in", 0.30))  # colorbar width (in)
    cbar_pad_in = float(params.get("cbar_pad_in", 0.06))  # gap heatmap↔cbar (in)
    mse_gap_in = float(params.get("mse_gap_in", 0.40))  # gap (heatmap+cbar)↔MSE (in)
    min_fig_h_in = float(params.get("min_fig_h_in", 3.6))  # guard for 1–2 rows

    # ---- Data (minimal columns; target setpoint from run_meta) ----
    need = {
        "as_of_round",
        "run_id",
        "pred__y_hat_model",
        "obj__diag__setpoint",
    }
    df = load_events_with_setpoint(events_path, need)  # keeps memory narrow

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

    # ---- Figure layout: 2 columns (heatmap+attached-cbar | MSE) ----
    # Derive figure size from cell geometry if user didn't pass figsize explicitly.
    if figsize_in is None:
        left_w = cell_size_in * 4.0
        left_h = cell_size_in * float(heat.shape[0])
        # include cbar width and its pad in the left block width
        left_block_w = left_w + cbar_pad_in + cbar_w_in
        fig_w = left_block_w + mse_panel_w_in
        # Readability guard: don't let the figure collapse to ~1" tall when rows are few.
        fig_h = max(left_h, min_fig_h_in)
        figsize = (fig_w, fig_h)
    else:
        if not (isinstance(figsize_in, (list, tuple)) and len(figsize_in) == 2):
            raise ValueError("figsize_in must be [width_in, height_in].")
        figsize = (float(figsize_in[0]), float(figsize_in[1]))

    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    # Build two axes (heatmap block | MSE). Attach the colorbar to heatmap with inch-precise pad.
    fig = plt.figure(figsize=figsize)  # explicit spacing control; no constrained_layout
    # compute relative width ratio based on inches for intuitive control
    left_block_w = (cell_size_in * 4.0) + cbar_pad_in + cbar_w_in
    gs = fig.add_gridspec(1, 2, width_ratios=[left_block_w, mse_panel_w_in])
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_mse = fig.add_subplot(gs[0, 1])
    # translate desired inch gap into 'wspace' fraction (fraction of average axes width)
    avg_ax_w_in = 0.5 * (left_block_w + mse_panel_w_in)
    fig.subplots_adjust(wspace=mse_gap_in / max(avg_ax_w_in, 1e-6))

    # Style: hide top/right spines
    for ax in (ax_hm, ax_mse):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Left: grayscale heatmap, square cells, shared bottom x-axis tick labels only
    im = ax_hm.imshow(
        heat,
        aspect="equal",  # 1:1 per scalar cell
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
    )
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

    # Right: closeness vs setpoint (MSE)
    ax_mse.plot(rows, mse_series, marker="o", linewidth=2.0)
    ax_mse.set_xlabel("Round")
    ax_mse.set_ylabel("MSE vs setpoint")
    title_suffix = "" if top_percentile is None else f" (top {top_percentile:.0f}%)"
    ax_mse.set_title("Pool closeness" + title_suffix)
    ax_mse.set_xticks(rows)

    # Save
    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)

    if context.save_data:
        tidy = pd.DataFrame({"as_of_round": rows, "mse": mse_series})
        context.save_df(tidy)
