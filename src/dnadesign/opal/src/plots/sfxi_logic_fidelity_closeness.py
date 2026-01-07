"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/sfxi_logic_fidelity_closeness.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.stderr_filter import maybe_install_pyarrow_sysctl_filter
from ..core.utils import ExitCodes, OpalError
from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._mpl_utils import annotate_plot_meta, ensure_mpl_config_dir

if TYPE_CHECKING:
    import numpy as np
    from pyarrow.dataset import Dataset


def _import_pyarrow():
    maybe_install_pyarrow_sysctl_filter()
    from pyarrow import compute as arrow_pc
    from pyarrow import dataset as ds

    return arrow_pc, ds


@register_plot(
    "sfxi_logic_fidelity_closeness",
    meta=PlotMeta(
        summary="Logic fidelity vs closeness to setpoint (observed labels).",
        params={
            "top_percentile": "Optional percentile cutoff for highlighting.",
            "violin": "Show violin distributions (default true).",
            "on_violin_invalid": "error|line (default error).",
            "setpoint_override": "Override setpoint vector (length-4).",
        },
        requires=["observed_round", "y_obs", "objective__params"],
        notes=["Reads ledger.labels + ledger.runs for setpoint."],
    ),
)
def render(context, params: dict) -> None:
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable

    arrow_pc, ds = _import_pyarrow()
    # ---- Parameters (assertive, yet simple to change) ----
    # Source is now *observed* labels (ledger.labels) instead of predictions.
    outputs_dir = resolve_outputs_dir(context)  # ledger sinks live here
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
    violin_width = float(params.get("violin_width", 0.9))
    # Validation policy for violin inputs (assertive, explicit)
    violin_min_points = int(params.get("violin_min_points", 3))
    violin_require_nonzero_var = bool(params.get("violin_require_nonzero_var", True))
    on_violin_invalid = str(params.get("on_violin_invalid", "error")).strip().lower()  # "error" | "line"
    if on_violin_invalid not in {"error", "line"}:
        raise ValueError("on_violin_invalid must be 'error' or 'line'.")

    # Logic extraction policy for y_obs
    # Default: first 4 components of y_obs are logic in [0,1].
    # Set `coerce_clip: true` to clip small out-of-range noise into [0,1].
    coerce_clip = bool(params.get("coerce_clip", False))
    _TOL = 1e-6

    # ---- Data: read ledger.labels and join a setpoint from ledger.runs ----
    root = outputs_dir
    labels_path = root / "ledger.labels.parquet"
    runs_path = root / "ledger.runs.parquet"
    if not labels_path.exists():
        raise OpalError(
            f"Missing labels sink: {labels_path}. Run `opal ingest-y -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )
    if not runs_path.exists():
        raise OpalError(
            f"Missing runs sink (for setpoint): {runs_path}. Run `opal run -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )

    # Helper: filter by rounds (on 'observed_round')3
    def _round_filter(dset: Dataset):
        sel = context.rounds
        if sel in (None, "all"):
            return None
        if sel in ("unspecified", "latest"):
            t = dset.to_table(columns=["observed_round"])
            if t.num_rows == 0:
                return None
            latest = int(pd.Series(t.column("observed_round").to_pylist()).max())
            return arrow_pc.field("observed_round") == latest
        if isinstance(sel, list):
            try:
                vals = [int(x) for x in sel]
            except Exception:
                return None
            if not vals:
                return None
            return arrow_pc.field("observed_round").isin(vals)
        try:
            r = int(sel)
            return arrow_pc.field("observed_round") == r
        except Exception:
            return None

    # Read labels (observed Y)
    dlab = ds.dataset(str(labels_path))
    names = {f.name for f in dlab.schema}
    need = {"observed_round", "y_obs"}
    missing = sorted(need - names)
    if missing:
        raise ValueError(f"ledger.labels missing columns: {missing}")
    filt = _round_filter(dlab)
    df = dlab.to_table(columns=list(need), filter=filt).to_pandas()
    if df.empty:
        raise ValueError("ledger.labels had zero rows for the requested rounds.")

    # Resolve setpoint: override > specific round > latest
    def _extract_setpoint(obj):
        try:
            return [float(x) for x in (obj or {}).get("setpoint_vector", [])]
        except Exception:
            return None

    # Param overrides for setpoint (optional but assertive)
    sp_override = params.get("setpoint") or params.get("setpoint_override")
    sp_round = params.get("setpoint_round")  # int (as_of_round in ledger.runs)
    if sp_override is not None:
        sp_arr = np.asarray(list(sp_override), dtype=float).ravel()
        if sp_arr.size != 4 or not np.all(np.isfinite(sp_arr)):
            raise ValueError("setpoint_override must be a finite length-4 vector.")
        setpoint = sp_arr
    else:
        druns = ds.dataset(str(runs_path))
        nn = {f.name for f in druns.schema}
        need_runs = {"as_of_round", "objective__params"}
        miss = sorted(need_runs - nn)
        if miss:
            raise ValueError(f"ledger.runs missing columns: {miss}")
        if sp_round is None:
            # Pick the latest run that has a setpoint
            meta = druns.to_table(columns=list(need_runs)).to_pandas()
            meta["setpoint"] = meta["objective__params"].map(_extract_setpoint)
            meta = meta.dropna(subset=["setpoint"])
            if meta.empty:
                raise ValueError("No setpoint_vector found in ledger.runs objective__params.")
            meta = meta.sort_values(["as_of_round"]).tail(1)
            setpoint = np.asarray(list(meta["setpoint"].iloc[0]), dtype=float).ravel()
        else:
            try:
                sp_round = int(sp_round)
            except Exception as e:
                raise ValueError("setpoint_round must be an integer.") from e
            filt_r = arrow_pc.field("as_of_round") == sp_round
            meta = druns.to_table(columns=list(need_runs), filter=filt_r).to_pandas()
            meta["setpoint"] = meta["objective__params"].map(_extract_setpoint)
            meta = meta.dropna(subset=["setpoint"])
            if meta.empty:
                raise ValueError(f"No setpoint_vector found in ledger.runs for as_of_round={sp_round}.")
            setpoint = np.asarray(list(meta["setpoint"].iloc[0]), dtype=float).ravel()

    if setpoint.size != 4 or not np.all(np.isfinite(setpoint)):
        raise ValueError("Resolved setpoint must be a finite length-4 vector.")

    # Extract first 4 components of y_obs as 'logic' in [0,1] (assertive)
    def _logic4_from_yobs(y):
        arr = np.asarray(y, dtype=float).ravel()
        if arr.size < 4:
            raise ValueError(f"y_obs must have at least 4 components; got length={arr.size}.")
        lg = arr[:4]
        if not np.all(np.isfinite(lg)):
            raise ValueError("y_obs logic components contain non-finite values.")
        if coerce_clip:
            return np.clip(lg, 0.0, 1.0)
        lo, hi = float(np.min(lg)), float(np.max(lg))
        if lo < -_TOL or hi > (1.0 + _TOL):
            raise ValueError(
                "Observed logic components must lie in [0,1]. "
                "Found range=({:.4g},{:.4g}). "
                "Pass coerce_clip: true to clip into [0,1].".format(lo, hi)
            )
        return lg

    df["logic_obs_4"] = df["y_obs"].map(_logic4_from_yobs)

    rows = sorted(df["observed_round"].unique().astype(int).tolist())
    if not rows:
        raise ValueError("No observed rounds available after filtering.")

    # Compute mean logic per observed_round (n_rounds x 4) and an MSE series
    mean_logic = []
    mse_series = []
    # Keep per-round MSE arrays (after percentile filtering) for potential violin
    mse_arrays_by_round: list[np.ndarray] = []
    for r in rows:
        sub = df.loc[df["observed_round"] == r, "logic_obs_4"]
        M = np.vstack(sub.to_list())  # (n, 4)
        mean_logic.append(np.nanmean(M, axis=0))
        mse_all = np.nanmean((M - setpoint[None, :]) ** 2, axis=1)
        if top_percentile is None:
            sel = mse_all
            mse_use = float(np.nanmean(sel))
        else:
            k = max(1, int(np.ceil(len(mse_all) * (top_percentile / 100.0))))
            sel = np.sort(mse_all)[:k]
            mse_use = float(sel.mean())
        # Store cleaned, finite arrays for violin viability checks
        sel = sel[np.isfinite(sel)]
        mse_arrays_by_round.append(sel)
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

    # ---- Decide how to draw the right panel (assertive preflight, no hidden fallbacks)
    def _violins_viable(series_list: list[np.ndarray]) -> tuple[bool, list[str]]:
        problems: list[str] = []
        for r, arr in zip(rows, series_list):
            n = int(arr.size)
            if n < violin_min_points:
                problems.append(f"r{r}: n={n} < min={violin_min_points}")
                continue
            if violin_require_nonzero_var:
                amax = float(np.nanmax(arr))
                amin = float(np.nanmin(arr))
                if not (amax > amin):
                    problems.append(f"r{r}: zero variance (all {amin:.3g})")
        return (len(problems) == 0, problems)

    draw_violin = bool(use_violin)
    violin_ok, issues = _violins_viable(mse_arrays_by_round) if draw_violin else (False, [])
    if draw_violin and not violin_ok:
        msg = "Violin disabled: " + "; ".join(issues)
        if on_violin_invalid == "error":
            raise ValueError("Cannot draw violin — " + "; ".join(issues))
        # switch to line explicitly
        draw_violin = False
        context.logger.info("[sfxi_logic_closeness] %s", msg)

    if draw_violin:
        parts = ax_mse.violinplot(
            mse_arrays_by_round,
            positions=rows,
            widths=violin_width,
            showmeans=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_alpha(violin_alpha)
        parts["cmeans"].set_alpha(min(1.0, violin_alpha + 0.2))
        ax_mse.set_ylabel("MSE vs setpoint")
        ax_mse.set_title("Pool closeness (violin)" + title_suffix)
    else:
        ax_mse.plot(rows, mse_series, marker="o", linewidth=2.0)
        ax_mse.set_ylabel("MSE vs setpoint")
        subtitle = "mean line (auto)" if use_violin else "mean line"
        ax_mse.set_title(f"Pool closeness — {subtitle}" + title_suffix)
    ax_mse.set_xlabel("Observed round")
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
            "source": "y_obs (ledger.labels)",
        },
    )
    context.logger.info(
        "params sfxi_logic_fidelity_closeness: source=labels rounds=%s figsize=%s panel=%.2f top_percentile=%s coerce_clip=%s draw=%s violin_min_pts=%d nonzero_var=%s",  # noqa
        rows,
        (figsize if figsize_in is not None else "(auto)"),
        (right_block_w if figsize_in is not None else panel_size_in),
        (f"{top_percentile:.0f}" if top_percentile else "all"),
        bool(coerce_clip),
        ("violin" if draw_violin else "line"),
        violin_min_points,
        violin_require_nonzero_var,
    )

    # Save
    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)

    if context.save_data:
        tidy = pd.DataFrame({"observed_round": rows, "mse": mse_series})
        context.save_df(tidy)
