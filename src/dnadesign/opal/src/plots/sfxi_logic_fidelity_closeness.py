"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/sfxi_logic_fidelity_closeness.py

Plots observed label logic fidelity vs closeness for SFXI campaigns. Reads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.stderr_filter import maybe_install_pyarrow_sysctl_filter
from ..core.utils import ExitCodes, OpalError
from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._mpl_utils import (
    COLORBLIND_PALETTE,
    DEFAULT_LANDSCAPE_FIGSIZE,
    DEFAULT_SQUARE_FIGSIZE,
    add_flush_colorbar,
    apply_notebook_axes_style,
    apply_plot_style,
    ensure_mpl_config_dir,
    math_label,
    save_notebook_square_figure,
    sequential_colormap,
)
from .sfxi_diag_data import parse_setpoint_from_runs

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
        summary="Observed SFXI logic closeness to the setpoint by round.",
        params={
            "top_percentile": "Optional percentile cutoff for highlighting.",
            "violin": "Show violin distributions (default true).",
            "on_violin_invalid": "error|line (default error).",
            "setpoint_override": "Override setpoint vector (length-4).",
        },
        requires=["observed_round", "y_obs", "objective__defs_json"],
        notes=["Reads outputs/ledger/labels.parquet + outputs/ledger/runs.parquet for setpoint."],
        data_shape="observed label agreement matrix",
        tidy_schema=["observed_round", "mse"],
        objective_family="sfxi",
        data_layer="labels_objective",
        round_scope="single_or_round_history",
        label_requirement="required",
        failure_modes=[
            "missing labels or runs ledger",
            "invalid length-8 observed label vectors",
            "missing setpoint metadata",
            "insufficient points for violin mode",
        ],
    ),
)
def render(context, params: dict) -> None:
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl

    apply_plot_style()
    arrow_pc, ds = _import_pyarrow()
    # ---- Parameters (assertive, yet simple to change) ----
    # Source is now *observed* labels (outputs/ledger/labels.parquet) instead of predictions.
    outputs_dir = resolve_outputs_dir(context)  # ledger sinks live here
    top_percentile = params.get("top_percentile")
    if top_percentile is not None:
        top_percentile = float(top_percentile)
        if not (0.0 < top_percentile <= 100.0):
            raise ValueError("top_percentile must be in (0, 100].")

    cmap = sequential_colormap(params.get("cmap", "opal_seafoam"))
    # Geometry: keep both main panels square (1:1). Allow explicit figsize_in to tune fonts vs. plot area.
    panel_size_in = float(params.get("panel_size_in", 4.0))  # used if no figsize_in
    figsize_in = params.get("figsize_in")  # optional [W,H] in inches
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

    # ---- Data: read outputs/ledger/labels.parquet and join a setpoint from outputs/ledger/runs.parquet ----
    root = outputs_dir
    labels_path = root / "ledger" / "labels.parquet"
    runs_path = root / "ledger" / "runs.parquet"
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
        raise ValueError(f"outputs/ledger/labels.parquet missing columns: {missing}")
    filt = _round_filter(dlab)
    df = dlab.to_table(columns=list(need), filter=filt).to_pandas()
    if df.empty:
        raise ValueError("outputs/ledger/labels.parquet had zero rows for the requested rounds.")

    # Param overrides for setpoint (optional but assertive)
    sp_override = params.get("setpoint") or params.get("setpoint_override")
    sp_round = params.get("setpoint_round")  # int (as_of_round in outputs/ledger/runs.parquet)
    if sp_override is not None:
        sp_arr = np.asarray(list(sp_override), dtype=float).ravel()
        if sp_arr.size != 4 or not np.all(np.isfinite(sp_arr)):
            raise ValueError("setpoint_override must be a finite length-4 vector.")
        setpoint = sp_arr
    else:
        runs = pl.read_parquet(runs_path)
        if context.run_id is not None:
            runs = runs.filter(pl.col("run_id") == str(context.run_id))
        if sp_round is None:
            latest = int(runs["as_of_round"].max())
            runs = runs.filter(pl.col("as_of_round") == latest)
        else:
            try:
                sp_round = int(sp_round)
            except Exception as e:
                raise ValueError("setpoint_round must be an integer.") from e
            runs = runs.filter(pl.col("as_of_round") == sp_round)
        setpoint = np.asarray(
            parse_setpoint_from_runs(runs, selection_view_id=context.selection_view_id),
            dtype=float,
        ).ravel()

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
    labels_y = ["Target"] + [f"R{r}" for r in rows]
    heat = np.vstack([setpoint[None, :], mean_logic])
    if heat.shape[1] != 4:
        raise ValueError("Expected 4 logic dimensions for SFXI plots.")

    # ---- Figure layout: heatmap plus MSE panel. The SFXI heatmap has 4 columns
    # and many round rows, so a side-by-side landscape layout avoids overlap.
    if figsize_in is not None:
        figsize = (float(figsize_in[0]), float(figsize_in[1]))
    else:
        figsize = (10.8, 5.4)

    fig, (ax_hm, ax_mse) = plt.subplots(
        1,
        2,
        figsize=figsize if figsize != DEFAULT_SQUARE_FIGSIZE else DEFAULT_LANDSCAPE_FIGSIZE,
        gridspec_kw={"width_ratios": [0.85, 1.0], "wspace": 0.40},
    )

    apply_notebook_axes_style(ax_hm, grid=False, square=False)
    apply_notebook_axes_style(ax_mse, square=False)

    # Left: mean observed logic by round. aspect='equal' keeps each heatmap cell square.
    im = ax_hm.imshow(
        heat,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
    )
    ax_hm.set_yticks(np.arange(heat.shape[0]))
    ax_hm.set_yticklabels(labels_y)
    ax_hm.set_xticks(np.arange(4))
    ax_hm.set_xticklabels(["v00", "v10", "v01", "v11"], rotation=45, ha="right")
    ax_hm.set_xlabel("Logic component")
    ax_hm.set_ylabel("Observed round")
    ax_hm.set_title("Observed logic vs target")

    add_flush_colorbar(fig, ax_hm, im, label="Logic value")

    # Right: closeness vs setpoint distributions (violin by default; mean line if not)
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
            body.set_facecolor(COLORBLIND_PALETTE[0])
            body.set_edgecolor("#444444")
            body.set_linewidth(0.7)
            body.set_alpha(violin_alpha)
        parts["cmeans"].set_alpha(min(1.0, violin_alpha + 0.2))
        ax_mse.set_ylabel(math_label("mse_to_reference"))
        ax_mse.set_title("Observed-label MSE by round" + title_suffix)
    else:
        ax_mse.axhline(0.0, color="#B8B8B8", linewidth=0.9, linestyle="--", zorder=0)
        ax_mse.plot(
            rows,
            mse_series,
            marker="o",
            color=COLORBLIND_PALETTE[0],
            linewidth=2.2,
            markersize=6,
        )
        ax_mse.set_ylabel(math_label("mse_to_reference"))
        subtitle = "mean line (auto)" if use_violin else "mean line"
        ax_mse.set_title(f"Observed-label MSE, {subtitle}" + title_suffix)
    ax_mse.set_xlabel("Observed round")
    ax_mse.set_xticks(rows)

    context.logger.info(
        "params sfxi_logic_fidelity_closeness: source=labels rounds=%s figsize=%s panel=%.2f top_percentile=%s coerce_clip=%s draw=%s violin_min_pts=%d nonzero_var=%s",  # noqa
        rows,
        (figsize if figsize_in is not None else "(auto)"),
        panel_size_in,
        (f"{top_percentile:.0f}" if top_percentile else "all"),
        bool(coerce_clip),
        ("violin" if draw_violin else "line"),
        violin_min_points,
        violin_require_nonzero_var,
    )

    # Save
    out = context.output_dir / context.filename
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.16, top=0.86, wspace=0.42)
    save_notebook_square_figure(fig, out, dpi=context.dpi, tight=False)
    plt.close(fig)

    if context.save_data:
        tidy = pd.DataFrame({"observed_round": rows, "mse": mse_series})
        context.save_df(tidy)
