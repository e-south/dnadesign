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


@register_plot("fold_change_vs_logic_fidelity")
def render(context, params: dict) -> None:
    events_path = resolve_events_path(context)

    delta = float(params.get("intensity_log2_offset_delta", 0.0))

    # Ensure setpoint is available (backfill from run_meta if needed)
    need = {
        "as_of_round",
        "run_id",
        "pred__y_hat_model",
        "sel__is_selected",
        "obj__diag__setpoint",
    }
    df = load_events_with_setpoint(events_path, need)

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
    ax.scatter(lf, fold_change, s=20, alpha=0.85)
    if "sel__is_selected" in df.columns:
        sel_mask = df["sel__is_selected"].fillna(False).astype(bool)
        if sel_mask.any():
            idx = df.index[sel_mask]
            ax.scatter(lf[idx], fold_change[idx], s=36, alpha=0.95, edgecolor="black")

    ax.set_xlabel("Logic fidelity (0–1)")
    ax.set_ylabel("Fold change (max–min) in linear intensity")
    ax.set_title("Trade-off: Fold Change vs Logic Fidelity")

    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)

    if context.save_data:
        tidy = pd.DataFrame({"logic_fidelity": lf, "fold_change": fold_change})
        context.save_df(tidy)
