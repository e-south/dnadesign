"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/percent_high_activity_over_rounds.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from pyarrow import dataset as ds

from ..registries.plot import register_plot
from ._events_util import resolve_events_path


@register_plot("percent_high_activity_over_rounds")
def render(context, params: dict) -> None:
    threshold = float(params.get("threshold", 0.8))
    events_path = resolve_events_path(context)
    dset = ds.dataset(str(events_path))
    need = {"as_of_round", "pred__y_obj_scalar"}
    names = {f.name for f in dset.schema}
    missing = sorted(need - names)
    if missing:
        raise ValueError(f"ledger.index.parquet missing columns: {missing}")
    df = dset.to_table(columns=list(need)).to_pandas()
    if df.empty:
        raise ValueError("ledger.index.parquet contained zero rows after projection.")

    rsel = context.rounds
    if rsel == "unspecified":
        latest = int(df["as_of_round"].max())
        df = df[df["as_of_round"] == latest]
    elif rsel != "all":
        df = df[df["as_of_round"].isin(rsel if isinstance(rsel, list) else [rsel])]
    if df.empty:
        raise ValueError("No rows matched the requested round selector.")

    grp = df.groupby("as_of_round").agg(
        total=("pred__y_obj_scalar", "size"),
        high=("pred__y_obj_scalar", lambda s: (s >= threshold).sum()),
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
    ax.plot(grp["as_of_round"], grp["percent_high"], marker="o", linewidth=2.0)
    ax.set_xlabel("Round")
    ax.set_ylabel(f"% ≥ {threshold:.2f}")
    ax.set_title("Percent of High Activity Over Rounds")
    ax.set_xticks(grp["as_of_round"])
    ax.set_ylim(0, 100)

    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)
    if context.save_data:
        context.save_df(grp)
