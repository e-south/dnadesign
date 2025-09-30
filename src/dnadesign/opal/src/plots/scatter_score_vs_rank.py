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
from pyarrow import dataset as ds

from ..registries.plot import register_plot
from ._events_util import resolve_events_path


@register_plot("scatter_score_vs_rank")
def render(context, params: dict) -> None:
    events_path = resolve_events_path(context)

    score_field = params.get("score_field", "pred__y_obj_scalar")
    dset = ds.dataset(str(events_path))
    needed = {
        "as_of_round",
        "run_id",
        "id",
        score_field,
        "sel__rank_competition",
        "sel__is_selected",
    }
    schema_names = {f.name for f in dset.schema}
    missing = sorted(needed - schema_names)
    if missing:
        raise ValueError(f"ledger.index.parquet missing columns: {missing}")

    df = dset.to_table(columns=list(needed)).to_pandas()
    if df.empty:
        raise ValueError("ledger.index.parquet contained zero rows after projection.")

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

    if "sel__rank_competition" not in df.columns:
        df = df.sort_values(
            ["as_of_round", score_field], ascending=[True, False]
        ).assign(
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
        sub = df[df["as_of_round"] == r].sort_values(
            "sel__rank_competition", ascending=True
        )
        figsize = tuple(params.get("figsize_in", (8.5, 5.0)))
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        x = sub["sel__rank_competition"].to_numpy()
        y = sub[score_field].to_numpy(dtype=float)
        ax.plot(x, y, linewidth=1.5, alpha=0.8)
        ax.scatter(x, y, s=22, alpha=0.9)
        if "sel__is_selected" in sub.columns:
            sel_mask = (
                sub["sel__is_selected"]
                .astype("boolean")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            if sel_mask.any():
                ax.scatter(
                    x[sel_mask],
                    y[sel_mask],
                    s=36,
                    alpha=1.0,
                    edgecolor="black",
                )
        ax.set_xlabel("Rank (competition)")
        ax.set_ylabel("Objective score")
        ax.set_title(f"Score vs Rank — round {r}")
        ax.set_xlim(left=sub["sel__rank_competition"].max(), right=1)
    else:
        default_multi = (9.5, 6.0)
        figsize = tuple(params.get("figsize_in", default_multi))
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for r, sub in df.groupby("as_of_round"):
            sub = sub.sort_values("sel__rank_competition")
            ax.plot(
                sub["sel__rank_competition"],
                sub[score_field],
                label=f"r{r}",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.scatter(sub["sel__rank_competition"], sub[score_field], s=18, alpha=0.85)
        ax.legend(title="round", frameon=False)
        ax.set_xlabel("Rank (competition)")
        ax.set_ylabel("Objective score")
        ax.set_title("Score vs Rank by Round")
        ax.set_xlim(left=df["sel__rank_competition"].max(), right=1)

    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
    plt.close(fig)

    if context.save_data:
        keep = [
            "as_of_round",
            "id",
            "sel__rank_competition",
            "sel__is_selected",
            score_field,
        ]
        context.save_df(df[keep])
