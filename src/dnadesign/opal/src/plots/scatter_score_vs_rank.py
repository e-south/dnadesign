"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/scatter_score_vs_rank.py

Scatter of `score_field` vs `rank`, optionally colored by round and
optionally highlighting selected==True. Efficient Parquet scans via pyarrow.

Params (opaque to core):
- score_field (str, required): name of the score column in events.parquet.
- hue (None|"round"): default None; if "round", series by round.
- highlight_selected (bool): default False; emphasize selected points if column exists.
- title (str): optional title.

Round handling:
- If context.rounds == "unspecified": pick latest round found in events.parquet.

Built-ins:
- Uses context.data_paths["events"] (auto-injected) unless overridden by YAML data entries.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
from pyarrow import dataset as ds

from ..registries.plot import register_plot


@register_plot("scatter_score_vs_rank")
def render(context, params: dict) -> None:
    score_field = params.get("score_field")
    if not score_field:
        raise ValueError("params.score_field is required (e.g., 'score_sfxi').")

    events_path = context.data_paths.get("events")
    if not events_path or not events_path.exists():
        raise FileNotFoundError(
            "Built-in 'events' source not found in campaign directory."
        )

    # Efficient projected scan of Parquet
    dset = ds.dataset(str(events_path))
    available = {f.name for f in dset.schema}
    cols: List[str] = ["round", "id", "rank", "selected"]
    if score_field not in available:
        raise ValueError(f"Score field '{score_field}' not found in events schema.")
    cols = [c for c in cols if c in available] + [score_field]

    table = dset.to_table(columns=cols)
    df = table.to_pandas()

    if df.empty:
        raise ValueError("events.parquet contained zero rows after projection.")

    # Resolve round selection
    rsel = context.rounds
    if rsel == "unspecified":
        if "round" not in df.columns:
            raise ValueError("No 'round' column to resolve latest from events.parquet.")
        latest = int(df["round"].max())
        context.logger.info(f"No --round provided → using latest (round={latest}).")
        df = df[df["round"] == latest]
    elif rsel == "latest":
        latest = int(df["round"].max())
        df = df[df["round"] == latest]
    elif rsel == "all":
        pass
    else:  # explicit list of ints
        if "round" not in df.columns:
            raise ValueError(
                "Round list provided but 'round' column missing in events.parquet."
            )
        df = df[df["round"].isin(rsel)]

    if df.empty:
        raise ValueError("No rows matched the requested round selector.")

    # Ensure rank exists; derive if absent by sorting score desc
    if "rank" not in df.columns:
        if "round" in df.columns:
            df = df.sort_values(["round", score_field], ascending=[True, False]).assign(
                rank=lambda x: x.groupby("round").cumcount() + 1
            )
        else:
            df = df.sort_values(score_field, ascending=False)
            df["rank"] = range(1, len(df) + 1)

    # Plot
    hue = params.get("hue")  # None or "round"
    highlight = bool(params.get("highlight_selected", False))
    title = params.get("title") or "Score vs Rank"

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    # Lightweight, readable defaults within plugin
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if hue == "round" and "round" in df.columns:
        for r, sub in df.groupby("round"):
            ax.scatter(sub["rank"], sub[score_field], label=f"r{r}", s=18, alpha=0.85)
        ax.legend(title="round", frameon=False)
    else:
        ax.scatter(df["rank"], df[score_field], s=18, alpha=0.85)

    if highlight and "selected" in df.columns:
        sel = df[df["selected"] == True]  # noqa: E712
        if not sel.empty:
            ax.scatter(
                sel["rank"], sel[score_field], s=38, alpha=0.95, edgecolor="black"
            )

    ax.set_xlabel("Rank (1 = best)")
    ax.set_ylabel(score_field)
    ax.set_title(title)

    out_path = context.output_dir / context.filename
    if context.format == "png":
        fig.savefig(out_path, dpi=context.dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, bbox_inches="tight")  # svg/pdf ignore dpi
    plt.close(fig)

    # Optional tidy export (save next to image)
    if context.save_data:
        context.save_df(
            df[["rank", score_field] + (["round"] if "round" in df.columns else [])]
        )
