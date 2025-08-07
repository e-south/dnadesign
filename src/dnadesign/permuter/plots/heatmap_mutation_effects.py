"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/plots/heatmap_mutation_effects.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parse_mod(token: str) -> tuple[int, str, str]:
    """
    Parse a modification string like "A5T" → (position=5, from='A', to='T').
    """
    from_res = token[0]
    to_res = token[-1]
    pos = int(token[1:-1])
    return pos, from_res, to_res


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
) -> None:
    """
    Heatmap: x=position, y=mutated residue → average metric.
    Highlights the original reference residue row in white for every position.
    Uses 'crest' diverging palette, square cells, with a slim horizontal colorbar
    placed below the heatmap. No cell borders.
    """
    # 1) collect per‐mutation records
    records: List[Dict] = []
    for _, row in all_df.iterrows():
        for mod in row["modifications"]:
            pos, _from, to = _parse_mod(mod)
            records.append(
                {
                    "position": pos,
                    "to_res": to,
                    "score": row["score"],
                }
            )

    if not records:
        raise RuntimeError(f"{job_name}: no mutation records to plot")

    dfm = pd.DataFrame(records)

    # 2) extract the original reference sequence (unmutated seed)
    seed = next(
        (
            r
            for r in all_df.to_dict("records")
            if r["round"] == 1 and not r["modifications"]
        ),
        None,
    )
    if seed is None:
        raise RuntimeError(f"{job_name}: original reference not found")
    ref_seq = seed["sequence"]

    # 3) full range of positions and residues
    full_positions = list(range(1, len(ref_seq) + 1))
    residues = sorted(set(dfm["to_res"]).union(set(ref_seq)))

    # 4) pivot to wide form, ensure all rows & cols present
    pivot = dfm.pivot_table(
        index="to_res", columns="position", values="score", aggfunc="mean"
    ).reindex(index=residues, columns=full_positions)

    # 5) styling
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(len(full_positions) * 0.3, len(residues) * 0.3))
    ax.grid(False)
    ax.set_aspect("equal", "box")

    metric_label = all_df["score_type"].iloc[0].replace("_", " ").title()

    # 6) draw heatmap with no cell borders and a slim horizontal colorbar below
    hm = sns.heatmap(
        pivot,
        cmap="crest",
        linewidths=0,  # no borders between cells
        linecolor=None,
        square=True,
        ax=ax,
        cbar=True,
        cbar_kws={
            "orientation": "horizontal",
            "shrink": 0.2,
            "pad": 0.02,
            "aspect": 20,
            "label": metric_label,
        },
    )

    # 7) labels & ticks
    ax.set_xlabel("Sequence position", fontsize=8)
    ax.set_ylabel("Mutated residue", fontsize=8)
    ax.set_title(f"{job_name} ({all_df['ref_name'].iloc[0]})", fontsize=10)
    ax.tick_params(axis="x", labelsize=6, rotation=0)
    ax.tick_params(axis="y", labelsize=6, rotation=0)

    # 8) highlight the reference residue cell at each position
    for j, pos in enumerate(full_positions):
        ref_res = ref_seq[pos - 1]
        i = residues.index(ref_res)
        hm.add_patch(
            plt.Rectangle(
                (j, i), 1, 1, fill=True, edgecolor="white", facecolor="white", lw=0
            )
        )

    # 9) tighten layout
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.95)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
