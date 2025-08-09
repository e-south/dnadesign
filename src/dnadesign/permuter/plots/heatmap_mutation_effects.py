"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/plots/heatmap_mutation_effects.py

Heatmap: x = position, y = mutated residue → average score.
Prefers provided ref_sequence; falls back to seed; avoids seaborn.

Notes:
  • Only simple single-nucleotide edits like "A5T" are visualized.
    Complex tokens (e.g., codon edits "AAA@10→GCT(ALA)") are ignored.
  • Reference residue at each position is outlined.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _choose_y_series(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    if "objective_score" in df.columns:
        return df["objective_score"], "Objective score"

    if "norm_metrics" in df.columns and not df["norm_metrics"].isna().all():
        key: Optional[str] = None
        for d in df["norm_metrics"]:
            if isinstance(d, dict) and d:
                key = next(iter(d.keys()))
                break
        if key:
            return (
                df["norm_metrics"].apply(lambda d: (d or {}).get(key, None)),
                f"Norm {key}",
            )

    if "score" in df.columns:
        return df["score"], "Score"

    raise RuntimeError("No objective_score, norm_metrics, or score available to plot.")


def _parse_simple_nt_edit(token: str) -> Optional[tuple[int, str, str]]:
    """
    Parse "A5T" → (position=5, from='A', to='T').
    Return None for non-conforming tokens (e.g., codon edits).
    """
    s = str(token).strip()
    if len(s) < 3:
        return None
    first, last = s[0], s[-1]
    mid = s[1:-1]
    if not (first.isalpha() and last.isalpha() and mid.isdigit()):
        return None
    return int(mid), first.upper(), last.upper()


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
    ref_sequence: Optional[str] = None,
) -> None:
    """
    Heatmap of average score by position × mutated residue.
    """
    df = all_df.copy()

    # y-series selection
    y, y_label = _choose_y_series(df)
    df = df.assign(_y=y).dropna(subset=["_y"])

    # pick reference sequence (prefer explicit arg, else seed fallback)
    ref_seq = (ref_sequence or "").strip().upper()
    if not ref_seq:
        seed = next(
            (
                r
                for r in df.to_dict("records")
                if int(r.get("round", 0)) == 1
                and isinstance(r.get("modifications"), list)
                and len(r["modifications"]) == 0
            ),
            None,
        )
        if seed and seed.get("sequence"):
            ref_seq = str(seed["sequence"]).upper()

    if not ref_seq:
        raise RuntimeError(
            f"{job_name}: reference sequence unavailable. "
            f"Re-run the job (or analysis mode) so MANIFEST contains ref_sequence."
        )

    # collect simple per-mutation records
    records: List[Dict] = []
    for _, row in df.iterrows():
        mods = row.get("modifications", [])
        if not isinstance(mods, list):
            continue
        for token in mods:
            parsed = _parse_simple_nt_edit(token)
            if parsed is None:
                continue
            pos, _from, to = parsed
            records.append({"position": pos, "to_res": to, "y": float(row["_y"])})

    if not records:
        raise RuntimeError(f"{job_name}: no simple single-nucleotide edits to plot")

    dfm = pd.DataFrame(records)

    # define full axes
    full_positions = list(range(1, len(ref_seq) + 1))
    residues = sorted(set(dfm["to_res"]).union(set(list(ref_seq))))
    if not residues:
        raise RuntimeError(f"{job_name}: no residues available for heatmap")

    # aggregate mean per cell
    pivot = dfm.pivot_table(
        index="to_res", columns="position", values="y", aggfunc="mean"
    ).reindex(index=residues, columns=full_positions)
    mat = pivot.to_numpy(dtype=float)

    # figure sizing
    fig_w = max(6, min(14, len(full_positions) * 0.15 + 2))
    fig_h = max(3.5, min(10, len(residues) * 0.3 + 1.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    # axes & ticks
    ax.set_xlabel("Sequence position", fontsize=9)
    ax.set_ylabel("Mutated residue", fontsize=9)
    ref_name = (
        df["ref_name"].iloc[0] if "ref_name" in df.columns and not df.empty else ""
    )
    ax.set_title(f"{job_name}{f' ({ref_name})' if ref_name else ''}", fontsize=10)

    ax.set_xticks(
        np.linspace(
            0, len(full_positions) - 1, num=min(12, len(full_positions)), dtype=int
        )
    )
    ax.set_xticklabels(
        [str(full_positions[i]) for i in ax.get_xticks().astype(int)], fontsize=7
    )
    ax.set_yticks(range(len(residues)))
    ax.set_yticklabels(residues, fontsize=7)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(y_label, rotation=90, va="center", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # highlight reference residue at each position (outline box)
    for j, pos in enumerate(full_positions):
        ref_res = ref_seq[pos - 1]
        if ref_res in residues:
            i = residues.index(ref_res)
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )

    # clean style
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
