"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/plots/position_scatter_and_heatmap.py

Combined figure with shared X:
  • Top: round-1 scatter (mean±SD) with optional baseline
  • Middle: reference sequence strip (monospace), one char per position
  • Bottom: heatmap (mutated residue x position) with 1:1 squares
  • Colorbar spans the full height of the superplot (right side, extra narrow)
  • Subtitle shows evaluator(s)

Baselines:
  • 'log_likelihood' → y=0 (gray line)
  • 'log_likelihood_ratio' or 'llr' → y=1 (gray line)

Single metric → axis label = that metric; multi-metric → 'Objective'.

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
from matplotlib.ticker import NullLocator

# Short → pretty mapping for axis/labels
_METRIC_LABELS = {
    "ll": "log_likelihood",
    "llr": "log_likelihood_ratio",
    "emb": "embedding_distance",
}


def _pretty_metric(mid: str | None, fallback: str) -> str:
    """Map compact metric ids like 'll' → 'log_likelihood' for labels."""
    return _METRIC_LABELS.get(str(mid).strip(), str(mid).strip()) if mid else fallback


def _series_for_metric(
    df: pd.DataFrame, metric_id: Optional[str]
) -> Tuple[pd.Series, str]:
    """
    Return the series to plot (normalized metric, raw metric, or objective)
    and a readable axis label.
    """
    if metric_id and "norm_metrics" in df.columns:
        s = df["norm_metrics"].apply(lambda d: (d or {}).get(metric_id, None))
        if not s.isna().all():
            return s, _pretty_metric(metric_id, "Objective")
    if metric_id and "metrics" in df.columns:
        s = df["metrics"].apply(lambda d: (d or {}).get(metric_id, None))
        if not s.isna().all():
            return s, _pretty_metric(metric_id, "Objective")
    if "objective_score" in df.columns:
        return df["objective_score"], "Objective"
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


def _extract_position(modifications: List[str] | str) -> int:
    """Parse position from simple edit tokens like 'A12T'."""
    mods = modifications if isinstance(modifications, list) else [modifications]
    for token in mods:
        digits = "".join(ch for ch in str(token) if ch.isdigit())
        if digits:
            return int(digits)
    return 0


def _baseline_for_metric(label: str) -> Optional[float]:
    """Return a baseline y-value for common metrics, else None."""
    lab = (label or "").lower()
    if "ratio" in lab or lab == "llr" or "llr" in lab:
        return 1.0
    if "likelihood" in lab or lab == "ll":
        return 0.0
    return None


def _parse_simple_nt_edit(token: str) -> Optional[tuple[int, str, str]]:
    """Parse a simple single-nucleotide edit token like 'A12T'."""
    s = str(token).strip()
    if len(s) < 3:
        return None
    first, last, mid = s[0], s[-1], s[1:-1]
    if not (first.isalpha() and last.isalpha() and mid.isdigit()):
        return None
    return int(mid), first.upper(), last.upper()


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
    ref_sequence: Optional[str] = None,
    metric_id: Optional[str] = None,
    evaluators: str = "",
) -> None:
    # ---------- data ----------
    df_all = all_df.copy()
    df1 = df_all[df_all["round"] == 1].copy()
    if df1.empty:
        raise RuntimeError(f"{job_name}: no round-1 variants to plot")

    y1, y_label = _series_for_metric(df1, metric_id)
    y_label = _pretty_metric(metric_id, y_label)
    df1 = df1.assign(_y=y1).dropna(subset=["_y"])
    df1["position"] = df1["modifications"].apply(_extract_position)

    stats = (
        df1.groupby("position")["_y"]
        .agg(mean="mean", sd="std")
        .reset_index()
        .sort_values("position")
    )

    y_all, _ = _series_for_metric(df_all, metric_id)
    df_all = df_all.assign(_y=y_all).dropna(subset=["_y"])

    # reference sequence (prefer explicit arg, else seed)
    ref_seq = (ref_sequence or "").strip().upper()
    if not ref_seq:
        seed = next(
            (
                r
                for r in df_all.to_dict("records")
                if int(r.get("round", 0)) == 1
                and isinstance(r.get("modifications"), list)
                and len(r["modifications"]) == 0
            ),
            None,
        )
        if seed and seed.get("sequence"):
            ref_seq = str(seed["sequence"]).upper()
    if not ref_seq:
        raise RuntimeError(f"{job_name}: reference sequence unavailable.")

    seed_row = next(
        (
            r
            for r in df_all.to_dict("records")
            if int(r.get("round", 0)) == 1
            and isinstance(r.get("modifications"), list)
            and len(r["modifications"]) == 0
        ),
        None,
    )
    ref_value = float(seed_row["_y"]) if seed_row and pd.notna(seed_row["_y"]) else None

    # collect simple per-mutation records
    records: List[Dict] = []
    for _, row in df_all.iterrows():
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
    ncols = len(ref_seq)
    full_positions = list(range(1, ncols + 1))
    residues = sorted(set(dfm["to_res"]).union(set(list(ref_seq))))

    pivot = dfm.pivot_table(
        index="to_res", columns="position", values="y", aggfunc="mean"
    ).reindex(index=residues, columns=full_positions)

    if ref_value is not None:
        for pos in full_positions:
            ref_res = ref_seq[pos - 1]
            if ref_res in residues:
                pivot.loc[ref_res, pos] = ref_value

    mat = pivot.to_numpy(dtype=float)
    nrows = len(residues)

    # ---------- sizing & layout ----------
    cell = 0.28  # inches per heatmap cell to keep 1:1 squares
    hm_w = max(6.4, ncols * cell)
    hm_h = max(2.2, nrows * cell)
    top_h = 2.6
    ref_h = 0.34  # very thin middle strip

    fig_w = hm_w
    fig_h = top_h + ref_h + hm_h + 0.7

    # Use explicit GridSpec without constrained_layout; hspace=0 for zero gap
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[top_h, ref_h, hm_h],
        width_ratios=[1.0, 0.012],  # extra-narrow colorbar column
        hspace=0.0,  # ← no vertical space between rows
        wspace=0.05,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[:, 1])  # span ALL rows → full-height colorbar

    # ---------- TOP: scatter ----------
    base = _baseline_for_metric(metric_id or y_label)
    if base is not None:
        ax_top.axhline(base, color="0.85", linewidth=1.0, zorder=0)

    ax_top.set_axisbelow(True)
    ax_top.grid(True, color="0.90", linewidth=0.5, zorder=0)

    ax_top.scatter(
        df1["position"],
        df1["_y"],
        color="lightgray",
        alpha=0.3,
        s=10,
        edgecolors="none",
        zorder=1,
    )
    ax_top.errorbar(
        stats["position"],
        stats["mean"],
        yerr=stats["sd"].fillna(0),
        fmt="o-",
        color="gray",
        alpha=0.75,
        markersize=3.8,
        capsize=2,
        elinewidth=0.8,
        markeredgecolor="none",
        zorder=2,
    )

    # titles positioned high but inside figure (avoid clipping)
    ref_name = (
        df1["ref_name"].iloc[0] if "ref_name" in df1.columns and not df1.empty else ""
    )
    title = f"{job_name}{f' ({ref_name})' if ref_name else ''}"
    fig.suptitle(title, fontsize=12, y=0.99)
    if evaluators:
        fig.text(0.5, 0.965, evaluators, ha="center", va="top", fontsize=9, alpha=0.85)

    ax_top.set_ylabel(y_label, fontsize=11)
    ax_top.tick_params(axis="both", labelsize=10)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(0.5, ncols + 0.5)

    # ---------- MIDDLE: reference sequence strip ----------
    # fully hide ticks/labels/spines and remove internal padding
    ax_mid.set_ylim(0, 1)
    ax_mid.set_yticks([])
    ax_mid.margins(x=0, y=0)
    ax_mid.set_facecolor("none")  # transparent so it visually touches the heatmap
    ax_mid.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax_mid.xaxis.set_major_locator(NullLocator())
    ax_mid.xaxis.set_minor_locator(NullLocator())
    for sp in ax_mid.spines.values():
        sp.set_visible(False)

    fs_char = max(10, min(20, 42 - 0.16 * ncols))
    for i, ch in enumerate(ref_seq, start=1):
        ax_mid.text(
            i, 0.5, ch, ha="center", va="center", fontsize=fs_char, family="monospace"
        )

    # ---------- BOTTOM: heatmap ----------
    extent = (0.5, ncols + 0.5, -0.5, nrows - 0.5)
    im = ax_bot.imshow(
        mat, extent=extent, origin="lower", interpolation="nearest", aspect="equal"
    )

    xticks = (
        full_positions
        if ncols <= 24
        else list(np.linspace(1, ncols, num=12, dtype=int))
    )
    ax_bot.set_xticks(xticks)
    ax_bot.set_xticklabels([str(x) for x in xticks], fontsize=10)
    ax_bot.set_yticks(range(nrows))
    ax_bot.set_yticklabels(residues, fontsize=10)
    ax_bot.set_xlabel("Sequence position", fontsize=11)
    ax_bot.set_ylabel("Mutated residue", fontsize=11)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    # narrow, full-height colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(y_label, rotation=90, va="center", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ensure titles/ticks never clip
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
