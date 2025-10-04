"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/position_scatter_and_heatmap.py

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

AA-aware:
  • If AA signals are present (aa_pos/aa_alt or 'aa pos=.. wt=X alt=Y' tokens),
    the plot switches to amino-acid mode:
      - X axis indexes amino-acid positions.
      - Middle strip shows the protein sequence (translated from ref DNA).
      - Heatmap rows are amino acids.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
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

# robust regex for the "nt pos=.. wt=.. alt=.." style
_RE_EDIT_STRUCT = re.compile(
    r"\bpos\s*=\s*(?P<pos>\d+)\b.*?\bwt\s*=\s*(?P<wt>[ACGTN])\b.*?\balt\s*=\s*(?P<alt>[ACGTN])\b",
    flags=re.IGNORECASE,
)

# AA token parser: "aa pos=.. wt=X alt=Y" or compact "K12D"
_RE_AA_EDIT_STRUCT = re.compile(
    r"\bpos\s*=\s*(?P<pos>\d+)\b.*?\bwt\s*=\s*(?P<wt>[A-Z\*])\b.*?\balt\s*=\s*(?P<alt>[A-Z\*])\b",
    flags=re.IGNORECASE,
)

# genetic code for translating DNA → protein (stops → '*')
_GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def _translate_dna(dna: str) -> str:
    dna = (dna or "").upper()
    if len(dna) < 3:
        return ""
    usable = len(dna) - (len(dna) % 3)
    codons = [dna[i : i + 3] for i in range(0, usable, 3)]
    return "".join(_GENETIC_CODE.get(c, "X") for c in codons)


def _pretty_metric(mid: str | None, fallback: str) -> str:
    return _METRIC_LABELS.get(str(mid).strip(), str(mid).strip()) if mid else fallback


def _series_for_metric(
    df: pd.DataFrame, metric_id: Optional[str]
) -> Tuple[pd.Series, str]:
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


def _extract_position_from_token(token: str) -> int:
    """Best-effort position extraction from any NT token text."""
    m = _RE_EDIT_STRUCT.search(str(token))
    if m:
        return int(m.group("pos"))
    # fallback: pull first integer run from token
    digits = "".join(ch for ch in str(token) if ch.isdigit())
    return int(digits) if digits else 0


def _baseline_for_metric(label: str) -> Optional[float]:
    lab = (label or "").lower()
    if "ratio" in lab or lab == "llr" or "llr" in lab:
        return 1.0
    if "likelihood" in lab or lab == "ll":
        return 0.0
    return None


def _parse_any_nt_edit(token: str) -> Optional[tuple[int, str, str]]:
    """
    Accept either:
      • 'A12T'
      • 'nt pos=12 wt=A alt=T' (order/spacing flexible)
    Returns (pos, from_nt, to_nt) or None.
    """
    s = str(token).strip()

    # structured style
    m = _RE_EDIT_STRUCT.search(s)
    if m:
        return int(m.group("pos")), m.group("wt").upper(), m.group("alt").upper()

    # compact A12T style
    if len(s) >= 3 and s[0].isalpha() and s[-1].isalpha() and s[1:-1].isdigit():
        return int(s[1:-1]), s[0].upper(), s[-1].upper()

    return None


def _parse_any_aa_edit(token: str) -> Optional[tuple[int, str, str]]:
    """
    Accept either:
      • 'aa pos=12 wt=K alt=D'
      • 'K12D'
    Returns (pos, from_aa, to_aa) or None.
    """
    s = str(token).strip()
    m = _RE_AA_EDIT_STRUCT.search(s)
    if m:
        return int(m.group("pos")), m.group("wt").upper(), m.group("alt").upper()
    if len(s) >= 3 and s[0].isalpha() and s[-1].isalpha() and s[1:-1].isdigit():
        return int(s[1:-1]), s[0].upper(), s[-1].upper()
    return None


def _has_aa_signals(df: pd.DataFrame) -> bool:
    if {"aa_pos", "aa_alt"}.issubset(df.columns):
        return True
    mods = df.get("modifications")
    if isinstance(mods, pd.Series):
        for m in mods.dropna():
            if any(_parse_any_aa_edit(t) for t in (m or [])):
                return True
    return False


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

    # Decide NT vs AA mode
    aa_mode = _has_aa_signals(df_all)

    # Position extractor per mode
    def _row_pos_aa(row):
        if "aa_pos" in row and pd.notna(row["aa_pos"]):
            try:
                return int(row["aa_pos"])
            except Exception:
                pass
        mods = row.get("modifications", [])
        if isinstance(mods, list) and mods:
            for tok in mods:
                p = _parse_any_aa_edit(tok)
                if p:
                    return int(p[0])
        return 0

    def _row_pos_nt(row):
        if "nt_pos" in row and pd.notna(row["nt_pos"]):
            try:
                return int(row["nt_pos"])
            except Exception:
                pass
        mods = row.get("modifications", [])
        if isinstance(mods, list) and mods:
            return _extract_position_from_token(mods[0])
        return 0

    # choose the position extractor
    df1["position"] = df1.apply(_row_pos_aa if aa_mode else _row_pos_nt, axis=1)
    df1 = df1[df1["position"] > 0]  # drop seeds/invalids

    if df1.empty:
        raise RuntimeError(
            f"{job_name}: no round-1 {'AA' if aa_mode else 'single-nt'} edits to plot"
        )

    stats = (
        df1.groupby("position")["_y"]
        .agg(mean="mean", sd="std")
        .reset_index()
        .sort_values("position")
    )

    y_all, _ = _series_for_metric(df_all, metric_id)
    df_all = df_all.assign(_y=y_all).dropna(subset=["_y"])

    # reference sequence (prefer explicit arg, else seed)
    ref_dna = (ref_sequence or "").strip().upper()
    if not ref_dna:
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
            ref_dna = str(seed["sequence"]).upper()
    if not ref_dna:
        raise RuntimeError(f"{job_name}: reference sequence unavailable.")

    # Build the strip to display in the middle panel
    ref_strip = _translate_dna(ref_dna) if aa_mode else ref_dna

    # Try to find a baseline score for the reference
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
    ref_value = (
        float(seed_row["_y"])
        if seed_row and pd.notna(seed_row.get("_y", np.nan))
        else None
    )

    # ---------- collect per-mutation records ----------
    records: List[Dict] = []

    for _, row in df_all.iterrows():
        yv = float(row["_y"])
        if aa_mode:
            # Prefer structured AA columns
            if (
                "aa_pos" in row
                and pd.notna(row["aa_pos"])
                and "aa_alt" in row
                and pd.notna(row["aa_alt"])
            ):
                try:
                    records.append(
                        {
                            "position": int(row["aa_pos"]),
                            "to_res": str(row["aa_alt"]).upper(),
                            "y": yv,
                        }
                    )
                except Exception:
                    pass
            # Also parse tokens
            mods = row.get("modifications", [])
            if isinstance(mods, list):
                for token in mods:
                    parsed = _parse_any_aa_edit(token)
                    if parsed is None:
                        continue
                    pos, _from, to = parsed
                    records.append({"position": pos, "to_res": to, "y": yv})
        else:
            # NT mode: structured columns
            if (
                "nt_pos" in row
                and pd.notna(row["nt_pos"])
                and "nt_alt" in row
                and pd.notna(row["nt_alt"])
            ):
                try:
                    records.append(
                        {
                            "position": int(row["nt_pos"]),
                            "to_res": str(row["nt_alt"]).upper(),
                            "y": yv,
                        }
                    )
                except Exception:
                    pass
            # Parse tokens
            mods = row.get("modifications", [])
            if isinstance(mods, list):
                for token in mods:
                    parsed = _parse_any_nt_edit(token)
                    if parsed is None:
                        continue
                    pos, _from, to = parsed
                    records.append({"position": pos, "to_res": to, "y": yv})

    if not records:
        raise RuntimeError(
            f"{job_name}: no {'amino-acid' if aa_mode else 'single-nucleotide'} edits recognized. "
            f"Expected {'K12D or aa pos=.. wt=X alt=Y' if aa_mode else 'A12T or nt pos=.. wt=X alt=Y'}."
        )

    dfm = pd.DataFrame(records)
    # De-duplicate identical (pos, to_res, y) rows that can arise when both
    # structured columns and tokens describe the same edit.
    dfm = dfm.drop_duplicates(subset=["position", "to_res", "y"])

    ncols = len(ref_strip)
    full_positions = list(range(1, ncols + 1))
    residues = sorted(set(dfm["to_res"]).union(set(list(ref_strip))))

    pivot = dfm.pivot_table(
        index="to_res", columns="position", values="y", aggfunc="mean"
    ).reindex(index=residues, columns=full_positions)

    if ref_value is not None:
        for pos in full_positions:
            ref_res = ref_strip[pos - 1]
            if ref_res in residues:
                pivot.loc[ref_res, pos] = ref_value

    mat = pivot.to_numpy(dtype=float)
    nrows = len(residues)

    # ---------- sizing & layout ----------
    cell = 0.28
    hm_w = max(6.4, ncols * cell)
    hm_h = max(2.2, nrows * cell)
    top_h = 2.6
    ref_h = 0.34

    fig_w = hm_w
    fig_h = top_h + ref_h + hm_h + 0.7

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[top_h, ref_h, hm_h],
        width_ratios=[1.0, 0.012],
        hspace=0.0,
        wspace=0.05,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[:, 1])

    # ---------- TOP: scatter ----------
    base = _baseline_for_metric(metric_id or y_label)
    if base is not None:
        ax_top.axhline(base, linewidth=1.0, color="0.85", zorder=0)

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
    ax_mid.set_ylim(0, 1)
    ax_mid.set_yticks([])
    ax_mid.margins(x=0, y=0)
    ax_mid.set_facecolor("none")
    ax_mid.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax_mid.xaxis.set_major_locator(NullLocator())
    ax_mid.xaxis.set_minor_locator(NullLocator())
    for sp in ax_mid.spines.values():
        sp.set_visible(False)

    fs_char = max(10, min(20, 42 - 0.16 * ncols))
    for i, ch in enumerate(ref_strip, start=1):
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
    ax_bot.set_xlabel(
        "Protein position" if aa_mode else "Sequence position", fontsize=11
    )
    ax_bot.set_ylabel(
        "Mutated amino acid" if aa_mode else "Mutated residue", fontsize=11
    )
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(y_label, rotation=90, va="center", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ensure titles/ticks never clip
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
