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

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator
from matplotlib.transforms import blended_transform_factory
from numpy import ndarray

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

# AA token parser: must be prefixed with "aa " → "aa pos=.. wt=X alt=Y"
# (keeps NT tokens like "nt pos=.. wt=A alt=T" from being misclassified),
# or compact "K12D" (but *not* "A12T", which is ambiguous with NT and should stay NT).
_RE_AA_EDIT_STRUCT = re.compile(
    r"\baa\b.*?\bpos\s*=\s*(?P<pos>\d+)\b.*?\bwt\s*=\s*(?P<wt>[A-Z\*])\b.*?\balt\s*=\s*(?P<alt>[A-Z\*])\b",
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

# Biochemical classes (top→bottom order on the heatmap)
AA_CAT_ORDER: list[tuple[str, list[str]]] = [
    ("(+)", ["H", "K", "R"]),
    ("(-)", ["D", "E"]),
    ("Polar-neutral", ["C", "M", "N", "Q", "S", "T"]),
    ("Non-polar", ["A", "I", "L", "V"]),
    ("Aromatic", ["F", "W", "Y"]),
    ("Unique", ["G", "P"]),
    ("*", ["*"]),
]

_HEATMAP_P_LO = 0.05  # e.g., 0.05 for the 5th percentile
_HEATMAP_P_HI = 0.95  # e.g., 0.95 for the 95th percentile

_LOG = logging.getLogger("permuter.plot.position_scatter_and_heatmap")


def _order_residues_aa(
    residues: list[str],
) -> tuple[list[str], list[tuple[str, int, int]]]:
    """
    Return (ordered residues, spans), where spans are (category, start_idx, end_idx)
    rows that exist *in this dataset*. Contiguity is guaranteed by construction.
    """
    present = set(residues)
    ordered: list[str] = []
    spans: list[tuple[str, int, int]] = []
    for cat, group in AA_CAT_ORDER:
        g = [aa for aa in group if aa in present]
        if not g:
            continue
        start = len(ordered)
        ordered.extend(g)
        spans.append((cat, start, len(ordered) - 1))
    # Any unexpected letters (e.g., X, U) appear at the end
    extras = [r for r in residues if r not in set(ordered)]
    if extras:
        start = len(ordered)
        ordered.extend(sorted(extras))
        spans.append(("Other", start, len(ordered) - 1))
    return ordered, spans


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
    """
    Native path: a single scalar column named `permuter__metric__<id>`.
    """
    if not metric_id:
        raise RuntimeError(
            "metric_id must be provided to select a `permuter__metric__<id>` column"
        )
    col = f"permuter__metric__{metric_id}"
    if col not in df.columns:
        # User might have passed the pretty name ("llr") while column is long ("log_likelihood_ratio") or vice versa.
        # Fall back to scanning for suffix matches.
        cand = [
            c
            for c in df.columns
            if c.startswith("permuter__metric__")
            and c.split("__", 2)[-1] == str(metric_id)
        ]
        if len(cand) == 1:
            col = cand[0]
        else:
            raise RuntimeError(f"Metric column not found: {col}")
    s = df[col].astype("float64")
    return s, _pretty_metric(metric_id, str(metric_id))


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
    # LLR is v - ref → baseline at 0.0
    if "ratio" in lab or lab == "llr" or "llr" in lab:
        return 0.0
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
    # Compact AA notation — accept only if not trivially a nucleotide→nucleotide change
    if len(s) >= 3 and s[0].isalpha() and s[-1].isalpha() and s[1:-1].isdigit():
        frm, to = s[0].upper(), s[-1].upper()
        # If both ends are nucleotides, treat as NT (e.g., 'A12T')
        if frm in "ACGT" and to in "ACGT":
            return None
        return int(s[1:-1]), frm, to
    return None


def _as_token_seq(x: object) -> List[str]:
    """
    Normalize a 'modifications' cell into a list of strings without relying on truthiness.
    Accept list/tuple/ndarray; anything else → empty list.
    """
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, ndarray):
        # pyarrow can hand us numpy arrays for list columns
        return [str(t) for t in x.tolist()]
    return []


def _row_to_y(idx: float, nrows: int) -> float:
    """
    Map a matrix row index (0 = top) to a plot Y coordinate when imshow is
    drawn with origin='upper' and extent=(0.5, ncols+0.5, -0.5, nrows-0.5).
    The center of row i is at y = (nrows - 1 - i).
    """
    return (nrows - 1) - float(idx)


def _robust_vmin_vmax(
    mat: np.ndarray, baseline: Optional[float]
) -> Tuple[float, float]:
    """
    Percentile-based clipping (no two-slope normalization).

    • If baseline == 0.0 (e.g., LLR), use a symmetric scale around 0:
        vabs = max(|Q_lo|, |Q_hi|) with Q at p_lo and p_hi.
        Return (-vabs, +vabs).
    • Otherwise, use [Q_lo, Q_hi] directly.
    • Fallbacks keep the plot informative when data are degenerate.

    Tune p_lo / p_hi via the module-level _HEATMAP_P_LO / _HEATMAP_P_HI.
    """
    # Validate / clamp percentiles defensively
    p_lo = float(min(max(_HEATMAP_P_LO, 0.0 + 1e-9), 0.5))
    p_hi = float(max(min(_HEATMAP_P_HI, 1.0 - 1e-9), 0.5))
    if not (0.0 < p_lo < p_hi < 1.0):
        raise ValueError(f"Invalid percentiles: p_lo={p_lo}, p_hi={p_hi}")

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return (-1.0, 1.0)

    q_lo = float(np.nanquantile(finite, p_lo))
    q_hi = float(np.nanquantile(finite, p_hi))

    if baseline == 0.0:
        # Symmetric color scale around 0 so 'X' (reference) is neutral white.
        vabs = max(abs(q_lo), abs(q_hi))
        if not np.isfinite(vabs) or vabs == 0.0:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            vabs = max(abs(lo), abs(hi), 1e-9)
        return (-vabs, vabs)

    # Non-zero baseline: asymmetric but still percentile-driven.
    if q_hi <= q_lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if hi == lo:
            # last-resort minimal span to avoid a flat image
            return (lo - 1e-9, hi + 1e-9)
        return (lo, hi)

    return (q_lo, q_hi)


def _has_aa_signals(df: pd.DataFrame) -> bool:
    # Native schema first
    if {"permuter__aa_pos", "permuter__aa_alt"}.issubset(df.columns):
        return True
    mods = df.get("permuter__modifications")
    if isinstance(mods, pd.Series):
        for m in mods.dropna():
            for t in _as_token_seq(m):
                if _parse_any_aa_edit(t):
                    return True
            # keep scanning
            if False:
                return True
    return False


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
    ref_sequence: Optional[str] = None,
    ref_aa_sequence: Optional[str] = None,
    metric_id: Optional[str] = None,
    evaluators: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    font_scale: Optional[float] = None,
    ref_strip_every: Optional[int] = None,
) -> None:
    # ---------- data ----------
    df_all = all_df.copy()
    df1 = df_all[df_all["permuter__round"] == 1].copy()
    if df1.empty:
        raise RuntimeError(f"{job_name}: no round-1 variants to plot")

    y1, y_label = _series_for_metric(df1, metric_id)
    y_label = _pretty_metric(metric_id, y_label)
    # Baseline for this metric (e.g., 0 for LLR). Compute ONCE and reuse.
    base = _baseline_for_metric(metric_id or y_label)
    df1 = df1.assign(_y=y1).dropna(subset=["_y"])

    # Decide NT vs AA mode
    aa_mode = _has_aa_signals(df_all)

    # Position extractor per mode
    def _row_pos_aa(row):
        if "permuter__aa_pos" in row and pd.notna(row["permuter__aa_pos"]):
            try:
                return int(row["permuter__aa_pos"])
            except Exception:
                pass
        mods = row.get("permuter__modifications", [])
        if isinstance(mods, list) and mods:
            for tok in mods:
                p = _parse_any_aa_edit(tok)
                if p:
                    return int(p[0])
        return 0

    def _row_pos_nt(row):
        if "permuter__nt_pos" in row and pd.notna(row["permuter__nt_pos"]):
            try:
                return int(row["permuter__nt_pos"])
            except Exception:
                pass
        mods = row.get("permuter__modifications", [])
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

    # reference sequences: prefer protein if provided; else translate DNA
    ref_dna = (ref_sequence or "").strip().upper()
    if not ref_dna:
        seed = next(
            (
                r
                for r in df_all.to_dict("records")
                if int(r.get("permuter__round", 0)) == 1
                and isinstance(r.get("permuter__modifications"), list)
                and len(r["permuter__modifications"]) == 0
            ),
            None,
        )
        if seed and seed.get("sequence"):
            ref_dna = str(seed["sequence"]).upper()
    if aa_mode:
        # 1) Strongest: user-provided REF_AA.fa
        if ref_aa_sequence and ref_aa_sequence.strip():
            ref_strip = ref_aa_sequence.strip().upper()
        # 2) Else derive from DNA
        elif ref_dna:
            ref_strip = _translate_dna(ref_dna)
        else:
            raise RuntimeError(f"{job_name}: no reference DNA or protein available.")
    else:
        if not ref_dna:
            raise RuntimeError(f"{job_name}: reference DNA unavailable.")
        ref_strip = ref_dna

    # If we’re in AA mode, verify positions fit within the reference protein.
    if aa_mode and "permuter__aa_pos" in df_all.columns:
        max_pos = int(pd.to_numeric(df_all["permuter__aa_pos"], errors="coerce").max())
        if max_pos > len(ref_strip):
            raise RuntimeError(
                f"{job_name}: aa positions (max={max_pos}) exceed reference protein length ({len(ref_strip)}). "
                "Provide an authoritative protein via (a) job.input.aa_col in your job YAML, or (b) a "
                "REF_AA.fa sidecar in the dataset directory."
            )

    # Try to find a baseline score for the reference
    seed_row = next(
        (
            r
            for r in df_all.to_dict("records")
            if int(r.get("permuter__round", 0)) == 1
            and isinstance(r.get("permuter__modifications"), list)
            and len(r["permuter__modifications"]) == 0
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
                "permuter__aa_pos" in row
                and pd.notna(row["permuter__aa_pos"])
                and "permuter__aa_alt" in row
                and pd.notna(row["permuter__aa_alt"])
            ):
                try:
                    records.append(
                        {
                            "position": int(row["permuter__aa_pos"]),
                            "to_res": str(row["permuter__aa_alt"]).upper(),
                            "y": yv,
                        }
                    )
                except Exception:
                    pass
            # Also parse tokens
            for token in _as_token_seq(row.get("permuter__modifications", [])):
                parsed = _parse_any_aa_edit(token)
                if parsed is None:
                    continue
                pos, _from, to = parsed
                records.append({"position": pos, "to_res": to, "y": yv})
        else:
            # NT mode: structured columns
            if (
                "permuter__nt_pos" in row
                and pd.notna(row["permuter__nt_pos"])
                and "permuter__nt_alt" in row
                and pd.notna(row["permuter__nt_alt"])
            ):
                try:
                    records.append(
                        {
                            "position": int(row["permuter__nt_pos"]),
                            "to_res": str(row["permuter__nt_alt"]).upper(),
                            "y": yv,
                        }
                    )
                except Exception:
                    pass
            # Parse tokens
            mods = row.get("permuter__modifications", [])
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
    spans: list[tuple[str, int, int]] = []
    if aa_mode:
        residues, spans = _order_residues_aa(residues)

    pivot = dfm.pivot_table(
        index="to_res", columns="position", values="y", aggfunc="mean"
    ).reindex(index=residues, columns=full_positions)

    if ref_value is not None:
        for pos in full_positions:
            ref_res = ref_strip[pos - 1]
            if ref_res in residues:
                pivot.loc[ref_res, pos] = ref_value
    else:
        # If we know the metric baseline, paint the reference cells with it so
        # the 'X' sits on a neutral (white) square for LLR.
        if base is not None:
            for pos in full_positions:
                ref_res = ref_strip[pos - 1]
                if ref_res in residues:
                    pivot.loc[ref_res, pos] = float(base)

    mat = pivot.to_numpy(dtype=float)
    nrows = len(residues)
    # Assert invariants early (fail fast)
    if mat.shape != (nrows, ncols):
        raise RuntimeError(
            f"Heatmap shape mismatch: got {mat.shape}, expected {(nrows, ncols)}. "
            "This might be a position/residue indexing bug upstream."
        )
    if df1["position"].min() < 1 or df1["position"].max() > ncols:
        raise RuntimeError(
            f"Positions out of bounds: min={int(df1['position'].min())}, "
            f"max={int(df1['position'].max())}, protein length={ncols}. "
            "If you scanned a region, ensure AA positions are absolute, not window-relative."
        )

    # ---------- sizing & layout (square cells by pixels) ----------
    fs = float(font_scale) if font_scale else 1.0
    # Emphasize titles/labels without cramming ticks
    LABEL_BOOST = 1.60  # axis & colorbar labels
    TITLE_BOOST = 1.85  # main title
    SUBTITLE_BOOST = 1.45
    TARGET_DPI = 300
    CELL_PX = (
        8  # width & height of each heatmap cell (adjustable; keeps files manageable)
    )
    hm_w_in = max(6.0, (ncols * CELL_PX) / TARGET_DPI)
    hm_h_in = max(1.6, (nrows * CELL_PX) / TARGET_DPI)
    top_h_in = 1.8
    ref_h_in = 0.40 if aa_mode else 0.30
    cbar_w_in = 0.06

    fig_w = hm_w_in + cbar_w_in
    fig_h = top_h_in + ref_h_in + hm_h_in + 0.40
    if figsize:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=TARGET_DPI)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[top_h_in, ref_h_in, hm_h_in],
        width_ratios=[1.0, cbar_w_in / hm_w_in],
        hspace=0.0,  # no vertical gap between top/mid/bottom
        wspace=0.02,  # tighter gap to right colorbar
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
        df1["permuter__ref"].iloc[0]
        if "permuter__ref" in df1.columns and not df1.empty
        else ""
    )

    title = f"{job_name}{f' ({ref_name})' if ref_name else ''}"
    fig.suptitle(title, fontsize=int(round(12 * fs * TITLE_BOOST)), y=0.94)

    if evaluators:
        fig.text(
            0.5,
            0.90,
            evaluators,
            ha="center",
            va="top",
            fontsize=int(round(9.5 * fs * SUBTITLE_BOOST)),
            alpha=0.80,
        )

    ax_top.set_ylabel(y_label, fontsize=int(round(11 * fs * LABEL_BOOST)))
    ax_top.tick_params(axis="both", labelsize=int(round(10 * fs)))
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

    # To avoid a dense black band for long proteins, draw at most ~200 letters by default.
    draw_every = (
        int(ref_strip_every) if ref_strip_every else max(1, int(np.ceil(ncols / 200)))
    )
    fs_char_pt = max(6, min(12, int(round(8 * fs))))
    for i, ch in enumerate(ref_strip, start=1):
        if (i - 1) % draw_every != 0:
            continue
        ax_mid.text(
            i,
            0.5,
            ch,
            ha="center",
            va="center",
            fontsize=fs_char_pt,
            family="monospace",
            alpha=0.75,
        )
    # Clarify the down‑sampling for the reader
    if draw_every > 1:
        ax_mid.text(
            0.995,
            0.05,
            f"(every {draw_every}ᵗʰ AA)",
            transform=ax_mid.transAxes,
            ha="right",
            va="bottom",
            fontsize=int(round(8 * fs)),
            alpha=0.6,
        )

    # ---------- BOTTOM: heatmap ----------
    # Diverging colormap for baseline==0; otherwise default
    cmap = plt.get_cmap("RdBu_r") if base == 0.0 else None
    vmin = vmax = None
    if np.isfinite(mat).any():
        vmin, vmax = _robust_vmin_vmax(mat, baseline=base)
        _LOG.info(
            "heatmap scale: vmin=%.4f, vmax=%.4f (quartile clip; baseline=%s)",
            vmin,
            vmax,
            str(base),
        )
    # Draw heatmap with explicit 1-based extent so col centers are at x=1..ncols.
    imshow_kwargs = dict(
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        extent=(0.5, ncols + 0.5, -0.5, nrows - 0.5),
    )
    im = ax_bot.imshow(mat, vmin=vmin, vmax=vmax, **imshow_kwargs)

    # Adaptive, non-cramming X ticks (aim for 12–24 labels depending on width)
    hm_w_px = ncols * CELL_PX
    max_labels = max(10, min(100, hm_w_px // 20))
    step = max(1, int(np.ceil(ncols / max_labels)))

    # round step to a "nice" 1/2/5×10^k increment
    def _nice(k: int) -> int:
        base = 10 ** int(np.floor(np.log10(k))) if k > 0 else 1
        for m in (1, 2, 5, 10):
            if m * base >= k:
                return m * base
        return k

    step = _nice(step)
    xticks = list(range(1, ncols + 1, step))
    if xticks[-1] != ncols:
        xticks.append(ncols)
    ax_bot.set_xticks(xticks)
    # indices (0=top) to y-coordinates with origin='upper'.
    ytick_pos = [_row_to_y(i, nrows) for i in range(nrows)]
    ax_bot.set_yticks(ytick_pos)
    ax_bot.set_xticklabels([str(x) for x in xticks], fontsize=int(round(10 * fs)))
    ax_bot.set_yticklabels(residues, fontsize=int(round(10 * fs)))
    ax_bot.set_xlabel(
        "Protein position" if aa_mode else "Sequence position",
        fontsize=int(round(11 * fs * LABEL_BOOST)),
    )
    ax_bot.set_ylabel(
        "Mutated amino acid" if aa_mode else "Mutated residue",
        fontsize=int(round(11 * fs * LABEL_BOOST)),
    )
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    # Move main Y label further left to make room for category tags (requested)
    ax_bot.set_ylabel(
        "Mutated amino acid" if aa_mode else "Mutated residue",
        fontsize=int(round(11 * fs * LABEL_BOOST)),
        labelpad=10,
    )
    ax_bot.yaxis.set_label_coords(-0.030, 0.5)

    # Category separators and side legend (AA mode only)
    if aa_mode and spans:
        # thin separators between classes (mapped to display y)
        for _, _s, _e in spans[:-1]:
            ax_bot.axhline(
                _row_to_y(_e, nrows) - 0.5, color="0.85", linewidth=0.8, zorder=3
            )
        # left-side category labels at group center (mapped)
        trans = blended_transform_factory(ax_bot.transAxes, ax_bot.transData)
        for label, s, e in spans:
            y_mid = _row_to_y((s + e) / 2.0, nrows)
            ax_bot.text(
                -0.009,
                y_mid,
                label,
                transform=trans,
                ha="right",
                va="center",
                fontsize=int(round(9.5 * fs)),
                color="0.35",
                clip_on=False,
                rotation=0,
            )

    # Mark reference squares with an “X” for readability
    try:
        res_to_row = {r: i for i, r in enumerate(residues)}
        for x_pos in range(1, ncols + 1):
            ref_res = ref_strip[x_pos - 1]
            y_row = res_to_row.get(ref_res)
            if y_row is None:
                continue
            # cell center at 1-based x (matches extent)
            x = x_pos
            y = _row_to_y(y_row, nrows)
            # two diagonals across the cell bounds
            ax_bot.plot(
                [x - 0.5, x + 0.5],
                [y - 0.5, y + 0.5],
                linewidth=0.4,
                color="k",
                alpha=0.2,
                zorder=4,
            )
            ax_bot.plot(
                [x - 0.5, x + 0.5],
                [y + 0.5, y - 0.5],
                linewidth=0.4,
                color="k",
                alpha=0.2,
                zorder=4,
            )
    except Exception as _e:
        _LOG.debug("reference 'X' overlay skipped: %s", _e)

    cbar = fig.colorbar(im, cax=cax)

    cbar.ax.set_ylabel(
        y_label, rotation=90, va="center", fontsize=int(round(11 * fs * LABEL_BOOST))
    )
    cbar.ax.tick_params(labelsize=int(round(9 * fs)))

    # ensure titles/ticks never clip
    fig.savefig(output_path, dpi=TARGET_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
