"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/aa_category_effects.py

Category-level summary for amino-acid substitutions.

Top: mean Δ per biochemical class (IQR whiskers) + fraction |Δ| ≥ thresh.
Bottom: mean Δ per residue (grouped by class order).

Module Author(s): Eric J. South
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

# Reuse the same category order as the heatmap
AA_CAT_ORDER: list[tuple[str, list[str]]] = [
    ("(+)", ["H", "K", "R"]),
    ("(-)", ["D", "E"]),
    ("Polar-neutral", ["C", "M", "N", "Q", "S", "T"]),
    ("Non-polar", ["A", "I", "L", "V"]),
    ("Aromatic", ["F", "W", "Y"]),
    ("Unique", ["G", "P"]),
    ("*", ["*"]),
]


_RE_AA_EDIT_STRUCT = re.compile(
    r"\baa\b.*?\bpos\s*=\s*(?P<pos>\d+)\b.*?\bwt\s*=\s*(?P<wt>[A-Z\*])\b.*?\balt\s*=\s*(?P<alt>[A-Z\*])\b",
    flags=re.IGNORECASE,
)


def _parse_any_aa_edit(token: str) -> Optional[tuple[int, str, str]]:
    s = str(token).strip()
    m = _RE_AA_EDIT_STRUCT.search(s)
    if m:
        return int(m.group("pos")), m.group("wt").upper(), m.group("alt").upper()
    # Compact AA notation (not all-nucleotide)
    if len(s) >= 3 and s[0].isalpha() and s[-1].isalpha() and s[1:-1].isdigit():
        frm, to = s[0].upper(), s[-1].upper()
        if frm in "ACGT" and to in "ACGT":
            return None
        return int(s[1:-1]), frm, to
    return None


def _has_aa_signals(df: pd.DataFrame) -> bool:
    if {"permuter__aa_pos", "permuter__aa_alt"}.issubset(df.columns):
        return True
    mods = df.get("permuter__modifications")
    if isinstance(mods, pd.Series):
        for m in mods.dropna():
            for t in list(m) if isinstance(m, (list, tuple)) else []:
                if _parse_any_aa_edit(str(t)):
                    return True
    return False


def _series_for_metric(
    df: pd.DataFrame, metric_id: Optional[str]
) -> Tuple[pd.Series, str]:
    if not metric_id:
        raise RuntimeError("metric_id must be provided for aa_category_effects")
    col = f"permuter__metric__{metric_id}"
    if col not in df.columns:
        raise RuntimeError(f"Metric column not found: {col}")
    return df[col].astype("float64"), str(metric_id)


_LOG = logging.getLogger("permuter.plot.aa_category_effects")


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
    ref_sequence: Optional[str] = None,  # unused
    metric_id: Optional[str] = None,
    evaluators: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    font_scale: Optional[float] = None,
    large_delta_threshold: float = 1.0,  # absolute Δ threshold; for LLR this is in log units
) -> None:
    if not _has_aa_signals(all_df):
        raise RuntimeError(
            "aa_category_effects requires amino-acid edits (K12D or aa pos=..)."
        )

    fs = float(font_scale) if font_scale else 1.0
    LABEL_BOOST = 1.60
    TITLE_BOOST = 1.85
    SUBTITLE_BOOST = 1.45

    df = all_df.copy()
    y, y_label = _series_for_metric(df, metric_id)
    df = df.assign(_y=y).dropna(subset=["_y"])

    # Reference (round-1 seed with no modifications) for delta; if absent, Δ := y
    seed_row = next(
        (
            r
            for r in df.to_dict("records")
            if int(r.get("permuter__round", 0)) == 1
            and isinstance(r.get("permuter__modifications"), list)
            and len(r["permuter__modifications"]) == 0
        ),
        None,
    )
    ref_value = (
        float(seed_row["_y"])
        if seed_row and pd.notna(seed_row.get("_y", np.nan))
        else 0.0
    )

    # Extract AA edits (WT, POS, ALT) + Δ
    # This drives both the plot (ALT-only aggregations) and the residue-specific summary (WT+POS+ALT).
    records: List[Dict] = []
    for _, row in df.iterrows():
        # structured fields first
        if (
            pd.notna(row.get("permuter__aa_pos", np.nan))
            and pd.notna(row.get("permuter__aa_wt", np.nan))
            and pd.notna(row.get("permuter__aa_alt", np.nan))
        ):
            records.append(
                {
                    "pos": int(row["permuter__aa_pos"]),
                    "wt": str(row["permuter__aa_wt"]).upper(),
                    "to_res": str(row["permuter__aa_alt"]).upper(),
                    "delta": float(row["_y"]) - ref_value,
                }
            )
        # parse tokens too
        mods = row.get("permuter__modifications", [])
        for tok in mods if isinstance(mods, (list, tuple)) else []:
            p = _parse_any_aa_edit(str(tok))
            if p:
                pos, wt, to = p
                records.append(
                    {
                        "pos": int(pos),
                        "wt": wt,
                        "to_res": to,
                        "delta": float(row["_y"]) - ref_value,
                    }
                )
    if not records:
        raise RuntimeError("No amino-acid edits recognized for aa_category_effects.")

    dfm = pd.DataFrame(records).drop_duplicates()
    # Map residue → class
    res2cat: Dict[str, str] = {}
    for cat, group in AA_CAT_ORDER:
        for aa in group:
            res2cat[aa] = cat
    dfm["category"] = dfm["to_res"].map(lambda r: res2cat.get(r, "Other"))

    # ---- Category-level summary -----------------
    cat_order = [c for c, grp in AA_CAT_ORDER if any(dfm["category"] == c)]
    if (dfm["category"] == "Other").any():
        cat_order += ["Other"]

    g = dfm.groupby("category")["delta"]
    cs = g.agg(
        mean="mean",
        q1=lambda s: s.quantile(0.25),
        median="median",
        q3=lambda s: s.quantile(0.75),
        count="count",
    )
    # fraction of large effects by group (explicit to avoid shape pitfalls)
    frac = (
        dfm.assign(_big=dfm["delta"].abs() >= float(large_delta_threshold))
        .groupby("category")["_big"]
        .mean()
    )
    cs["frac_big"] = frac
    cs = cs.loc[cat_order]

    # Per-residue summary (for the bottom panel)
    rs = (
        dfm.groupby("to_res")["delta"]
        .agg(
            mean="mean",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            count="count",
        )
        .reset_index()
    )
    # keep class order, inner residue order as in AA_CAT_ORDER
    ordered_letters = [
        aa for _, grp in AA_CAT_ORDER for aa in grp if (rs["to_res"] == aa).any()
    ]
    rs = rs.set_index("to_res").loc[ordered_letters].reset_index()
    rs["category"] = rs["to_res"].map(lambda r: res2cat.get(r, "Other"))

    # ---- figure ----
    # ---- figure ----
    # Square by default; if a non-square is passed, collapse to square (min side).
    if figsize:
        w, h = float(figsize[0]), float(figsize[1])
        side = min(max(5.0, w), max(5.0, h))  # keep it reasonable
        fig_w, fig_h = side, side
    else:
        fig_w, fig_h = (6.0, 6.0)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h), gridspec_kw=dict(height_ratios=[1.1, 1.0])
    )

    # Title & subtitle
    ref_name = (
        df["permuter__ref"].iloc[0]
        if "permuter__ref" in df.columns and not df.empty
        else ""
    )
    fig.suptitle(
        f"{job_name}{f' ({ref_name})' if ref_name else ''}",
        fontsize=int(round(12 * fs * TITLE_BOOST)),
        y=0.995,
    )
    if evaluators:
        fig.text(
            0.5,
            0.965,
            evaluators,
            ha="center",
            va="top",
            fontsize=int(round(9.5 * fs * SUBTITLE_BOOST)),
            alpha=0.80,
        )

    # Top: category bars
    x = np.arange(len(cs))
    ax1.bar(x, cs["mean"].values, width=0.62, alpha=0.9)
    # IQR whiskers
    ax1.vlines(x, cs["q1"], cs["q3"], linewidth=3.0, color="k", alpha=0.35)
    # Fraction of large deltas (as % above bars)
    for i, (m, f, n) in enumerate(zip(cs["mean"], cs["frac_big"], cs["count"])):
        ax1.text(
            i,
            m,
            f"{f*100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=int(round(9.5 * fs)),
            alpha=0.7,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(cs.index.tolist(), rotation=0, fontsize=int(round(10 * fs)))
    ax1.set_ylabel(
        f"Δ {y_label} (mean; whiskers=IQR)", fontsize=int(round(11 * fs * LABEL_BOOST))
    )
    ax1.grid(axis="y", color="0.9")

    # Bottom: per-residue bars (grouped by class visually via light separators)
    x2 = np.arange(len(rs))
    ax2.bar(x2, rs["mean"].values, width=0.75, alpha=0.9)
    for i, (q1, q3) in enumerate(zip(rs["q1"], rs["q3"])):
        ax2.vlines(i, q1, q3, linewidth=2.5, color="k", alpha=0.35)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(rs["to_res"], fontsize=int(round(10 * fs)))
    ax2.set_ylabel(
        f"Δ {y_label} (mean; whiskers=IQR)", fontsize=int(round(11 * fs * LABEL_BOOST))
    )
    ax2.grid(axis="y", color="0.9")

    # vertical faint separators between classes
    pos = 0
    for cat, grp in AA_CAT_ORDER:
        letters = [aa for aa in grp if aa in rs["to_res"].values]
        if not letters:
            continue
        end = pos + len(letters) - 1
        if end < len(rs) - 1:
            ax2.axvline(end + 0.5, color="0.85", lw=0.8)
        # label the class above the group
        mid = (pos + end) / 2.0
        ax2.text(
            mid,
            ax2.get_ylim()[1],
            cat,
            ha="center",
            va="bottom",
            fontsize=int(round(9.0 * fs)),
            alpha=0.6,
        )
        pos = end + 1

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
