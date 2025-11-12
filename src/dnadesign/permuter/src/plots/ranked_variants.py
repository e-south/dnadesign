"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/ranked_variants.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray


def _series_for_metric(
    df: pd.DataFrame, metric_id: Optional[str]
) -> Tuple[pd.Series, str]:
    if not metric_id:
        raise RuntimeError("ranked_variants: metric_id is required")
    col = f"permuter__metric__{metric_id}"
    if col not in df.columns:
        raise RuntimeError(f"ranked_variants: metric column not found: {col}")
    return df[col].astype("float64"), str(metric_id)


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
    annotate_top_k: Optional[int] = None,
    summary_top_n: Optional[int] = None,
    xtick_every: Optional[int] = None,
    export_top_k: Optional[int] = None,
    dataset_dir: Optional[Path] = None,
) -> None:
    df = all_df.copy()
    y, y_label = _series_for_metric(df, metric_id)
    df = df.assign(_y=y).dropna(subset=["_y"]).copy()

    # Prefer round-2 variants for ranking (combinations), but fall back to all if absent
    if "permuter__round" in df.columns and (df["permuter__round"] == 2).any():
        df2 = df[df["permuter__round"] == 2].copy()
    else:
        df2 = df

    # Total eligible rows — used by export safeguards and labels
    N_all = int(len(df2))

    # Ensure mut count exists (fallback = number of tokens in 'modifications')
    def _count_mods(m: List[str] | object) -> int:
        if isinstance(m, (list, tuple)):
            return len(m)
        return 0
    if "permuter__mut_count" in df2.columns:
        mut_col = "permuter__mut_count"
    else:
        df2["permuter__mut_count"] = df2["permuter__modifications"].apply(_count_mods)
        mut_col = "permuter__mut_count"

    # Sort ASCENDING by metric (lower ranks to the left), then assign ranks 1..N
    df2 = df2.sort_values("_y", ascending=True, kind="mergesort").reset_index(drop=True)
    df2["rank"] = np.arange(1, len(df2) + 1)
    # We'll highlight the top (by metric DESC) but show ALL points faintly
    H = min(100, len(df2))  # highlight window (top variants)
    df_top = df2.sort_values("_y", ascending=False, kind="mergesort").head(H).copy()

    # Title scaffold
    fs = float(font_scale) if font_scale else 1.0
    # Square by default; honor user‑provided size if given
    if figsize:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])
    else:
        fig_w, fig_h = (6.0, 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.90", linewidth=0.7)

    # Discrete palette by mutation count (hue=k)
    mut_levels = sorted(df2[mut_col].fillna(-1).astype(int).unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_by = {lvl: cmap(i % 10) for i, lvl in enumerate(mut_levels)}

    # A) ALL variants, faint points
    for lvl in mut_levels:
        sub_all = df2[df2[mut_col].fillna(-1).astype(int) == lvl]
        if sub_all.empty:
            continue
        ax.scatter(
            sub_all["rank"],
            sub_all["_y"],
            s=16,
            alpha=0.30,                # slight alpha (requested)
            edgecolors="none",
            color=color_by[lvl],
            label=None,
            zorder=1.0,
        )

    # B) TOP (by metric DESC), highlighted overlay
    for lvl in mut_levels:
        sub_top = df_top[df_top[mut_col].fillna(-1).astype(int) == lvl]
        if sub_top.empty:
            continue
        ax.scatter(
            sub_top["rank"],
            sub_top["_y"],
            s=30,
            alpha=0.95,
            linewidths=0.6,
            edgecolors="black",
            color=color_by[lvl],
            zorder=2.0,
            # Give each present level a label once (legend built conditionally below)
            label=f"k={lvl}",
        )

    # Optional, sparse X ticks for readability on very wide plots
    if isinstance(xtick_every, int) and xtick_every > 0:
        ticks = list(range(1, len(df2) + 1, int(xtick_every)))
        if ticks[-1] != len(df2):
            ticks.append(len(df2))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], rotation=0)
        ax.tick_params(axis="x", labelsize=int(round(9 * (font_scale or 1.0))))
    else:
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.set_xlim(0.5, len(df2) + 0.5)

    ax.set_xlabel("Variant rank", fontsize=int(round(11 * fs)))
    ax.set_ylabel(y_label, fontsize=int(round(11 * fs)))
    ax.tick_params(axis="y", labelsize=int(round(10 * fs)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    labels = [str(lab) for lab in labels if lab and not str(lab).startswith("_")]
    if handles and labels:
        leg = ax.legend(
            title="Mutations (k)",
            fontsize=int(round(9 * fs)),
            title_fontsize=int(round(10 * fs)),
            frameon=False,
            loc="best",
        )
        try:
            leg.set_frame_on(False)
        except Exception:
            pass

    # Large red star: reference sequence score (from round‑1 seed)
    ref_value = None
    try:
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
        if seed_row is not None and pd.notna(seed_row.get("_y", np.nan)):
            ref_value = float(seed_row["_y"])
    except Exception:
        ref_value = None
    if ref_value is not None and np.isfinite(ref_value):
        ax.scatter([0.5], [ref_value], marker="*", s=180, color="red", alpha=0.90, zorder=3)
        ax.text(0.5, ref_value, "  REF", va="center", ha="left",
                fontsize=int(round(9 * fs)), color="red", alpha=0.85, zorder=3.1)

    # Annotate top‑K (by metric DESC) with the AA combo string
    ann_k = int(annotate_top_k) if (annotate_top_k is not None) else 5
    if ann_k > 0 and "permuter__aa_combo_str" in df2.columns:
        ann_df = df2.sort_values("_y", ascending=False, kind="mergesort").head(ann_k)
        ymin, ymax = float(df2["_y"].min()), float(df2["_y"].max())
        dy = 0.015 * (ymax - ymin) if ymax > ymin else 0.0
        for _, r in ann_df.iterrows():
            ax.text(
                float(r["rank"]) + 0.15,
                float(r["_y"]) + dy,
                str(r["permuter__aa_combo_str"]),
                fontsize=int(round(8.3 * fs)),
                alpha=0.90,
                rotation=0,
                zorder=3.2,
            )

    # --- In-plot summary (top left): counts by k, median(metric), and count in Top-N ---
    try:
        summary_n = int(summary_top_n) if summary_top_n is not None else 100
        med = df2.groupby(mut_col)["_y"].median().sort_index()
        cnt = df2.groupby(mut_col).size().sort_index()
        topN = df2.sort_values("_y", ascending=False, kind="mergesort").head(summary_n)
        cnt_top = topN.groupby(mut_col).size()
        lines = []
        for k in sorted(set(cnt.index).union(set(cnt_top.index))):
            n_all = int(cnt.get(k, 0))
            n_top = int(cnt_top.get(k, 0))
            m = float(med.get(k, np.nan))
            lines.append(f"k={k}: n={n_all:,}  median={m:+.3f}  in top {summary_n} → {n_top:,}")
        block = "\n".join(lines) if lines else "—"
        ax.text(0.01, 0.99, block, transform=ax.transAxes, ha="left", va="top",
                fontsize=int(round(8.8 * fs)),
                bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.85, pad=5.0),
                zorder=4.0)
    except Exception:
        pass

    # Titles/subtitle
    ref_name = (
        df2["permuter__ref"].iloc[0]
        if "permuter__ref" in df2.columns and not df2.empty
        else ""
    )
    title = f"{job_name}{f' ({ref_name})' if ref_name else ''}"
    fig.suptitle(title, fontsize=int(round(13 * fs)))
    if evaluators:
        fig.text(
            0.5,
            0.96,
            evaluators,
            ha="center",
            va="top",
            fontsize=int(round(9.5 * fs)),
            alpha=0.75,
        )

    # --- Export TOP-K observed (id, sequence, and all metadata) ---
    try:
        top_k = int(export_top_k) if export_top_k is not None else 500
        if top_k > 0 and dataset_dir and N_all > 0:
            limit = min(top_k, N_all)
            top_observed = df2.sort_values("_y", ascending=False, kind="mergesort").head(limit).copy()
            # Select id + sequence + all columns (keep schema as-is)
            cols = ["id", "sequence"] + [c for c in top_observed.columns if c not in ("id", "sequence")]
            top_observed = top_observed[cols]
            dst = Path(dataset_dir).expanduser().resolve()
            dst.mkdir(parents=True, exist_ok=True)
            pqt = dst / f"TOPN_OBSERVED__{metric_id}__N={limit}.parquet"
            csv = dst / f"TOPN_OBSERVED__{metric_id}__N={limit}.csv"
            top_observed.to_parquet(pqt, index=False)
            top_observed.to_csv(csv, index=False)
    except Exception:
        pass

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
