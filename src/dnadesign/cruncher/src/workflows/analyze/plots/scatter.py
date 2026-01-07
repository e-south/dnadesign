"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/workflows/analyze/plots/scatter.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.utils.elites import find_elites_parquet
from dnadesign.cruncher.utils.manifest import load_manifest
from dnadesign.cruncher.utils.parquet import read_parquet
from dnadesign.cruncher.workflows.analyze.plots.scatter_utils import (
    compute_consensus_points,
    encode_sequence,
    generate_random_baseline,
    load_per_pwm,
    subsample_df,
)

mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.compression"] = 9  # smaller file size

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)


def plot_scatter(
    run_dir: Path,
    pwms: Dict[str, PWM],
    cfg: CruncherConfig,
    tf_pair: tuple[str, str],
    per_pwm_path: Path,
    out_dir: Path,
    *,
    bidirectional: bool,
    pwm_sum_threshold: float,
    annotation: str,
) -> None:
    """
    Orchestrator for pwm__scatter.{png,pdf} under the analysis plots directory.
    Robust to the case where there are zero elites.
    """
    # 1) Load per‐PWM scores
    df_per_pwm = load_per_pwm(per_pwm_path)
    if df_per_pwm.empty:
        raise ValueError("plot_scatter: per-PWM score table is empty; check sequences.parquet and subsampling_epsilon.")

    # 2) Load elites parquet (may be empty)
    elites_path = find_elites_parquet(run_dir)
    df_elites = read_parquet(elites_path)

    # 3) Get sequence length from manifest (explicit)
    manifest = load_manifest(run_dir)
    seq_len = int(manifest.get("sequence_length") or 0)
    if seq_len < 1:
        raise ValueError("plot_scatter: sequence_length missing from run_manifest.json")

    # 4) Pick TF pair
    x_tf, y_tf = tf_pair
    required_cols = {f"score_{x_tf}", f"score_{y_tf}"}
    missing_cols = [col for col in required_cols if col not in df_per_pwm.columns]
    if missing_cols:
        raise ValueError("plot_scatter: per-PWM table missing required score columns: " + ", ".join(missing_cols))

    style = cfg.analysis.scatter_style.lower()
    scale = cfg.analysis.scatter_scale.lower()
    if style == "thresholds" and scale != "llr":
        raise ValueError("scatter_style='thresholds' requires scatter_scale='llr'.")

    # 5) Subsample up to 2000 mcmc points
    df_sub = subsample_df(df_per_pwm, max_n=2000, sort_by="draw")

    # 6) Random baseline
    df_random = generate_random_baseline(pwms, cfg, length=seq_len, n_samples=len(df_sub), bidirectional=bidirectional)

    # 7) Consensus points
    consensus_pts = compute_consensus_points(
        pwms,
        cfg,
        length=seq_len,
        tf_pair=(x_tf, y_tf),
        bidirectional=bidirectional,
    )

    # 8) Elite coordinates (skip cleanly if no elites or no 'sequence' column)
    elite_coords: List[Tuple[float, float, int]] = []
    if not df_elites.empty:
        if "sequence" not in df_elites.columns:
            raise ValueError("plot_scatter: elites parquet missing 'sequence' column")
        pair_scorer = Scorer(
            {x_tf: pwms[x_tf], y_tf: pwms[y_tf]},
            bidirectional=bidirectional,
            scale=scale,
        )
        for idx, row in df_elites.iterrows():
            seq = row["sequence"]
            seq_ints = encode_sequence(seq, context=f"elites.parquet row={idx}")
            per_tf = pair_scorer.compute_all_per_pwm(seq_ints, seq_len)
            # 'rank' may be absent; default to a large number so outline order is deterministic
            rank_val = int(row["rank"]) if "rank" in row and pd.notna(row["rank"]) else 10**9
            elite_coords.append((float(per_tf[x_tf]), float(per_tf[y_tf]), rank_val))

    # 9) PWM widths for subtitle
    width_x = getattr(pwms[x_tf], "length", None)
    width_y = getattr(pwms[y_tf], "length", None)
    if width_x is None or width_y is None:
        raise AttributeError("plot_scatter: cannot find PWM.length attributes")

    # 10) Draw
    out_pdf = out_dir / "pwm__scatter.pdf"
    out_png = out_dir / "pwm__scatter.png"
    out_pdf.parent.mkdir(exist_ok=True, parents=True)

    _draw_scatter_figure(
        df_samples=df_sub,
        df_random=df_random,
        consensus_pts=consensus_pts,
        elite_coords=elite_coords,
        x_tf=x_tf,
        y_tf=y_tf,
        seq_len=seq_len,
        width_x=width_x,
        width_y=width_y,
        pwms=pwms,
        cfg=cfg,
        bidirectional=bidirectional,
        pwm_sum_threshold=pwm_sum_threshold,
        annotation=annotation,
        out_path=out_pdf,
    )
    _draw_scatter_figure(
        df_samples=df_sub,
        df_random=df_random,
        consensus_pts=consensus_pts,
        elite_coords=elite_coords,
        x_tf=x_tf,
        y_tf=y_tf,
        seq_len=seq_len,
        width_x=width_x,
        width_y=width_y,
        pwms=pwms,
        cfg=cfg,
        bidirectional=bidirectional,
        pwm_sum_threshold=pwm_sum_threshold,
        annotation=annotation,
        out_path=out_png,
    )


def _draw_scatter_figure(
    df_samples: pd.DataFrame,
    df_random: pd.DataFrame,
    consensus_pts: List[Tuple[float, float, str]],
    elite_coords: List[Tuple[float, float, int]],
    x_tf: str,
    y_tf: str,
    seq_len: int,
    width_x: int,
    width_y: int,
    pwms: Dict[str, PWM],
    cfg: CruncherConfig,
    bidirectional: bool,
    pwm_sum_threshold: float,
    annotation: str,
    out_path: Path,
) -> None:
    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(6, 6))

    # Global baseline alpha for non-chain layers
    uni_alpha = 0.6

    # 1) RANDOM BASELINE (always raw-LLR space) — BACKGROUND layer
    if not df_random.empty:
        ax.scatter(
            df_random[f"score_{x_tf}"],
            df_random[f"score_{y_tf}"],
            c="lightgray",
            alpha=uni_alpha,
            s=20,
            linewidth=0,
            edgecolors="none",
            label="random",
            zorder=0,
        )

    style = cfg.analysis.scatter_style.lower()
    scale = cfg.analysis.scatter_scale.lower()

    # 2) “EDGES” STYLE
    if style == "edges":
        chains = sorted(df_samples["chain"].unique())
        palette = sns.color_palette("deep", len(chains))
        chain_to_color = {c: np.array(palette[i]) for i, c in enumerate(chains)}

        # Point alpha progression (slightly higher at onset, ramps with draw index)
        alpha_min = 0.65  # higher baseline opacity at t0
        alpha_max = 0.95  # grows toward near-opaque

        # Precompute per-chain geometry & colors so we can
        # draw EDGES first (middle layer), then POINTS (top layer).
        per_chain = []
        for c in chains:
            dfc = df_samples[df_samples["chain"] == c].sort_values("draw")
            iters = dfc["draw"].to_numpy()
            t0, t1 = iters.min(), iters.max()
            norm = np.zeros_like(iters, float) if t1 == t0 else (iters - t0) / (t1 - t0)
            hue = chain_to_color[c]

            # Color progression (fade from near-white to chain hue)
            shade = 0.35 + 0.65 * norm  # start slightly closer to white than before
            rgb = (1 - shade)[:, None] + shade[:, None] * hue  # Nx3

            # Alpha progression for points
            alphas = alpha_min + (alpha_max - alpha_min) * norm  # N
            rgba = np.concatenate([rgb, alphas[:, None]], axis=1)  # Nx4

            pts = dfc[[f"score_{x_tf}", f"score_{y_tf}"]].to_numpy()
            per_chain.append((c, dfc, pts, hue, rgba))

        # 2a) Draw EDGES (constant hue) — MIDDLE layer
        for c, dfc, pts, hue, _rgba in per_chain:
            if len(pts) >= 2:
                segs = [np.array([pts[i], pts[i + 1]]) for i in range(len(pts) - 1)]
                ax.add_collection(
                    LineCollection(
                        segs,
                        colors=[hue] * (len(segs)),
                        linewidths=1.2,
                        alpha=0.85,
                        zorder=2,
                    )
                )

        # 2b) Draw POINTS with maturation color+alpha — TOP layer
        for c, dfc, _pts, _hue, rgba in per_chain:
            ax.scatter(
                dfc[f"score_{x_tf}"],
                dfc[f"score_{y_tf}"],
                c=rgba,  # per-point RGBA
                s=30,
                linewidth=0,
                edgecolors="none",
                label=f"chain {c}",
                zorder=3,
            )

    # 3) “THRESHOLDS” STYLE (unchanged layering; background then highlights)
    elif style == "thresholds":
        thr = pwm_sum_threshold

        # (i) consensus LLRs — still in raw-LLR scale
        raw_scorer = Scorer(
            {x_tf: pwms[x_tf], y_tf: pwms[y_tf]},
            bidirectional=bidirectional,
            scale="llr",
        )
        cons_x = raw_scorer.consensus_llr(x_tf)
        cons_y = raw_scorer.consensus_llr(y_tf)

        # sanity-check anchors
        if not np.isfinite(cons_x) or not np.isfinite(cons_y) or cons_x <= 0 or cons_y <= 0:
            raise ValueError(f"Consensus LLR must be positive and finite; got {cons_x:.3f} / {cons_y:.3f}")

        # (ii) normalise every sample; discard any invalid rows
        x_norm = (df_samples[f"score_{x_tf}"] / cons_x).to_numpy(dtype=float)
        y_norm = (df_samples[f"score_{y_tf}"] / cons_y).to_numpy(dtype=float)
        finite = np.isfinite(x_norm) & np.isfinite(y_norm)
        x_norm, y_norm = x_norm[finite], y_norm[finite]

        # (iii) scatter the background
        ax.scatter(
            x_norm,
            y_norm,
            c="lightgray",
            alpha=uni_alpha,
            s=30,
            linewidth=0,
            edgecolors="none",
            label="mcmc",
            zorder=0,
        )

        # (iv) highlight points above Σ-threshold
        mask = (x_norm + y_norm) >= thr
        ax.scatter(
            x_norm[mask],
            y_norm[mask],
            c="red",
            alpha=uni_alpha,
            s=30,
            linewidth=0,
            edgecolors="none",
            label=f"x+y ≥ {thr:.2f}",
            zorder=1,
        )

        # (v) iso-cost diagonal & axes limits
        xs = np.linspace(0.0, 1.0, 200)
        ax.plot(xs, thr - xs, "--", linewidth=1, label=f"x+y = {thr:.2f}", zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        xlabel = f"{x_tf} / consensus_{x_tf}"
        ylabel = f"{y_tf} / consensus_{y_tf}"

    else:
        raise ValueError(f"Unknown scatter_style '{cfg.analysis.scatter_style}'")

    # 4) CONSENSUS STARS & ELITE OUTLINES — FOREGROUND accents
    for cx, cy, name in consensus_pts:
        ax.scatter(
            cx,
            cy,
            marker="*",
            s=200,
            c="red",
            edgecolors="none",
            alpha=uni_alpha,
            zorder=4,
        )
        ax.text(cx, cy, f" {name}", va="center", ha="left", fontsize=10, zorder=4)

    for ex, ey, _ in elite_coords:
        ax.scatter(
            ex,
            ey,
            marker="o",
            s=50,
            facecolors="none",
            edgecolors="blue",
            linewidth=1,
            zorder=4,
        )

    # 5) TITLES, LABELS, LEGEND
    if style != "thresholds":
        label_map = {
            "llr": "LLR",
            "z": "Z",
            "logp": "-log10(p)",
            "consensus-neglop-sum": "NeglogP/consensus",
        }
        xlabel = f"{label_map.get(scale, scale)}_{x_tf}"
        ylabel = f"{label_map.get(scale, scale)}_{y_tf}"

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"MCMC vs. Random for {x_tf}/{y_tf}\nSeq length={seq_len}, PWM widths: {x_tf}={width_x}, {y_tf}={width_y}",
        fontsize=10,
    )

    if annotation:
        ax.text(
            0.01,
            0.90,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.95, edgecolor="none", pad=3),
            zorder=5,
        )

    sns.despine(ax=ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
