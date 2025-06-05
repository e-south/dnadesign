"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/analyse/plots/scatter.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection

from dnadesign.cruncher.analyse.plots.scatter_utils import (
    _TRANS,
    compute_consensus_points,
    generate_random_baseline,
    get_tf_pair,
    load_per_pwm,
    subsample_df,
)
from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.utils.config import CruncherConfig


def plot_scatter(
    run_dir: Path,
    pwms: Dict[str, PWM],
    cfg: CruncherConfig,
) -> None:
    """
    Orchestrator for <run_dir>/scatter_pwm.png. Steps:
      1) Load gathered_per_pwm_everyN.csv.
      2) Load elites.json (to get sequence length + top‐K).
      3) Subsample up to max_n = 2000 points (sorted by 'draw').
      4) Pick x_tf, y_tf from cfg.regulator_sets.
      5) Generate random baseline (mock sequences) scored w/ scatter_scale.
      6) Compute consensus‐only points (x_val, y_val, tfname).
      7) Annotate top‐K elites from elites.json onto the same axes.
      8) Plot combined figure and save.
    """
    # 1) Load per‐PWM scores
    df_per_pwm = load_per_pwm(run_dir)  # gathered_per_pwm_everyN.csv

    # 2) Load elites.json (to get sequences + ranks, even though we won't annotate them)
    import json

    elites_path = run_dir / "elites.json"
    with elites_path.open("r") as fh:
        raw_elites = json.load(fh)
    df_elites_full = pd.DataFrame(raw_elites)

    # 3) Determine sequence length (all elites have same length)
    seq_len = len(df_elites_full["sequence"].iloc[0])
    if seq_len < 1:
        raise ValueError("plot_scatter: elite sequences must be length ≥ 1")

    # 4) Pick x_tf, y_tf
    x_tf, y_tf = get_tf_pair(cfg)

    # 5) Subsample up to 2000 draws, sorted by 'draw'
    df_sub = subsample_df(df_per_pwm, max_n=2000, sort_by="draw")

    # 6) Generate random baseline (length = seq_len, n_samples = len(df_sub))
    n_random = len(df_sub)
    df_random = generate_random_baseline(pwms, cfg, length=seq_len, n_samples=n_random)

    # 7) Compute consensus‐only points (x_val, y_val, tfname)
    consensus_pts = compute_consensus_points(pwms, cfg, length=seq_len)

    # 8) Pick top‐K elites (by rank ascending) from elites.json (we won't annotate them below)
    K = min(3, len(df_elites_full))
    df_topk = df_elites_full.nsmallest(K, "rank")

    # 9) For each of the top‐K sequences, compute (x,y) under the same scale
    pair_scorer = Scorer(
        {x_tf: pwms[x_tf], y_tf: pwms[y_tf]},
        bidirectional=cfg.sample.bidirectional,
        scale=cfg.analysis.scatter_scale.lower(),
    )
    topk_coords: List[Tuple[float, float, int]] = []
    for _, row in df_topk.iterrows():
        seq_str = row["sequence"]
        ascii_arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
        seq_ints = _TRANS[ascii_arr].astype(np.int8)
        per_tf = pair_scorer.compute_all_per_pwm(seq_ints, seq_len)
        cx = float(per_tf[x_tf])
        cy = float(per_tf[y_tf])
        rank_i = int(row["rank"])
        topk_coords.append((cx, cy, rank_i))

    # 10) Compute PWM widths for subtitle
    width_x = getattr(pwms[x_tf], "length", None)
    width_y = getattr(pwms[y_tf], "length", None)
    if width_x is None or width_y is None:
        raise AttributeError(
            "plot_scatter: Could not find `.length` on one of the PWMs; "
            "update these lines to use the correct PWM‐width attribute."
        )

    # 11) Now draw everything
    out_png = run_dir / "scatter_pwm.png"
    out_png.parent.mkdir(exist_ok=True, parents=True)

    _draw_scatter_figure(
        df_samples=df_sub,
        df_random=df_random,
        consensus_pts=consensus_pts,
        topk_coords=topk_coords,
        x_tf=x_tf,
        y_tf=y_tf,
        seq_len=seq_len,
        width_x=width_x,
        width_y=width_y,
        cfg=cfg,
        out_path=out_png,
    )


def _draw_scatter_figure(
    df_samples: pd.DataFrame,
    df_random: pd.DataFrame,
    consensus_pts: List[Tuple[float, float, str]],
    topk_coords: List[Tuple[float, float, int]],
    x_tf: str,
    y_tf: str,
    seq_len: int,
    width_x: int,
    width_y: int,
    cfg: CruncherConfig,
    out_path: Path,
) -> None:
    """
    Draw and save the combined PWM scatter-plot:

      • Light gray cloud = df_random
      • Colored trajectories & scatter = df_samples (with 'score_<x_tf>' & 'score_<y_tf>')
      • Big red stars = consensus_pts   (from compute_consensus_points)
      • (Elites have been computed but NOT annotated here)
      • Multiline annotation (top‐left) with sampler settings
      • Two‐line title:
          Line 1: "MCMC vs. Random Sequences for {x_tf} and {y_tf}"
          Line 2: "Seq length = {seq_len}, PWM widths: {x_tf}={width_x}, {y_tf}={width_y}"
    """
    # 0) Basic checks
    required_cols = {"chain", "draw", f"score_{x_tf}", f"score_{y_tf}"}
    missing = required_cols - set(df_samples.columns)
    if missing:
        raise ValueError(f"_draw_scatter_figure: missing columns {missing} in df_samples")

    unique_chains = sorted(df_samples["chain"].unique())
    n_chains = len(unique_chains)
    if n_chains == 0:
        raise ValueError("_draw_scatter_figure: no chains found in df_samples")

    # Color palette for chains
    base_palette = sns.color_palette("deep", n_chains)
    chain_to_color: dict[int, Tuple[float, float, float]] = {
        cid: base_palette[i] for i, cid in enumerate(unique_chains)
    }

    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(6, 6))

    # Uniform alpha for **all** points
    uni_alpha = 0.6

    # 1) Gray cloud for random baseline (no edge lines, uniform alpha)
    if not df_random.empty:
        ax.scatter(
            df_random[f"score_{x_tf}"],
            df_random[f"score_{y_tf}"],
            c="lightgray",
            alpha=uni_alpha,
            s=20,
            linewidth=0,  # no border
            edgecolors="none",  # force no edge
            label="random",
        )

    # 2) MCMC trajectories & scatter: each chain gets a color ramp
    for chain_id in unique_chains:
        df_chain = df_samples[df_samples["chain"] == chain_id]
        iters = df_chain["draw"].to_numpy()
        itmin, itmax = iters.min(), iters.max()
        if itmax == itmin:
            norm_iters = np.zeros_like(iters, dtype=float)
        else:
            norm_iters = (iters - itmin) / (itmax - itmin)

        base_hue = np.array(chain_to_color[chain_id])  # (3,)

        # “Floor” so that early draws are not pure white:
        floor = 0.4
        whites = np.ones((len(norm_iters), 3))
        shade = floor + (1.0 - floor) * norm_iters
        per_point_rgb = (1.0 - shade).reshape(-1, 1) * whites + shade.reshape(-1, 1) * base_hue

        # Extract the kept‐points for this chain, sorted by draw
        df_chain_sorted = df_chain.sort_values("draw")
        pts = df_chain_sorted[[f"score_{x_tf}", f"score_{y_tf}"]].to_numpy()

        # Build one continuous polyline through all kept points (in ascending‐draw order)
        if len(pts) >= 2:
            segments: List[np.ndarray] = []
            seg_shades: List[float] = []

            for i in range(len(pts) - 1):
                # Connect (draw_i) → (draw_{i+1})
                start_pt = pts[i]
                end_pt = pts[i + 1]
                segments.append(np.array([start_pt, end_pt]))
                seg_shades.append(0.5 * (shade[i] + shade[i + 1]))

            seg_shades = np.array(seg_shades)
            seg_whites = np.ones((len(seg_shades), 3))
            seg_colors = (1.0 - seg_shades).reshape(-1, 1) * seg_whites + seg_shades.reshape(-1, 1) * base_hue

            lc = LineCollection(segments, colors=seg_colors, linewidths=1, alpha=uni_alpha)
            ax.add_collection(lc)

        # Plot each kept point (colored by its normalized‐iteration shade)
        ax.scatter(
            df_chain_sorted[f"score_{x_tf}"],
            df_chain_sorted[f"score_{y_tf}"],
            c=per_point_rgb,
            alpha=uni_alpha,
            s=30,
            linewidth=0,
            edgecolors="none",
            label=f"chain {chain_id}",
        )

    # 3) Plot consensus points (big red stars)
    for cx, cy, tfname in consensus_pts:
        ax.scatter(
            cx,
            cy,
            marker="*",
            s=200,
            facecolors="red",
            edgecolors="none",
            alpha=uni_alpha,
            label=f"consensus_{tfname}",
        )
        ax.text(cx, cy, f" {tfname}", va="center", ha="left", fontsize=10)

    # ——— We have removed ALL “top‐K elites” annotations here ———

    # 4) Label axes based on scale
    scatter_scale = cfg.analysis.scatter_scale.lower()
    if scatter_scale == "llr":
        xlabel = f"LLR_{x_tf}"
        ylabel = f"LLR_{y_tf}"
    elif scatter_scale == "z":
        xlabel = f"Z_{x_tf}"
        ylabel = f"Z_{y_tf}"
    elif scatter_scale == "p":
        xlabel = f"P_{x_tf}"
        ylabel = f"P_{y_tf}"
    elif scatter_scale == "logp":
        xlabel = f"-log₁₀(p)_{x_tf}"
        ylabel = f"-log₁₀(p)_{y_tf}"
    else:
        xlabel = f"{scatter_scale}_{x_tf}"
        ylabel = f"{scatter_scale}_{y_tf}"

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # 5) Two‐line title (line break between title and subtitle)
    title_line1 = f"MCMC vs. Random Sequences for {x_tf} and {y_tf}"
    title_line2 = f"Seq length = {seq_len}, PWM widths: {x_tf}={width_x}, {y_tf}={width_y}"
    ax.set_title(title_line1 + "\n" + title_line2, fontsize=10)

    # 6) Build a small multiline annotation of sampler settings (top‐left inside axes)
    sample_cfg = cfg.sample
    opt_cfg = sample_cfg.optimiser

    # Compute “iterations” = tune + draws
    total_iters = sample_cfg.tune + sample_cfg.draws

    # Grab move probabilities (from sample.moves, not optimiser)
    mprobs = sample_cfg.moves.move_probs
    s_p = mprobs["S"]
    b_p = mprobs["B"]
    m_p = mprobs["M"]

    # Just show “cooling beta: <kind>”
    cooling_kind = opt_cfg.cooling.kind

    annotation_lines = [
        f"chains        = {sample_cfg.chains}",
        f"iterations    = {total_iters}",
        f"move_probs    = S:{s_p:.2f}, B:{b_p:.2f}, M:{m_p:.2f}",
        f"cooling beta  = {cooling_kind}",
    ]
    annotation_text = "\n".join(annotation_lines)

    ax.text(
        0.01,
        0.90,
        annotation_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.95, edgecolor="none", pad=3),
    )

    sns.despine(ax=ax)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
