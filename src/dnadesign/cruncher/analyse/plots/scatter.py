"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/analyse/plots/scatter.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# dnadesign/cruncher/analyse/plots/scatter.py

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
    load_elites,
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
      1. Load gathered_per_pwm_everyN.csv.
      2. Load elites.json (to get sequence length + top‐K).
      3. Subsample up to max_n = 2000 points (sorted by 'draw').
      4. Pick x_tf, y_tf from cfg.regulator_sets.
      5. Generate random baseline (mock sequences) scored w/ scatter_scale.
      6. Compute consensus‐only points (x_val, y_val, tfname).
      7. Annotate top‐K elites from elites.json onto the same axes.
      8. Plot combined figure and save.
    """
    # 1) Load per‐PWM scores
    df_per_pwm = load_per_pwm(run_dir)  # gathered_per_pwm_everyN.csv

    # 2) Load elites.json (to get sequences + ranks)
    df_elites_full = load_elites(run_dir)
    # load_elites above only returned "sequence" by default; instead, re‐read raw JSON:
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

    # 8) Pick top‐K elites (by rank ascending) from elites.json
    K = min(3, len(df_elites_full))  # annotate top 3 (or fewer if <3 elites)
    df_topk = df_elites_full.nsmallest(K, "rank")

    # 9) For each of the top‐K sequences, compute (x,y) under the same scale
    #    reuse a two‐PWM Scorer for x_tf, y_tf only
    pair_scorer = Scorer(
        {x_tf: pwms[x_tf], y_tf: pwms[y_tf]},
        bidirectional=cfg.sample.bidirectional,
        scale=cfg.analysis.scatter_scale.lower(),
    )
    topk_coords: List[Tuple[float, float, int]] = []
    rnd = np.random.default_rng(0)

    for _, row in df_topk.iterrows():
        seq_str = row["sequence"]
        ascii_arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
        seq_ints = _TRANS[ascii_arr].astype(np.int8)
        per_tf = pair_scorer.compute_all_per_pwm(seq_ints, seq_len)
        cx = float(per_tf[x_tf])
        cy = float(per_tf[y_tf])
        rank_i = int(row["rank"])
        topk_coords.append((cx, cy, rank_i))

    # 10) Now draw everything
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
    cfg: CruncherConfig,
    out_path: Path,
) -> None:
    """
    Draw and save the combined PWM scatter‐plot:

      • Light gray cloud = df_random
      • Colored trajectories & scatter = df_samples (with 'score_<x_tf>' & 'score_<y_tf>')
      • Big red stars = consensus_pts   (computed from compute_consensus_points)
      • Big black circles = top‐K elites (from elites.json)
      • Multiline annotation (top‐left) with sampler settings
      • Main title = "Simulated-annealing MCMC {x_tf} and {y_tf} (L={seq_len})"
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

    # color palette for chains
    base_palette = sns.color_palette("colorblind", n_chains)
    chain_to_color: dict[int, Tuple[float, float, float]] = {
        cid: base_palette[i] for i, cid in enumerate(unique_chains)
    }

    sns.set_style("ticks", {"axes.grid": False})
    fig, ax = plt.subplots(figsize=(6, 6))

    # 1) Gray cloud for random baseline
    if not df_random.empty:
        ax.scatter(
            df_random[f"score_{x_tf}"],
            df_random[f"score_{y_tf}"],
            c="lightgray",
            alpha=0.4,
            s=20,
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
        whites = np.ones((len(norm_iters), 3))
        alphas = norm_iters.reshape(-1, 1)
        per_point_rgb = (1 - alphas) * whites + (alphas * base_hue)

        pts = df_chain.sort_values("draw")[[f"score_{x_tf}", f"score_{y_tf}"]].to_numpy()
        if len(pts) >= 2:
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            seg_alphas = (norm_iters[:-1] + norm_iters[1:]) / 2.0
            seg_colors = (1 - seg_alphas[:, None]) * np.ones((len(seg_alphas), 3)) + (seg_alphas[:, None] * base_hue)
            lc = LineCollection(segments, colors=seg_colors, linewidths=1, alpha=0.5)
            ax.add_collection(lc)

        ax.scatter(
            df_chain[f"score_{x_tf}"],
            df_chain[f"score_{y_tf}"],
            c=per_point_rgb,
            alpha=0.7,
            s=30,
            linewidth=0,
            label=f"chain {chain_id}",
        )

    # 3) Plot consensus points (big red stars)
    for cx, cy, tfname in consensus_pts:
        ax.scatter(cx, cy, marker="*", s=200, facecolors="red", edgecolors="none", label=f"consensus_{tfname}")
        ax.text(cx, cy, f" {tfname}", va="center", ha="left", fontsize=10)

    # 4) Plot top‐K elites (big black circles, labeled by rank)
    for cx, cy, rank_i in topk_coords:
        ax.scatter(cx, cy, marker="o", s=150, facecolors="none", edgecolors="black", linewidth=1.5)
        ax.text(cx, cy, f" #{rank_i}", va="center", ha="left", color="black", fontsize=8, fontweight="bold")

    # 5) Label axes based on scale
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

    # 6) Main title (with sequence length)
    title_str = f"Simulated-annealing MCMC {x_tf} and {y_tf} (L={seq_len})"
    ax.set_title(title_str, fontsize=12)

    # 7) Build a small multiline annotation of sampler settings (top‐left inside axes)
    sample_cfg = cfg.sample
    opt_cfg = sample_cfg.optimiser
    cooling = getattr(opt_cfg, "cooling", None)
    if getattr(cooling, "kind", "") == "linear":
        b0, b1 = cooling.beta
        cooling_str = f"linear (β={b0:.2f}→{b1:.2f})"
    else:
        betas = ", ".join(f"{b:.2f}" for b in getattr(cooling, "beta", []))
        cooling_str = f"geometric (β=[{betas}])"

    # initial_keep is always 10 in gather_per_pwm; if you ever want to make it configurable, pull from cfg.analysis
    initial_keep = 10

    annotation_lines = [
        f"chains = {sample_cfg.chains}",
        f"draws  = {sample_cfg.draws}",
        f"tune   = {sample_cfg.tune}",
        f"init_keep = {initial_keep}",
        f"cooling   = {cooling_str}",
        f"swap_prob = {opt_cfg.swap_prob:.2f}",
    ]
    annotation_text = "\n".join(annotation_lines)

    ax.text(
        0.01,
        0.99,
        annotation_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )

    sns.despine(ax=ax)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
