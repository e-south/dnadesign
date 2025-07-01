"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/analyse/run_analyse.py

When run_analyse is called, it iterates through cfg.analysis.runs (a list of
existing “sample” batch names), and for each batch:
  1) It looks under the “results” folder for that sample's directory.
  2) If sequences.csv and trace.nc exist, runs gather_per_pwm_scores(...)
     → gathered_per_pwm_everyN.csv.
  3) Depending on cfg.analysis.plots, produces:
       - trace_score.png            (Trace + posterior + other diagnostics)
       - autocorr_score.png         (Auto-correlation)
       - convergence.txt            (R-hat & ESS)
       - rank_plot_score.png        (Rank-plot)
       - ess_evolution_score.png    (ESS evolution)
       - posterior_score.png        (1D posterior of combined score)
       - forest_score.png           (Forest plot of score)
       - pair_pwm_scores.png        (2D KDE + marginals for (cpxR, soxR))
       - parallel_pwm_scores.png    (Parallel coordinates of cpxR vs. soxR)
       - scatter_pwm.png            (MCMC vs. Random PWM scatter)
     All outputs are saved directly under <RESULTS_DIR>/<batch_name>.
Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import pandas as pd
import torch

from dnadesign.cruncher.analyse.per_pwm import gather_per_pwm_scores
from dnadesign.cruncher.analyse.plots.diagnostics import (
    make_pair_idata,
    plot_autocorr,
    plot_ess,
    plot_pair_pwm_scores,
    plot_parallel_pwm_scores,
    plot_rank_diagnostic,
    plot_trace,
    report_convergence,
)
from dnadesign.cruncher.analyse.plots.scatter import plot_scatter
from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.parse.registry import Registry
from dnadesign.cruncher.utils.config import CruncherConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "src" / "dnadesign" / "cruncher" / "results"
PWM_PATH = PROJECT_ROOT.parent / "dnadesign-data" / "primary_literature" / "OMalley_et_al" / "escherichia_coli_motifs"


def run_analyse(cfg: CruncherConfig) -> None:
    """Iterate over batches listed in cfg.analysis.runs and generate diagnostics."""
    # Shared PWM registry
    reg = Registry(PWM_PATH, cfg.parse.formats)
    flat_tfs = {tf for group in cfg.regulator_sets for tf in group}
    pwms: dict[str, PWM] = {tf: reg.load(tf) for tf in sorted(flat_tfs)}

    for batch_name in cfg.analysis.runs or []:
        sample_dir = RESULTS_DIR / batch_name

        if not sample_dir.is_dir():
            print(f"[analyse] Warning: batch '{batch_name}' not found under {RESULTS_DIR}")
            continue

        print(f"[analyse] Analysing batch: {batch_name}")

        # ── locate newest elites file (new or legacy layout) ──────────────────
        pt_candidates: list[Path] = []
        pt_candidates += list(sample_dir.glob("cruncher_elites_*/*.pt"))
        pt_candidates += list(sample_dir.glob("elites_*.pt"))

        if not pt_candidates:
            print(f"[analyse:{batch_name}] skip — no elites file found.")
            continue

        latest_pt = max(pt_candidates, key=lambda p: p.stat().st_mtime)
        elites_df = pd.DataFrame(torch.load(latest_pt))
        print(f"[analyse:{batch_name}] Loaded {len(elites_df)} elites from {latest_pt.relative_to(sample_dir)}")

        # ── load trace & sequences for downstream plots ───────────────────────
        seq_path, trace_path = sample_dir / "sequences.csv", sample_dir / "trace.nc"
        seq_df = pd.read_csv(seq_path) if seq_path.exists() else pd.DataFrame()
        trace_idata = az.from_netcdf(trace_path) if trace_path.exists() else None

        # gather per-PWM scores (for scatter etc.)
        if not seq_df.empty and trace_idata is not None:
            gather_per_pwm_scores(
                sample_dir,
                cfg.analysis.subsampling_epsilon,
                pwms,
                bidirectional=cfg.sample.bidirectional,
                penalties=cfg.sample.penalties,
                scale=cfg.analysis.scatter_scale,
            )
            print(f"[analyse:{batch_name}] Wrote gathered_per_pwm_everyN.csv")

        # ── diagnostics ───────────────────────────────────────────────────────
        if trace_idata is not None:
            plot_trace(trace_idata, sample_dir)
            plot_autocorr(trace_idata, sample_dir)
            report_convergence(trace_idata, sample_dir)
            plot_rank_diagnostic(trace_idata, sample_dir)
            plot_ess(trace_idata, sample_dir)

            if seq_path.exists():
                idata_pair = make_pair_idata(sample_dir, cfg)
                plot_pair_pwm_scores(idata_pair, sample_dir, cfg)
                plot_parallel_pwm_scores(idata_pair, sample_dir, cfg)

        # ── scatter plot ──────────────────────────────────────────────────────
        if cfg.analysis.plots.get("scatter_pwm", False):
            gathered_path = sample_dir / "gathered_per_pwm_everyN.csv"
            if gathered_path.exists():
                plot_scatter(sample_dir, pwms, cfg)
                print(f"[analyse:{batch_name}] Wrote scatter_pwm.png")

        # ── top-5 tabular summary (unchanged logic – dynamic TF names) ────────
        if not elites_df.empty:
            x_tf, y_tf = cfg.regulator_sets[0][:2]
            elites_df[f"score_{x_tf}"] = elites_df["per_tf"].apply(lambda d: d[x_tf]["scaled_score"])
            elites_df[f"score_{y_tf}"] = elites_df["per_tf"].apply(lambda d: d[y_tf]["scaled_score"])

            print(f"\n=== Top-5 Elites ({x_tf} & {y_tf}) ===")
            print(f"{'Rank':<5} {'Sequence':<30} {x_tf:>10} {y_tf:>10}")
            print("-" * 57)
            for _, row in elites_df.nsmallest(5, "rank").iterrows():
                print(
                    f"{int(row['rank']):<5} {row['sequence']:<30} "
                    f"{row[f'score_{x_tf}']:10.1f} {row[f'score_{y_tf}']:10.1f}"
                )
            print("-" * 57 + "\n")
