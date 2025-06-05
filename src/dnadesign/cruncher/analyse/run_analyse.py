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
    """
    For each sample‐batch name in cfg.analysis.runs:
      1) Locate <RESULTS_DIR>/<batch_name>.
      2) If sequences.csv and trace.nc exist, run gather_per_pwm_scores(...) → gathered_per_pwm_everyN.csv.
      3) Produce ALL diagnostic plots + scatter_pwm.png.
    """
    # 1) Build a shared PWM registry (only once)
    reg = Registry(PWM_PATH, cfg.parse.formats)
    flat_tfs = {tf for group in cfg.regulator_sets for tf in group}
    pwms: dict[str, PWM] = {tf: reg.load(tf) for tf in sorted(flat_tfs)}

    # 2) Iterate over all batch names specified in cfg.analysis.runs
    for batch_name in cfg.analysis.runs or []:
        sample_dir = RESULTS_DIR / batch_name

        if not sample_dir.is_dir():
            print(f"[analyse] Warning: sample batch '{batch_name}' not found under {RESULTS_DIR}")
            continue

        print(f"[analyse] Analysing batch: {batch_name}")

        elites_path = sample_dir / "elites.json"
        seq_path = sample_dir / "sequences.csv"
        trace_path = sample_dir / "trace.nc"

        # 2a) Skip if no elites.json (meaning sampling never produced any elites)
        if not elites_path.exists():
            print(f"[analyse:{batch_name}] skip—elites.json not found.")
            continue
        df_elites = pd.read_json(elites_path)

        # 2b) Attempt to load sequences.csv & trace.nc
        seq_df = pd.read_csv(seq_path) if seq_path.exists() else pd.DataFrame()
        if trace_path.exists():
            try:
                trace_idata = az.from_netcdf(trace_path)
            except Exception as e:
                raise RuntimeError(f"[analyse:{batch_name}] failed to load trace.nc as InferenceData: {e!r}")
        else:
            trace_idata = None

        # 2c) If both exist, run gather_per_pwm_scores(...) → gathered_per_pwm_everyN.csv
        if not seq_df.empty and (trace_idata is not None):
            gather_per_pwm_scores(
                sample_dir,
                cfg.analysis.subsampling_epsilon,
                pwms,
                bidirectional=cfg.sample.bidirectional,
                penalties=cfg.sample.penalties,
                scale=cfg.analysis.scatter_scale,
            )
            print(f"[analyse:{batch_name}] Wrote gathered_per_pwm_everyN.csv")
        else:
            print(f"[analyse:{batch_name}] skip gather_per_pwm—missing sequences.csv or valid trace.nc.")

        # 3) If trace.nc is present, call every diagnostic in diagnostics.py:
        if trace_idata is not None:
            # 3a) Trace + Posterior Density
            plot_trace(trace_idata, sample_dir)
            print(f"[analyse:{batch_name}] Wrote trace_score.png")

            # 3b) Autocorrelation
            plot_autocorr(trace_idata, sample_dir)
            print(f"[analyse:{batch_name}] Wrote autocorr_score.png")

            # 3c) Convergence metrics (rhat, ess)
            report_convergence(trace_idata, sample_dir)
            print(f"[analyse:{batch_name}] Wrote convergence.txt")

            # 3d) Rank plot
            plot_rank_diagnostic(trace_idata, sample_dir)
            print(f"[analyse:{batch_name}] Wrote rank_plot_score.png")

            # 3e) ESS evolution
            plot_ess(trace_idata, sample_dir)
            print(f"[analyse:{batch_name}] Wrote ess_evolution_score.png")

            # 3f) Pairwise & Parallel plots of (cpxR, soxR)
            #     We assume sequences.csv has columns: chain, draw, phase, score_cpxR, score_soxR
            if seq_path.exists():
                idata_pair = make_pair_idata(sample_dir)
                plot_pair_pwm_scores(idata_pair, sample_dir)
                print(f"[analyse:{batch_name}] Wrote pair_pwm_scores.png")

                plot_parallel_pwm_scores(idata_pair, sample_dir)
                print(f"[analyse:{batch_name}] Wrote parallel_pwm_scores.png")

        else:
            print(f"[analyse:{batch_name}] skip diagnostics—trace.nc missing or failed to load.")

        # 4) Regardless of trace, if scatter_pwm is requested in cfg, plot it (after gather_per_pwm)
        if cfg.analysis.plots.get("scatter_pwm", False):
            gathered_path = sample_dir / "gathered_per_pwm_everyN.csv"
            if gathered_path.exists() and not df_elites.empty:
                plot_scatter(sample_dir, pwms, cfg)
                print(f"[analyse:{batch_name}] Wrote scatter_pwm.png")
            else:
                print(
                    f"[analyse:{batch_name}] skip 'scatter_pwm'—"
                    "missing gathered_per_pwm_everyN.csv or empty elites.json."
                )

        # 5)  Summary: top-5 elites with both TF scores side-by-side
        if not df_elites.empty:
            # 1) Expand the nested `per_tf` dict into flat score_<tf> columns
            for tf in df_elites.iloc[0]["per_tf"].keys():
                df_elites[f"score_{tf}"] = df_elites["per_tf"].apply(lambda d, tf=tf: d[tf]["scaled_score"])

            # 2) If you want specific TF order, put them here; otherwise we grab the first two keys
            #    (assuming exactly two regulators in cfg.regulator_sets)
            tf_order = ["cpxR", "soxR"]  # ← replace with your TF names if different

            # 3) Print header
            print("\n=== Top-5 Elites (sequence + cpxR + soxR) ===")
            print(f"{'Rank':<5} {'Sequence':<30} {tf_order[0]:>10} {tf_order[1]:>10}")
            print("-" * (5 + 1 + 30 + 1 + 10 + 1 + 10))

            # 4) Select the top-5 by combined rank (assumes `rank` column exists)
            top5 = df_elites.nsmallest(5, "rank")

            # 5) Print each in one line: rank, sequence, score_cpxR, score_soxR
            for _, row in top5.iterrows():
                seq_str = row["sequence"]
                s0 = row[f"score_{tf_order[0]}"]
                s1 = row[f"score_{tf_order[1]}"]
                print(f"{int(row['rank']):<5} {seq_str:<30} {s0:10.1f} {s1:10.1f}")

            print("=" * (5 + 1 + 30 + 1 + 10 + 1 + 10) + "\n")
