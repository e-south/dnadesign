"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/main.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr  # for loading trace.nc

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.parse.registry import Registry
from dnadesign.cruncher.sample.plots.autocorr import plot_autocorr
from dnadesign.cruncher.sample.plots.convergence import report_convergence
from dnadesign.cruncher.sample.plots.logo_elites import plot_logo_elites
from dnadesign.cruncher.sample.plots.scatter_pwm import plot_scatter_plot
from dnadesign.cruncher.sample.plots.score_kde import plot_score_kde
from dnadesign.cruncher.sample.plots.trace import plot_trace
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.transforms import LogPNorm, PlainP, ZTransform
from dnadesign.cruncher.utils.config import CruncherConfig


def run_analyse(cfg: CruncherConfig, base_out: Path) -> None:
    """
    “Analyse” stage: for each batch_name in cfg.analysis.runs, reload all saved data
    (hits.csv, samples.csv, random_samples.csv, trace.nc) and regenerate requested plots.

    - cfg.analysis.runs: List[str] of batch names (directory names under cfg.out_dir).
    - cfg.analysis.plot_scales: Dict[plot_name, scale], where plot_name ∈
      {"trace", "autocorr", "convergence", "scatter_pwm", "score_kde", "logo_elites"}.

    Example:
      cfg.analysis.runs = ["sample_cpxR-soxR_20250602"]
      cfg.analysis.plot_scales = {"scatter_pwm": "logp_norm", "score_kde": "llr"}

    For each run in runs:
      1. Construct `run_dir = <PROJECT_ROOT>/.../cruncher/<cfg.out_dir>/<batch_name>`
      2. Load hits.csv, samples.csv, random_samples.csv (if exists), trace.nc (if exists).
      3. Load PWMs via same logic as run_sample.
      4. For each plot_name in cfg.analysis.plot_scales:
         - Determine which transform to use (e.g. LogPNorm, ZTransform, etc.).
         - Call the corresponding plotting function, saving into run_dir.
    """
    # PROJECT_ROOT / "dnadesign"/ "src"/ "dnadesign"/ "cruncher"/ cfg.out_dir
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    base_results = PROJECT_ROOT / "dnadesign" / "src" / "dnadesign" / "cruncher" / cfg.out_dir

    for batch_name in cfg.analysis.runs or []:
        run_dir = base_results / batch_name
        if not run_dir.is_dir():
            print(f"[analyse] Warning: batch directory '{batch_name}' not found under {base_results}")
            continue

        print(f"[analyse] Re-analysing batch: {batch_name}")

        # 1) LOAD PWMS
        reg = Registry(
            PROJECT_ROOT.parent / "dnadesign-data" / "primary_literature" / "OMalley_et_al" / "escherichia_coli_motifs",
            cfg.parse.formats,
        )
        flat_tfs = {tf for group in cfg.regulator_sets for tf in group}
        pwms: dict[str, PWM] = {tf: reg.load(tf) for tf in flat_tfs}

        # 2) LOAD saved data
        hits_path = run_dir / "hits.csv"
        samples_path = run_dir / "samples.csv"
        random_path = run_dir / "random_samples.csv"
        trace_path = run_dir / "trace.nc"

        # hits.csv is mandatory
        hits_df = pd.read_csv(hits_path)

        # samples.csv might be missing if the optimiser didn't record it
        samples_df = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()

        # random_samples.csv might be missing
        random_df = pd.read_csv(random_path) if random_path.exists() else pd.DataFrame()

        # trace.nc might be missing
        trace_ds = xr.open_dataset(trace_path) if trace_path.exists() else None

        # 3) For every requested plot, dispatch to the correct plotting routine
        for plot_name, scale in cfg.analysis.plot_scales.items():
            match plot_name:
                case "trace":
                    if trace_ds is None:
                        print(f"[analyse:{batch_name}] skip 'trace'—trace.nc not found.")
                        continue
                    out_png = run_dir / "trace.png"
                    beta_series = None  # If you had a β‐trace saved, you'd load it too
                    plot_trace(trace_ds, out_png)
                    print(f"[analyse:{batch_name}] Plotted MCMC trace → {out_png.name}")

                case "autocorr":
                    if trace_ds is None:
                        print(f"[analyse:{batch_name}] skip 'autocorr'—trace.nc not found.")
                        continue
                    out_png = run_dir / "autocorr.png"
                    plot_autocorr(trace_ds, out_png)
                    print(f"[analyse:{batch_name}] Plotted autocorrelation → {out_png.name}")

                case "convergence":
                    if trace_ds is None:
                        print(f"[analyse:{batch_name}] skip 'convergence'—trace.nc not found.")
                        continue
                    out_md = run_dir / "convergence.md"
                    report_convergence(trace_ds, run_dir)
                    print(f"[analyse:{batch_name}] Generated convergence report → {out_md.name}")

                case "scatter_pwm":
                    # Need samples_df, random_df, hits_df, and Transform(logp_norm, z, etc.)
                    if samples_df.empty or hits_df.empty:
                        print(f"[analyse:{batch_name}] skip 'scatter_pwm'—missing samples.csv or hits.csv.")
                        continue

                    # Prepare a “raw” scorer to fetch raw LLRs (score_per_pwm gives raw LLR)
                    scorer_raw = Scorer(
                        pwms,
                        bidirectional=cfg.sample.bidirectional,
                        scale="llr",  # force raw-llr so we can transform ourselves
                        background=(0.25, 0.25, 0.25, 0.25),
                        penalties=cfg.sample.penalties,
                    )

                    # Build the appropriate Transform instance:
                    if scale == "logp_norm":
                        transform = LogPNorm(pwms, bidirectional=cfg.sample.bidirectional)
                    elif scale == "z":
                        transform = ZTransform(pwms, bidirectional=cfg.sample.bidirectional)
                    else:  # "llr" or "p"
                        transform = PlainP(pwms, bidirectional=cfg.sample.bidirectional)

                    # We delegate to plot_scatter_plot, which expects:
                    #   df_samples (already “scale‐transformed”), df_random, consensus points, etc.
                    out_png = run_dir / "scatter_pwm.png"
                    plot_scatter_plot(
                        df_samples=samples_df,
                        df_random=random_df,
                        consensus_pts=None,  # internally recomputed in plot_scatter_plot
                        x_tf=next(iter(pwms)),  # plot function will extract the first two TF names
                        y_tf=next(iter(pwms)),  # same here (internally overridden)
                        scale=scale,
                        cfg=cfg,
                        out_path=out_png,
                    )
                    print(f"[analyse:{batch_name}] Plotted scatter_pwm → {out_png.name}")

                case "score_kde":
                    # Requires hits_df to get raw scores per PWM
                    if hits_df.empty:
                        print(f"[analyse:{batch_name}] skip 'score_kde'—hits.csv not found.")
                        continue

                    # For KDE, we only need the raw LLRs (or whatever scale)
                    # Build Transform if needed:
                    if scale == "llr":
                        transform = None  # no transform; use raw LLR from hits_df
                    elif scale == "z":
                        transform = ZTransform(pwms, bidirectional=cfg.sample.bidirectional)
                    else:  # "p" or "logp_norm"
                        transform = (
                            PlainP(pwms, bidirectional=cfg.sample.bidirectional)
                            if scale == "p"
                            else LogPNorm(pwms, bidirectional=cfg.sample.bidirectional)
                        )

                    out_png = run_dir / "score_kde.png"
                    plot_score_kde(hits_df, pwms, cfg, transform, out_png, scale)
                    print(f"[analyse:{batch_name}] Plotted score_kde → {out_png.name}")

                case "logo_elites":
                    # Requires hits_df (top-K sequences) and PWM logos
                    if hits_df.empty:
                        print(f"[analyse:{batch_name}] skip 'logo_elites'—hits.csv not found.")
                        continue

                    out_png = run_dir / "logo_elites.png"
                    plot_logo_elites(hits_df, pwms, cfg, out_png)
                    print(f"[analyse:{batch_name}] Plotted logo_elites → {out_png.name}")

                case other:
                    print(f"[analyse:{batch_name}] Unknown plot '{other}'—skipping.")
