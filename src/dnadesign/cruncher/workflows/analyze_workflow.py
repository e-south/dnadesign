"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/workflows/analyze_workflow.py

When run_analyze is called, it iterates through cfg.analysis.runs (a list of
existing “sample” batch names), and for each batch:
  1) It looks under the configured output folder for that sample's directory.
  2) If sequences.parquet and trace.nc exist, runs gather_per_pwm_scores(...)
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
       - scatter_pwm.pdf            (MCMC vs. Random PWM scatter)
     All outputs are saved directly under <out_dir>/<batch_name>.
Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import yaml

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.services.run_service import list_runs
from dnadesign.cruncher.utils.manifest import load_manifest
from dnadesign.cruncher.workflows.analyze.per_pwm import gather_per_pwm_scores
from dnadesign.cruncher.workflows.analyze.plots.diagnostics import (
    make_pair_idata,
    plot_autocorr,
    plot_ess,
    plot_pair_pwm_scores,
    plot_parallel_pwm_scores,
    plot_rank_diagnostic,
    plot_trace,
    report_convergence,
)
from dnadesign.cruncher.workflows.analyze.plots.scatter import plot_scatter

logger = logging.getLogger(__name__)


def _load_pwms_from_config(run_dir: Path) -> tuple[dict[str, PWM], dict]:
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_used.yaml in {run_dir}")
    payload = yaml.safe_load(config_path.read_text()) or {}
    cruncher_cfg = payload.get("cruncher")
    if not isinstance(cruncher_cfg, dict):
        raise ValueError("config_used.yaml missing top-level 'cruncher' section.")
    pwms_info = cruncher_cfg.get("pwms_info")
    if not isinstance(pwms_info, dict) or not pwms_info:
        raise ValueError("config_used.yaml missing pwms_info; re-run `cruncher sample`.")
    pwms: dict[str, PWM] = {}
    for tf_name, info in pwms_info.items():
        matrix = info.get("pwm_matrix")
        if not matrix:
            raise ValueError(f"config_used.yaml missing pwm_matrix for TF '{tf_name}'.")
        pwms[tf_name] = PWM(name=tf_name, matrix=np.array(matrix, dtype=float))
    return pwms, cruncher_cfg


def run_analyze(cfg: CruncherConfig, config_path: Path) -> None:
    """Iterate over batches listed in cfg.analysis.runs and generate diagnostics."""
    # Shared PWM store
    if cfg.analysis is None:
        raise ValueError("analysis section is required for analyze")
    if cfg.sample is None:
        raise ValueError("sample section is required for analyze")
    results_dir = config_path.parent / Path(cfg.out_dir)
    runs = cfg.analysis.runs or []
    if not runs:
        latest = list_runs(cfg, config_path, stage="sample")
        if not latest:
            raise ValueError("No sample runs found. Run `cruncher sample <config>` first.")
        runs = [latest[0].name]
        logger.info("No analysis.runs configured; analyzing latest sample run: %s", runs[0])
    for batch_name in runs:
        sample_dir = results_dir / batch_name

        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Batch '{batch_name}' not found under {results_dir}")

        logger.info("Analyzing batch: %s", batch_name)
        manifest = load_manifest(sample_dir)
        if manifest.get("stage") != "sample":
            raise ValueError(f"Batch '{batch_name}' is not a sample run (stage={manifest.get('stage')})")

        # ── reconstruct PWMs from config_used.yaml for reproducibility ───────
        pwms, used_cfg = _load_pwms_from_config(sample_dir)
        tf_names = list(pwms.keys())
        used_sample = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
        bidirectional = cfg.sample.bidirectional
        if isinstance(used_sample, dict) and "bidirectional" in used_sample:
            bidirectional = bool(used_sample.get("bidirectional"))

        # ── locate newest elites parquet file ────────────────────────────────
        parquet_candidates = list(sample_dir.glob("cruncher_elites_*/*.parquet"))
        if not parquet_candidates:
            raise FileNotFoundError(f"No elites parquet found in {sample_dir}")
        latest_parquet = max(parquet_candidates, key=lambda p: p.stat().st_mtime)
        elites_df = pd.read_parquet(latest_parquet)
        logger.info("Loaded %d elites from %s", len(elites_df), latest_parquet.relative_to(sample_dir))

        # ── load trace & sequences for downstream plots ───────────────────────
        seq_path, trace_path = sample_dir / "sequences.parquet", sample_dir / "trace.nc"
        if not seq_path.exists():
            raise FileNotFoundError(
                f"Missing sequences.parquet in {sample_dir}. Re-run `cruncher sample` with sample.save_sequences=true."
            )
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Missing trace.nc in {sample_dir}. Re-run `cruncher sample` with sample.save_trace=true."
            )
        _ = pd.read_parquet(seq_path)
        trace_idata = az.from_netcdf(trace_path)

        # gather per-PWM scores (for scatter etc.)
        gather_per_pwm_scores(
            sample_dir,
            cfg.analysis.subsampling_epsilon,
            pwms,
            bidirectional=bidirectional,
            scale=cfg.analysis.scatter_scale,
        )
        logger.info("Wrote gathered_per_pwm_everyN.csv")

        # ── diagnostics ───────────────────────────────────────────────────────
        if cfg.analysis.plots.get("trace", False):
            plot_trace(trace_idata, sample_dir)
        if cfg.analysis.plots.get("autocorr", False):
            plot_autocorr(trace_idata, sample_dir)
        if cfg.analysis.plots.get("convergence", False):
            report_convergence(trace_idata, sample_dir)
            plot_rank_diagnostic(trace_idata, sample_dir)
            plot_ess(trace_idata, sample_dir)
        if any(cfg.analysis.plots.get(key, False) for key in ("trace", "autocorr", "convergence")):
            if len(tf_names) < 2:
                logger.warning("Skipping pairwise diagnostics: need at least two TFs (got %d).", len(tf_names))
            else:
                idata_pair = make_pair_idata(sample_dir, cfg, tf_names=tf_names)
                plot_pair_pwm_scores(idata_pair, sample_dir, cfg, tf_names=tf_names)
                plot_parallel_pwm_scores(idata_pair, sample_dir, cfg, tf_names=tf_names)

        # ── scatter plot ──────────────────────────────────────────────────────
        if cfg.analysis.plots.get("scatter_pwm", False):
            gathered_path = sample_dir / "gathered_per_pwm_everyN.csv"
            if not gathered_path.exists():
                raise FileNotFoundError(f"Missing gathered_per_pwm_everyN.csv in {sample_dir}")
            if len(tf_names) < 2:
                logger.warning("Skipping scatter plot: need at least two TFs (got %d).", len(tf_names))
            else:
                plot_scatter(sample_dir, pwms, cfg, tf_names=tf_names, bidirectional=bidirectional)
                logger.info("Wrote scatter_pwm.pdf")

        # ── top-5 tabular summary (unchanged logic – dynamic TF names) ────────
        if not elites_df.empty:
            if "rank" not in elites_df.columns:
                raise ValueError(
                    "Elites parquet missing 'rank' column. Re-run `cruncher sample` to regenerate elites.parquet."
                )
            if len(tf_names) >= 2:
                x_tf, y_tf = tf_names[:2]
                if f"score_{x_tf}" not in elites_df.columns or f"score_{y_tf}" not in elites_df.columns:
                    raise ValueError(
                        "Elites parquet missing score columns. Re-run `cruncher sample` to regenerate elites.parquet."
                    )

                logger.info("Top-5 elites (%s & %s):", x_tf, y_tf)
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    logger.info(
                        "rank=%d seq=%s %s=%.1f %s=%.1f",
                        int(row["rank"]),
                        row["sequence"],
                        x_tf,
                        row[f"score_{x_tf}"],
                        y_tf,
                        row[f"score_{y_tf}"],
                    )
            elif len(tf_names) == 1:
                x_tf = tf_names[0]
                if f"score_{x_tf}" not in elites_df.columns:
                    raise ValueError(
                        "Elites parquet missing score columns. Re-run `cruncher sample` to regenerate elites.parquet."
                    )
                logger.info("Top-5 elites (%s):", x_tf)
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    logger.info(
                        "rank=%d seq=%s %s=%.1f",
                        int(row["rank"]),
                        row["sequence"],
                        x_tf,
                        row[f"score_{x_tf}"],
                    )
            else:
                logger.warning("No regulators configured; skipping elite summary.")
