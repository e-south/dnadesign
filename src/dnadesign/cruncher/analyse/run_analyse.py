"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/analyse/run_analyse.py

When run_analyse is called, it iterates through cfg.analysis.runs (a list of
existing “sample” batch names), and for each batch:
  1) It looks under the “results” folder for that sample's directory.
  2) It writes all new “gathered_per_pwm_…”, trace_score.png, autocorr/convergence,
     and scatter_pwm.png files directly into the same sample folder.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import pandas as pd

from dnadesign.cruncher.analyse.per_pwm import gather_per_pwm_scores
from dnadesign.cruncher.analyse.plots.scatter import plot_scatter
from dnadesign.cruncher.analyse.plots.trace_autocorr_convergence import (
    plot_autocorr,
    plot_trace,
    report_convergence,
)
from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.parse.registry import Registry
from dnadesign.cruncher.utils.config import CruncherConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "src" / "dnadesign" / "cruncher" / "results"
PWM_PATH = PROJECT_ROOT.parent / "dnadesign-data" / "primary_literature" / "OMalley_et_al" / "escherichia_coli_motifs"


def run_analyse(cfg: CruncherConfig) -> None:
    """
    For each sample-batch name in cfg.analysis.runs:
      1) Locate <RESULTS_DIR>/<batch_name>.
      2) If sequences.csv and trace.nc exist, run gather_per_pwm_scores(...) → gathered_per_pwm_everyN.csv.
      3) Depending on cfg.analysis.plots, produce:
         - trace_score.png
         - autocorr_score.png
         - convergence.txt
         - scatter_pwm.png
       All outputs are saved directly under <RESULTS_DIR>/<batch_name>.
    """
    # 1) Build a shared PWM registry (only once)
    reg = Registry(PWM_PATH, cfg.parse.formats)
    flat_tfs = {tf for group in cfg.regulator_sets for tf in group}
    pwms: dict[str, PWM] = {tf: reg.load(tf) for tf in sorted(flat_tfs)}

    # Iterate over all batch names specified in cfg.analysis.runs
    for batch_name in cfg.analysis.runs or []:
        sample_dir = RESULTS_DIR / batch_name

        if not sample_dir.is_dir():
            print(f"[analyse] Warning: sample batch '{batch_name}' not found under {RESULTS_DIR}")
            continue

        print(f"[analyse] Re-analysing batch: {batch_name}")

        elites_path = sample_dir / "elites.json"
        seq_path = sample_dir / "sequences.csv"
        trace_path = sample_dir / "trace.nc"

        # 2) Load elites.json (to know which sequences are “elites”)
        if not elites_path.exists():
            print(f"[analyse:{batch_name}] skip—elites.json not found.")
            continue
        df_elites = pd.read_json(elites_path)

        # 3) Load sequences.csv (if present) and trace.nc (if present)
        seq_df = pd.read_csv(seq_path) if seq_path.exists() else pd.DataFrame()
        if trace_path.exists():
            try:
                trace_idata = az.from_netcdf(trace_path)
            except Exception as e:
                raise RuntimeError(f"[analyse:{batch_name}] failed to load trace.nc as InferenceData: {e!r}")
        else:
            trace_idata = None

        # 4) gather_per_pwm (only if both sequences.csv and trace.nc exist)
        if seq_df.empty or trace_idata is None:
            print(f"[analyse:{batch_name}] skip gather_per_pwm—missing sequences.csv or valid trace.nc.")
        else:
            gather_per_pwm_scores(
                sample_dir,
                cfg.analysis.gather_nth_iteration_for_scaling,
                pwms,
                bidirectional=cfg.sample.bidirectional,
                penalties=cfg.sample.penalties,
                scale=cfg.analysis.scatter_scale,
            )

        # 5) Generate any requested plots directly into sample_dir
        for plot_name, do_plot in cfg.analysis.plots.items():
            if not do_plot:
                continue

            if plot_name == "trace":
                if trace_idata is None:
                    print(f"[analyse:{batch_name}] skip 'trace'—trace.nc missing or failed to load.")
                    continue
                plot_trace(trace_idata, sample_dir)
                print(f"[analyse:{batch_name}] Wrote trace_score.png")

            elif plot_name == "autocorr":
                if trace_idata is None:
                    print(f"[analyse:{batch_name}] skip 'autocorr'—trace.nc missing or failed to load.")
                    continue
                plot_autocorr(trace_idata, sample_dir)
                print(f"[analyse:{batch_name}] Wrote autocorr_score.png")

            elif plot_name == "convergence":
                if trace_idata is None:
                    print(f"[analyse:{batch_name}] skip 'convergence'—trace.nc missing or failed to load.")
                    continue
                report_convergence(trace_idata, sample_dir)
                print(f"[analyse:{batch_name}] Wrote convergence.txt")

            elif plot_name == "scatter_pwm":
                gathered_path = sample_dir / "gathered_per_pwm_everyN.csv"
                if not gathered_path.exists() or df_elites.empty:
                    print(
                        f"[analyse:{batch_name}] skip 'scatter_pwm'—"
                        "missing gathered_per_pwm_everyN.csv or empty elites.json."
                    )
                    continue
                plot_scatter(sample_dir, pwms, cfg)
                print(f"[analyse:{batch_name}] Wrote scatter_pwm.png")

            else:
                print(f"[analyse:{batch_name}] Unknown plot '{plot_name}'—skipping.")
