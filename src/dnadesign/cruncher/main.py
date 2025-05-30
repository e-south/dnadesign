"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/main.py

Main entry point for the cruncher package.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.parse.plots.pssm import plot_pwm
from dnadesign.cruncher.parse.registry import Registry
from dnadesign.cruncher.sample.optimizer.cgm import GibbsOptimizer
from dnadesign.cruncher.sample.plots.autocorr import plot_autocorr
from dnadesign.cruncher.sample.plots.convergence import report_convergence
from dnadesign.cruncher.sample.plots.scatter_pwm import plot_scatter_pwm
from dnadesign.cruncher.sample.plots.trace import plot_trace
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.state import SequenceState
from dnadesign.cruncher.utils.config import CruncherConfig, load_config
from dnadesign.cruncher.utils.traces import save_trace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CFG_PATH = PROJECT_ROOT / "dnadesign" / "src" / "dnadesign" / "cruncher" / "config.yaml"
PWM_PATH = PROJECT_ROOT / "dnadesign-data" / "primary_literature" / "OMalley_et_al" / "escherichia_coli_motifs"


def _run_parse(cfg: CruncherConfig, base_out: Path) -> None:
    reg = Registry(PWM_PATH, cfg.parse.formats)
    out_base = base_out / "motif_logos"
    out_base.mkdir(parents=True, exist_ok=True)

    unique_regs = set(sum(cfg.regulator_sets, []))
    for tf in sorted(unique_regs):
        pwm = reg.load(tf)
        plot_pwm(
            pwm,
            mode=cfg.parse.plot.bits_mode,
            out=out_base / f"{tf}_logo.png",
            dpi=cfg.parse.plot.dpi,
        )
        length = pwm.length
        bits = pwm.information_bits()
        print(f"[OK] {tf:8s} L={length:2d} bits={bits:4.1f}")

        log_odds = pwm.log_odds()
        df_lo = pd.DataFrame(
            log_odds,
            columns=list(pwm.alphabet),
            index=[f"pos{i+1}" for i in range(log_odds.shape[0])],
        )
        print(f"Log-odds matrix for {tf}:")
        print(df_lo.to_string(float_format=lambda x: f"{x:.3f}"))
        print()

    print("Parse stage complete.")


def _initialize_state(init_cfg, pwms: dict[str, PWM], rng) -> SequenceState:
    if isinstance(init_cfg.kind, int):
        length = max(init_cfg.kind, max(p.length for p in pwms.values()))
    elif init_cfg.kind == "random":
        length = max(p.length for p in pwms.values())
    else:
        raise ValueError("init.kind must be int or 'random' or consensus")
    return SequenceState.random(length, rng)


def _save_config(cfg: CruncherConfig, batch_dir: Path) -> None:
    cfg_path = batch_dir / "config_used.yaml"
    with cfg_path.open("w") as fh:
        data = json.loads(cfg.json())
        yaml.safe_dump({"cruncher": data}, fh)


def _save_hits(
    ranked: list[SequenceState],
    scorer: Scorer,
    pwms: dict[str, PWM],
    batch_dir: Path,
    top_k: int,
) -> None:
    hits = []
    for rank, state in enumerate(ranked[:top_k], start=1):
        row = {"rank": rank, "sequence": state.seq_str()}
        per_pwm = scorer.score_per_pwm(state.seq_array())
        for tf, score in zip(pwms, per_pwm):
            row[f"score_{tf}"] = float(score)
        hits.append(row)
    pd.DataFrame(hits).to_csv(batch_dir / "hits.csv", index=False)


def _run_sample(cfg: CruncherConfig, base_out: Path) -> None:
    assert cfg.sample is not None, "sample config is required"
    sample_cfg = cfg.sample
    rng = np.random.default_rng()

    sample_dir = base_out / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    _save_config(cfg, sample_dir)

    reg = Registry(PWM_PATH, cfg.parse.formats)
    pwms = {tf: reg.load(tf) for tf in set(sum(cfg.regulator_sets, []))}

    initial = _initialize_state(sample_cfg.init, pwms, rng)

    scorer = Scorer(pwms, bidirectional=sample_cfg.bidirectional)
    gib_cfg = sample_cfg.optimiser.gibbs.dict()
    gib_cfg["top_k"] = sample_cfg.top_k
    gib_cfg["min_dist"] = sample_cfg.optimiser.gibbs.min_dist
    gib_cfg["random_init"] = sample_cfg.init.kind == "random"
    optimizer = GibbsOptimizer(scorer, gib_cfg, rng)

    print("Starting MCMC sampling…")
    ranked_states = optimizer.optimise(initial)
    print("Sampling complete.")

    _save_hits(ranked_states, scorer, pwms, sample_dir, sample_cfg.top_k)
    print(f"Saved top-{sample_cfg.top_k} hits to {sample_dir/'hits.csv'}")

    if hasattr(optimizer, "samples_df"):
        optimizer.samples_df.to_csv(sample_dir / "samples.csv", index=False)
        print("Saved samples.csv for scatter diagnostics")

    n_ref = len(optimizer.samples_df) if hasattr(optimizer, "samples_df") else sample_cfg.optimiser.gibbs.draws
    L = initial.seq.size
    ref_rows = []
    rng2 = np.random.default_rng(0)
    for i in range(n_ref):
        x_rand = SequenceState.random(L, rng2)
        scores = scorer.score_per_pwm(x_rand.seq)
        row = {"iter": i}
        for tf, sc in zip(pwms, scores):
            row[f"score_{tf}"] = float(sc)
        ref_rows.append(row)
    pd.DataFrame(ref_rows).to_csv(sample_dir / "random_samples.csv", index=False)

    trace_path = sample_dir / "trace.nc"
    idata = getattr(optimizer, "trace_idata", None)
    if idata is not None:
        save_trace(idata, trace_path)
        plots_cfg = sample_cfg.plots
        if plots_cfg.get("trace", False):
            plot_trace(idata, sample_dir)
            print("  • trace_score.png")
        if plots_cfg.get("autocorr", False):
            plot_autocorr(idata, sample_dir)
            print("  • autocorr_score.png")
        if plots_cfg.get("convergence", False):
            report_convergence(idata, sample_dir)
            print("  • convergence.txt")
        if plots_cfg.get("scatter_pwm", False):
            plot_scatter_pwm(sample_dir, pwms, cfg)
            print("  • scatter_pwm.png")
        print("Generated MCMC diagnostics.")
    else:
        logging.getLogger(__name__).info("No trace data found on optimizer; skipping MCMC diagnostics.")


def _run_analyse(cfg: CruncherConfig, base_out: Path) -> None:
    print("Analyse stage not yet implemented.")


def main(cfg_path: str | Path | None = None) -> None:
    cfg_file = Path(cfg_path) if cfg_path else CFG_PATH
    cfg = load_config(cfg_file)

    # timestamp and batch naming
    ts = datetime.now().strftime("%Y%m%dT%H%M")
    regs = [r for group in cfg.regulator_sets for r in group]
    regs_label = "-".join(regs)
    batch_name = f"{cfg.mode}_{regs_label}_{ts}"
    batch = PROJECT_ROOT / "dnadesign" / "src" / "dnadesign" / "cruncher" / cfg.out_dir / batch_name
    batch.mkdir(parents=True, exist_ok=True)

    match cfg.mode:
        case "parse":
            _run_parse(cfg, batch)
        case "sample":
            _run_sample(cfg, batch)
        case other:
            raise ValueError(f"Unknown mode '{other}'")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
