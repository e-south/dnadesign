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
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)

import sys
from pathlib import Path
from datetime import datetime
import json
import yaml
import numpy as np

import arviz as az
from dnadesign.cruncher.config import load_config, CruncherConfig
from dnadesign.cruncher.motif.registry import Registry
from dnadesign.cruncher.plots.pssm import plot_pwm
from dnadesign.cruncher.plots.arviz import mcmc_diagnostics, scatter_pwm
from dnadesign.cruncher.sample.state import SequenceState
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.optimizer.cgm import GibbsOptimizer
from dnadesign.cruncher.motif.model import PWM

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CFG_PATH     = PROJECT_ROOT / "dnadesign" / "src" / "dnadesign" / "configs" / "example.yaml"
PWM_PATH     = PROJECT_ROOT / "dnadesign-data" / "primary_literature" / "OMalley_et_al" / "escherichia_coli_motifs"


def _run_parse(cfg: CruncherConfig, base_out: Path) -> None:
    reg = Registry(PWM_PATH, cfg.motif.formats)
    out_base = base_out / "motif_logos"
    out_base.mkdir(parents=True, exist_ok=True)

    unique_regs = set(sum(cfg.regulator_sets, []))
    for tf in sorted(unique_regs):
        pwm = reg.load(tf)
        plot_pwm(
            pwm,
            mode=cfg.motif.plot.bits_mode,
            out=out_base / f"{tf}_logo.png",
            dpi=cfg.motif.plot.dpi,
        )
        print(f"[OK] {tf:8s} L={pwm.length:2d} bits={pwm.information_bits():4.1f}")

    print("Parse stage complete.")


def _initialize_state(init_cfg, pwms: dict[str, PWM], rng) -> SequenceState:
    # determine target length
    if isinstance(init_cfg.kind, int):
        length = init_cfg.kind
    else:
        max_len = max(p.length for p in pwms.values())
        if init_cfg.kind == 'random':
            length = max_len
        elif init_cfg.kind == 'consensus_shortest':
            length = min(p.length for p in pwms.values())
        elif init_cfg.kind == 'consensus_longest':
            length = max_len
        else:
            raise ValueError(f"Unknown init kind: {init_cfg.kind}")

    if init_cfg.kind == 'random' or isinstance(init_cfg.kind, int):
        return SequenceState.random(length, rng)

    mode = init_cfg.kind.split('_')[1]  # 'shortest' or 'longest'
    return SequenceState.from_consensus(
        pwms,
        mode=mode,
        target_length=length,
        pad_with=init_cfg.pad_with,
        rng=rng,
    )


def _save_config(cfg: CruncherConfig, batch_dir: Path) -> None:
    """
    Dump out the exact config used, converting all Paths to strings
    via JSON round-trip before writing YAML.
    """
    cfg_path = batch_dir / "config_used.yaml"
    with cfg_path.open('w') as fh:
        # JSON‐serialize then reload gives us pure dicts/lists/primitives
        data = json.loads(cfg.json())
        yaml.safe_dump({'cruncher': data}, fh)


def _save_hits(
    ranked: list[SequenceState],
    scorer: Scorer,
    pwms: dict[str, PWM],
    batch_dir: Path,
    top_k: int,
) -> None:
    import pandas as pd
    hits = []
    for rank, state in enumerate(ranked[:top_k], start=1):
        row = {'rank': rank, 'sequence': state.seq_str()}
        per_pwm = scorer.score_per_pwm(state.seq_array())
        for tf, score in zip(pwms, per_pwm):
            row[f'score_{tf}'] = float(score)
        hits.append(row)
    df = pd.DataFrame(hits)
    df.to_csv(batch_dir / 'hits.csv', index=False)


def _run_sample(cfg: CruncherConfig, base_out: Path) -> None:
    assert cfg.sample is not None, "sample config is required"
    sample_cfg = cfg.sample
    rng = np.random.default_rng()

    # 1. prepare batch dir
    batch_dir = base_out
    (batch_dir / 'plots').mkdir(parents=True, exist_ok=True)

    # 2. save config
    _save_config(cfg, batch_dir)

    # 3. load PWMs
    reg = Registry(PWM_PATH, cfg.motif.formats)
    pwms = {tf: reg.load(tf) for tf in set(sum(cfg.regulator_sets, []))}

    # 4. initialize
    initial = _initialize_state(sample_cfg.init, pwms, rng)

    # 5. set up scorer + optimizer
    scorer = Scorer(pwms)
    gib_cfg = sample_cfg.optimiser.gibbs.dict()
    gib_cfg['top_k']   = sample_cfg.top_k     # pass top_k in
    gib_cfg['min_dist'] = sample_cfg.optimiser.gibbs.min_dist
    optimizer = GibbsOptimizer(scorer, gib_cfg, rng)

    # 6. run
    print("Starting MCMC sampling…")
    ranked_states = optimizer.optimise(initial)
    print("Sampling complete.")

    # 7. save hits
    _save_hits(ranked_states, scorer, pwms, batch_dir, sample_cfg.top_k)
    print(f"Saved top-{sample_cfg.top_k} hits to hits.csv")

    # 8. (optional) diagnostics — only if you have a trace file
    trace_path = batch_dir / 'trace.nc'
    if trace_path.exists():
        idata = az.from_netcdf(trace_path)
        mcmc_diagnostics(idata, batch_dir / 'plots')
        scatter_pwm(idata, batch_dir / 'plots', pwms=list(pwms.keys()))
        print("Generated diagnostic plots.")
    else:
        logger = logging.getLogger(__name__)
        logger.info("No trace.nc found; skipping ArviZ diagnostics.")

    # 9. write a README
    readme = batch_dir / 'README.txt'
    with readme.open('w') as fh:
        fh.write(f"Batch run: {datetime.now().isoformat()}\n")
        fh.write(f"Regulator sets: {cfg.regulator_sets}\n")
        fh.write(f"Initialization: {sample_cfg.init!r}\n")
    print("Sample stage complete.")


def _run_analyse(cfg: CruncherConfig, base_out: Path) -> None:
    print("Analyse stage not yet implemented.")


def main(cfg_path: str | Path | None = None) -> None:
    cfg_file = Path(cfg_path) if cfg_path else CFG_PATH
    cfg      = load_config(cfg_file)

    ts = datetime.now().strftime("%Y%m%dT%H%M")
    batch = PROJECT_ROOT / "dnadesign" / "src" / "dnadesign" / "cruncher" / cfg.out_dir / f"batch_{ts}"
    batch.mkdir(parents=True, exist_ok=True)

    match cfg.mode:
        case 'parse':
            _run_parse(cfg, batch)
        case 'sample':
            _run_sample(cfg, batch)
        case m:
            raise ValueError(f"Unknown mode '{m}'")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)