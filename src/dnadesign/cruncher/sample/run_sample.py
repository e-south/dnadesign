"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/run_sample.py

Run the MCMC-based sequence optimization, save enriched config, trace.nc, and a JSON of elites.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.parse.registry import Registry
from dnadesign.cruncher.sample.evaluator import SequenceEvaluator
from dnadesign.cruncher.sample.optimizer.cgm import GibbsOptimizer
from dnadesign.cruncher.sample.optimizer.pt import PTGibbsOptimizer
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.state import SequenceState, make_seed
from dnadesign.cruncher.utils.config import CruncherConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PWM_PATH = PROJECT_ROOT.parent / "dnadesign-data" / "primary_literature" / "OMalley_et_al" / "escherichia_coli_motifs"


def _save_config(cfg: CruncherConfig, batch_dir: Path) -> None:
    """
    Save the exact Pydantic-validated config into <batch_dir>/config_used.yaml,
    plus, for each TF:
      - file: path of the PWM file loaded
      - consensus: consensus sequence string
    """
    cfg_path = batch_dir / "config_used.yaml"
    with cfg_path.open("w") as fh:
        # 1. Dump base Pydantic-validated config as a dict
        data = json.loads(cfg.json())

        # 2. Reconstruct Registry to get actual PWM file paths & compute consensus
        reg = Registry(PWM_PATH, cfg.parse.formats)
        flat_tfs = {tf for group in cfg.regulator_sets for tf in group}
        pwms_info: dict[str, dict[str, str]] = {}
        for tf in flat_tfs:
            pwm: PWM = reg.load(tf)
            # Assume Registry provides a method to retrieve the file path
            pwm_path: Path = reg.get_path(tf)
            # Build consensus: argmax over each column of the PWM matrix
            cons_vec = np.argmax(pwm.matrix, axis=1)
            consensus = "".join("ACGT"[i] for i in cons_vec)

            pwms_info[tf] = {
                "file": str(pwm_path),
                "consensus": consensus,
            }

        data["pwms_info"] = pwms_info
        yaml.safe_dump({"cruncher": data}, fh)


def run_sample(cfg: CruncherConfig, out_dir: Path) -> None:
    """
    Run MCMC sampler, save enriched config, trace.nc, and elites.json.
    """
    sample_cfg = cfg.sample
    rng = np.random.default_rng()

    # 1) LOAD all required PWMs
    reg = Registry(PWM_PATH, cfg.parse.formats)
    flat_tfs = {tf for group in cfg.regulator_sets for tf in group}
    pwms: dict[str, PWM] = {tf: reg.load(tf) for tf in flat_tfs}

    # 2) INSTANTIATE SCORER and SequenceEvaluator
    scorer = Scorer(
        pwms,
        bidirectional=sample_cfg.bidirectional,
        scale=sample_cfg.scorer_scale,
        background=(0.25, 0.25, 0.25, 0.25),
        penalties=sample_cfg.penalties,
    )
    evaluator = SequenceEvaluator(scorer, reducer=min)

    # 3) INITIALIZE seed sequence
    initial: SequenceState = make_seed(sample_cfg.init, pwms, rng)

    # 4) FLATTEN optimiser config for Gibbs/PT
    optblock = sample_cfg.optimiser
    opt_cfg: dict[str, object] = {
        "draws": sample_cfg.draws,
        "tune": sample_cfg.tune,
        "chains": sample_cfg.chains,
        "min_dist": sample_cfg.min_dist,
        "top_k": sample_cfg.top_k,
        **sample_cfg.moves.dict(),
        **optblock.cooling.dict(),
        "swap_prob": optblock.swap_prob,
    }
    if optblock.kind == "pt":
        opt_cfg["softmax_beta"] = optblock.softmax_beta  # guaranteed non-None

    # 5) RUN optimizer (Gibbs or PT)
    if sample_cfg.optimiser.kind == "gibbs":
        optimizer = GibbsOptimizer(evaluator, opt_cfg, rng)
    else:
        optimizer = PTGibbsOptimizer(
            evaluator,
            opt_cfg,
            rng,
            pwms=pwms,
            init_cfg=sample_cfg.init,
        )

    print("Starting MCMC sampling …", flush=True)
    ranked_states = optimizer.optimise(initial)
    print("MCMC sampling complete.")

    # 6) SAVE enriched config, trace.nc, and build elites.json
    _save_config(cfg, out_dir)

    # Save trace.nc (ArviZ InferenceData) if available
    if hasattr(optimizer, "trace_idata") and optimizer.trace_idata is not None:
        from dnadesign.cruncher.utils.traces import save_trace

        save_trace(optimizer.trace_idata, out_dir / "trace.nc")
        print(f"Saved MCMC trace → {(out_dir / 'trace.nc').relative_to(out_dir.parent)}")

    # Build elites.json with top-K sequences and per-TF metadata
    elites_data: list[dict[str, object]] = []
    for rank, state in enumerate(ranked_states[: sample_cfg.top_k], start=1):
        seq_str = state.to_string()
        per_tf: dict[str, dict[str, object]] = {}

        for tf, info in scorer._cache.items():
            # New method returns (raw_llr, extras, offset, strand)
            raw_llr, extras, offset, strand = scorer._best_llr_and_extra_hits_and_location(state.seq, info)
            L = len(state.seq)

            # Compute scaled score according to selected scale
            if scorer.scale == "llr":
                scaled_val = float(raw_llr)
            elif scorer.scale in ("p", "logp"):
                # _scale_other returns -log10(p), so invert if “p”
                s_nonlog = scorer._scale_other(raw_llr, info, seq_length=L)
                scaled_val = float((10 ** (-s_nonlog)) if scorer.scale == "p" else s_nonlog)
            elif scorer.scale == "z":
                scaled_val = float(scorer._scale_other(raw_llr, info, seq_length=L))
            else:  # "logp_norm"
                scaled_val = float(scorer._scale_llr(raw_llr, info, seq_length=L))

            w = info.width
            meme_fmt = f"{offset}_[{strand}]{w}"

            per_tf[tf] = {
                "raw_llr": float(raw_llr),
                "extras": int(extras),
                "offset": int(offset),
                "strand": strand,
                "meme_format": meme_fmt,
                "scaled_score": scaled_val,
            }

        elites_data.append(
            {
                "rank": rank,
                "sequence": seq_str,
                "per_tf": per_tf,
            }
        )

    with open(out_dir / "elites.json", "w") as fh:
        json.dump(elites_data, fh, indent=2)
    print(f"Saved top-{sample_cfg.top_k} elites → {(out_dir / 'elites.json').relative_to(out_dir.parent)}")
