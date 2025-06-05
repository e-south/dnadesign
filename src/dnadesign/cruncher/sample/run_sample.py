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

import csv
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.parse.registry import Registry
from dnadesign.cruncher.sample.evaluator import SequenceEvaluator
from dnadesign.cruncher.sample.optimizer.cgm import GibbsOptimizer
from dnadesign.cruncher.sample.optimizer.pt import PTGibbsOptimizer
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.state import SequenceState
from dnadesign.cruncher.utils.config import CruncherConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PWM_PATH = PROJECT_ROOT.parent / "dnadesign-data" / "primary_literature" / "OMalley_et_al" / "escherichia_coli_motifs"


def _save_config(cfg: CruncherConfig, batch_dir: Path) -> None:
    """
    Save the exact Pydantic-validated config into <batch_dir>/config_used.yaml,
    plus, for each TF:
      - alphabet: ["A","C","G","T"]
      - pwm_matrix: a list of [p_A, p_C, p_G, p_T] for each position
      - consensus: consensus sequence string
    """
    cfg_path = batch_dir / "config_used.yaml"
    data = json.loads(cfg.json())

    reg = Registry(PWM_PATH, cfg.parse.formats)
    flat_tfs = {tf for group in cfg.regulator_sets for tf in group}
    pwms_info: dict[str, dict[str, object]] = {}

    logger.debug("Saving PWM info for config_used.yaml")
    for tf in sorted(flat_tfs):
        pwm: PWM = reg.load(tf)

        # Build consensus: argmax over each column
        cons_vec = np.argmax(pwm.matrix, axis=1)
        consensus = "".join("ACGT"[i] for i in cons_vec)

        pwm_probs_rounded: list[list[float]] = []
        for row in pwm.matrix:
            rounded_row = [round(float(p), 6) for p in row]
            pwm_probs_rounded.append(rounded_row)

        pwms_info[tf] = {
            "alphabet": ["A", "C", "G", "T"],
            "pwm_matrix": pwm_probs_rounded,
            "consensus": consensus,
        }

    data["pwms_info"] = pwms_info
    with cfg_path.open("w") as fh:
        yaml.safe_dump({"cruncher": data}, fh, sort_keys=False, default_flow_style=False)
    logger.info("Wrote config_used.yaml to %s", cfg_path.relative_to(batch_dir.parent))


def run_sample(cfg: CruncherConfig, out_dir: Path) -> None:
    """
    Run MCMC sampler, save enriched config, trace.nc, sequences.csv (including per-TF scores),
    and elites.json.  Each chain now gets its own independent seed (random/consensus/consensus_mix).
    """
    sample_cfg = cfg.sample
    rng = np.random.default_rng()

    logger.info("=== RUN SAMPLE: %s ===", out_dir)
    logger.debug("Full sample config: %s", sample_cfg.json())

    # 1) LOAD all required PWMs
    logger.debug("Loading PWMs from %s", PWM_PATH)
    reg = Registry(PWM_PATH, cfg.parse.formats)
    flat_tfs = {tf for group in cfg.regulator_sets for tf in group}
    pwms: dict[str, PWM] = {}
    for tf in sorted(flat_tfs):
        logger.debug("  Loading PWM for %s", tf)
        pwms[tf] = reg.load(tf)

    # 2) INSTANTIATE SCORER and SequenceEvaluator
    scale = sample_cfg.optimiser.scorer_scale
    logger.info("Using scorer_scale = %r", scale)
    if scale == "consensus-neglop-sum":
        logger.debug("Building Scorer with inner scale='logp' in order to feed SequenceEvaluator(consensus-neglop-sum)")
        scorer = Scorer(
            pwms,
            bidirectional=sample_cfg.bidirectional,
            scale="logp",  # produce pure neglogp per PWM
            background=(0.25, 0.25, 0.25, 0.25),
        )
        evaluator = SequenceEvaluator(
            pwms=pwms,
            scale="consensus-neglop-sum",
            combiner=lambda vs: sum(vs),
        )
    else:
        logger.debug("Building Scorer and SequenceEvaluator with scale=%r", scale)
        scorer = Scorer(
            pwms,
            bidirectional=sample_cfg.bidirectional,
            scale=scale,
            background=(0.25, 0.25, 0.25, 0.25),
        )
        evaluator = SequenceEvaluator(
            pwms=pwms,
            scale=scale,
            combiner=None,  # defaults to min(...) if scale ∈ {"llr","logp"}
        )

    logger.info("Scorer and SequenceEvaluator instantiated")
    logger.debug("  Scorer.scale = %r", scorer.scale)
    logger.debug("  SequenceEvaluator.scale = %r", evaluator._scale)
    logger.debug("  SequenceEvaluator.combiner = %r", evaluator._combiner)

    # 3) FLATTEN optimiser config for Gibbs/PT
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
        opt_cfg["softmax_beta"] = optblock.softmax_beta

    if optblock.cooling.kind == "geometric":
        logger.info(
            "β-ladder (%d levels): %s",
            len(optblock.cooling.beta),
            ", ".join(f"{b:g}" for b in optblock.cooling.beta),
        )

    logger.debug("Optimizer config flattened: %s", opt_cfg)

    # 4) INSTANTIATE OPTIMIZER (Gibbs or PT), passing in init_cfg and pwms
    if sample_cfg.optimiser.kind == "gibbs":
        logger.info("Instantiating GibbsOptimizer")
        optimizer = GibbsOptimizer(
            evaluator=evaluator,
            cfg=opt_cfg,
            rng=rng,
            init_cfg=sample_cfg.init,
            pwms=pwms,
        )
    else:
        logger.info("Instantiating PTGibbsOptimizer")
        optimizer = PTGibbsOptimizer(
            evaluator=evaluator,
            cfg=opt_cfg,
            rng=rng,
            pwms=pwms,
            init_cfg=sample_cfg.init,
        )

    # 5) RUN the MCMC sampling (no separate “initial seed” is recorded here)
    logger.info("Starting MCMC sampling …")
    ranked_states = optimizer.optimise()
    logger.info("MCMC sampling complete.")

    # 6) SAVE enriched config
    _save_config(cfg, out_dir)

    # 7) SAVE trace.nc
    if hasattr(optimizer, "trace_idata") and optimizer.trace_idata is not None:
        from dnadesign.cruncher.utils.traces import save_trace

        save_trace(optimizer.trace_idata, out_dir / "trace.nc")
        logger.info("Saved MCMC trace → %s", (out_dir / "trace.nc").relative_to(out_dir.parent))

    # 8) SAVE sequences.csv (chain, draw, phase, sequence_string, per-TF scaled scores)
    if (
        sample_cfg.save_sequences
        and hasattr(optimizer, "all_samples")
        and hasattr(optimizer, "all_meta")
        and hasattr(optimizer, "all_scores")
    ):
        seq_csv = out_dir / "sequences.csv"
        with seq_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)

            # Header with “phase” column (either “tune” or “draw”)
            header = ["chain", "draw", "phase", "sequence"] + [f"score_{tf}" for tf in sorted(pwms.keys())]
            writer.writerow(header)

            for (chain_id, draw_i), seq_arr, per_tf_map in zip(
                optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
            ):
                # Decide phase by comparing draw_i to tune
                if draw_i < sample_cfg.tune:
                    phase = "tune"
                    draw_i_to_write = draw_i
                else:
                    phase = "draw"
                    draw_i_to_write = draw_i  # >= tune

                seq_str = SequenceState(seq_arr).to_string()
                row = [
                    int(chain_id),
                    int(draw_i_to_write),
                    phase,
                    seq_str,
                ] + [per_tf_map[tf] for tf in sorted(per_tf_map.keys())]
                writer.writerow(row)

        logger.info("Saved all sequences with per-TF scores → %s", seq_csv.relative_to(out_dir.parent))

    # 9) BUILD elites.json, tagging each with (chain, draw_index)
    elites_data: list[dict[str, object]] = []
    for rank, (state, (chain_id, draw_i)) in enumerate(zip(ranked_states, optimizer.elites_meta), start=1):
        seq_str = state.to_string()
        per_tf_scaled = evaluator(state)  # Dict[tf → scaled_value]

        per_tf: dict[str, dict[str, object]] = {}
        for tf, info in scorer._cache.items():
            raw_llr, offset, strand = scorer._best_llr_and_location(state.seq, info)
            scaled_val = float(per_tf_scaled[tf])

            w = info.width
            start_pos = offset + 1
            strand_label = f"{strand}1"
            motif_diagram = f"{start_pos}_[{strand_label}]_{w}"

            per_tf[tf] = {
                "raw_llr": float(raw_llr),
                "offset": int(offset),
                "strand": strand,
                "motif_diagram": motif_diagram,
                "scaled_score": scaled_val,
            }

        elites_data.append(
            {
                "rank": rank,
                "chain": int(chain_id),
                "draw_index": int(draw_i),
                "sequence": seq_str,
                "per_tf": per_tf,
            }
        )

    elites_path = out_dir / "elites.json"
    with elites_path.open("w") as fh:
        json.dump(elites_data, fh, indent=2)
    logger.info("Saved top-%d elites → %s", sample_cfg.top_k, elites_path.relative_to(out_dir.parent))
