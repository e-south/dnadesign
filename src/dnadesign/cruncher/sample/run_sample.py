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
from dnadesign.cruncher.sample.state import SequenceState, make_seed
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
    and elites.json.
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

    # 3) INITIALIZE seed sequence
    logger.info("Initializing seed sequence (kind=%r)", sample_cfg.init.kind)
    initial: SequenceState = make_seed(sample_cfg.init, pwms, rng)
    logger.debug("  Seed sequence (length=%d): %s", len(initial), initial.to_string())

    # DEBUG: Which init and what scores
    logger.debug(
        "=== DEBUG: init.kind = %s ; init.regulator = %s",
        sample_cfg.init.kind,
        getattr(sample_cfg.init, "regulator", None),
    )
    logger.debug("    → SequenceState length: %d", len(initial))
    logger.debug("    → SequenceState string: %s", initial.to_string())

    raw_scores = evaluator(initial)
    logger.debug("=== DEBUG: SequenceEvaluator returns per-TF values: %s", raw_scores)

    sc = evaluator._scorer
    for tf_name, info in sc._cache.items():
        raw_llr, offset, strand = sc._best_llr_and_location(initial.seq, info)
        neglogp = sc._per_pwm_neglogp(raw_llr, info, len(initial))
        logger.debug(
            "→ PWM %-5s  raw_llr = %8.3f,  neglog10(p_seq) = %8.3f,  offset,strand = (%d,%s)",
            tf_name,
            raw_llr,
            neglogp,
            offset,
            strand,
        )

    logger.debug("=== DEBUG: SequenceEvaluator scale = %r", evaluator._scale)
    logger.debug("    → combiner object: %r", evaluator._combiner)
    logger.debug("--- Now instantiating optimizer, combined(seed) will be shown once beta scheduler is available ---")

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
        opt_cfg["softmax_beta"] = optblock.softmax_beta

    logger.debug("Optimizer config flattened: %s", opt_cfg)

    # 5) INSTANTIATE OPTIMIZER (Gibbs or PT)
    if sample_cfg.optimiser.kind == "gibbs":
        logger.info("Instantiating GibbsOptimizer")
        optimizer = GibbsOptimizer(evaluator, opt_cfg, rng)
    else:
        logger.info("Instantiating PTGibbsOptimizer")
        optimizer = PTGibbsOptimizer(
            evaluator,
            opt_cfg,
            rng,
            pwms=pwms,
            init_cfg=sample_cfg.init,
        )

    # Now that optimizer exists, we can compute beta_final and show combined(seed)
    total_sweeps = sample_cfg.tune + sample_cfg.draws
    beta_final = optimizer.beta_of(total_sweeps - 1) if total_sweeps > 0 else optimizer.beta_of(0)
    combined_seed = evaluator.combined(initial, beta=beta_final)
    logger.debug("=== DEBUG: combined(seed) at beta_final=%.3f  → %.6f", beta_final, combined_seed)
    logger.debug("--- Now entering optimizer.optimise(...) ---")

    # 6) (REMOVED) We no longer explicitly record the seed here,
    #    because GibbsOptimizer.optimise(...) already pushes the initial state
    #    (draw_index = -1) into optimizer.all_samples, all_meta, and all_scores.

    logger.info("Starting MCMC sampling …")
    ranked_states = optimizer.optimise(initial)
    logger.info("MCMC sampling complete.")

    # 7) SAVE enriched config
    _save_config(cfg, out_dir)

    # 8) SAVE trace.nc
    if hasattr(optimizer, "trace_idata") and optimizer.trace_idata is not None:
        from dnadesign.cruncher.utils.traces import save_trace

        save_trace(optimizer.trace_idata, out_dir / "trace.nc")
        logger.info("Saved MCMC trace → %s", (out_dir / "trace.nc").relative_to(out_dir.parent))

    # 9) SAVE sequences.csv (chain, draw, phase, sequence_string, per-TF scaled scores)
    if (
        sample_cfg.save_sequences
        and hasattr(optimizer, "all_samples")
        and hasattr(optimizer, "all_meta")
        and hasattr(optimizer, "all_scores")
    ):
        # We will re-index negative (burn-in) draws to [0 .. tune-1] and sampling draws to [tune .. tune+draws-1],
        # and add a "phase" column indicating "tune" vs "draw".
        total_tune = sample_cfg.tune

        seq_csv = out_dir / "sequences.csv"
        with seq_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)

            # New header includes a "phase" column
            header = ["chain", "draw", "phase", "sequence"] + [f"score_{tf}" for tf in sorted(pwms.keys())]
            writer.writerow(header)

            for (chain_id, raw_draw_i), seq_arr, per_tf_map in zip(
                optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
            ):
                # Decide phase and re-index the draw number
                if raw_draw_i < 0:
                    # burn-in sweep → phase = "tune"
                    phase = "tune"
                    # raw_draw_i runs from −(tune+1) … −2, −1
                    # We want to map (−(tune+1)) → 0, … , (−2) → tune-2, (−1) → tune-1
                    burn_index = raw_draw_i + (total_tune + 1)
                    draw_i_to_write = burn_index
                else:
                    # sampling sweep → phase = "draw"
                    phase = "draw"
                    # raw_draw_i runs from 0 … draws−1
                    # We want to map 0 → total_tune, 1 → total_tune+1, …, draws-1 → total_tune + (draws-1)
                    draw_i_to_write = total_tune + raw_draw_i

                seq_str = SequenceState(seq_arr).to_string()
                row = [
                    int(chain_id),
                    int(draw_i_to_write),
                    phase,
                    seq_str,
                ] + [per_tf_map[tf] for tf in sorted(per_tf_map.keys())]
                writer.writerow(row)

        logger.info("Saved all sequences with per-TF scores → %s", seq_csv.relative_to(out_dir.parent))

    # 10) BUILD elites.json, tagging each with (chain, draw_index)
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
