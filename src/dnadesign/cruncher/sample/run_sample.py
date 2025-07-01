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
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
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
    and elites.json. Each chain gets its own independent seed (random/consensus/consensus_mix).
    """
    sample_cfg = cfg.sample
    rng = np.random.default_rng(42)

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

    # 5) RUN the MCMC sampling
    logger.info("Starting MCMC sampling …")
    optimizer.optimise()
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

    # 9)  BUILD elites list — keep draws whose ∑ normalised ≥ pwm_sum_threshold
    #     + enforce a per-sequence Hamming-distance diversity filter
    thr_norm: float = cfg.sample.pwm_sum_threshold  # e.g. 1.50
    min_dist: int = cfg.sample.min_dist  # e.g. 1
    raw_elites: list[tuple[np.ndarray, int, int, float, dict[str, float]]] = []
    norm_sums: list[float] = []  # diagnostics only

    for (chain_id, draw_idx), seq_arr, per_tf_map in zip(
        optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
    ):

        # ---------- per-PWM normalisation --------------------------
        #   raw_llr  →   frac = (raw_llr – μ_null) / (llr_consensus – μ_null)
        #   • μ_null  = null_mean  (expected background)
        #   • frac < 0  ⇒ treat as 0  (below background contributes nothing)
        #   • frac may exceed 1 if a window scores better than “column max”
        # -------------------------------------------------------------------
        fracs: list[float] = []
        for info in scorer._cache.values():
            raw_llr, *_ = scorer._best_llr_and_location(seq_arr, info)
            num, denom = raw_llr - info.null_mean, info.consensus_llr - info.null_mean
            frac = 0.0 if denom <= 0 else max(0.0, num / denom)  # no upper clamp
            fracs.append(frac)

        total_norm = float(sum(fracs))  # 0 … n_TF + ε
        norm_sums.append(total_norm)

        if total_norm >= thr_norm:
            raw_elites.append((seq_arr, chain_id, draw_idx, total_norm, per_tf_map))

    # -------- percentile diagnostics ----------------------------------------
    if norm_sums:
        p50, p90 = np.percentile(norm_sums, [50, 90])
        n_tf = len(scorer._cache)
        avg_thr_pct = 100 * thr_norm / n_tf
        logger.info("Normalised-sum percentiles  |  median %.2f   90%% %.2f", p50, p90)
        logger.info(
            "Threshold %.2f ⇒ ~%.0f%%-of-consensus on average per TF " "(%d regulators)", thr_norm, avg_thr_pct, n_tf
        )
        logger.info(
            "Typical draw: med %.2f (≈ %.0f%%/TF); top-10%% %.2f (≈ %.0f%%/TF)",
            p50,
            100 * p50 / n_tf,
            p90,
            100 * p90 / n_tf,
        )

    # -------- rank raw_elites by score --------------------------------------
    raw_elites.sort(key=lambda t: t[3], reverse=True)  # highest first

    # -------- apply Hamming diversity filter -------------------------------
    kept_elites: list[tuple[np.ndarray, int, int, float, dict[str, float]]] = []
    kept_seqs: list[np.ndarray] = []

    def _hamming(a: np.ndarray, b: np.ndarray) -> int:
        return int((a != b).sum())

    for tpl in raw_elites:
        seq_arr = tpl[0]
        if all(_hamming(seq_arr, s) >= min_dist for s in kept_seqs):
            kept_elites.append(tpl)
            kept_seqs.append(seq_arr)

    if min_dist > 0:
        logger.info(
            "Diversity filter (≥%d mismatches): kept %d / %d candidates", min_dist, len(kept_elites), len(raw_elites)
        )
    else:
        kept_elites = raw_elites  # no filtering

    # serialise elites
    elites: list[dict[str, object]] = []
    want_cons = bool(getattr(cfg.sample, "include_consensus_in_elites", False))

    for rank, (seq_arr, chain_id, draw_idx, total_norm, per_tf_map) in enumerate(kept_elites, 1):
        seq_str = SequenceState(seq_arr).to_string()
        per_tf_details: dict[str, dict[str, object]] = {}

        for tf_name, info in scorer._cache.items():
            # best site in this *sequence*
            raw_llr, offset, strand = scorer._best_llr_and_location(seq_arr, info)
            width = info.width
            start_pos = offset + 1
            strand_label = f"{strand}1"
            motif_diag = f"{start_pos}_[{strand_label}]_{width}"

            # OPTIONAL – consensus of the PWM (window only, no padding)
            if want_cons:
                idx_vec = np.argmax(info.lom, axis=1)  # (w,)
                consensus = "".join("ACGT"[i] for i in idx_vec)  # string
            else:
                consensus = None

            per_tf_details[tf_name] = {
                "raw_llr": float(raw_llr),
                "offset": offset,
                "strand": strand,
                "motif_diagram": motif_diag,
                "scaled_score": float(per_tf_map[tf_name]),
            }
            if want_cons:
                per_tf_details[tf_name]["consensus"] = consensus

        elites.append(
            {
                "id": str(uuid.uuid4()),
                "sequence": seq_str,
                "rank": rank,
                "norm_sum": total_norm,
                "chain": chain_id,
                "draw_idx": draw_idx,
                "per_tf": per_tf_details,
                "meta_type": "mcmc-elite",
                "meta_source": out_dir.name,
                "meta_date": datetime.now().isoformat(),
            }
        )

    logger.info(
        "Final elite count: %d (normalised-sum ≥ %.2f, min_dist ≥ %d)",
        len(elites),
        thr_norm,
        min_dist,
    )

    tf_label = "-".join(cfg.regulator_sets[0])
    date_stamp = datetime.now().strftime("%Y%m%d")
    prefix = "cruncher_elites"
    base_name = f"{prefix}_{tf_label}_{date_stamp}"

    save_dir = out_dir / base_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1)  .pt payload --------------------------------------------------------
    pt_path = save_dir / f"{base_name}.pt"
    torch.save(elites, str(pt_path))
    logger.info("Saved elites  → %s", pt_path.relative_to(out_dir.parent))

    # 2)  .yaml run-metadata -----------------------------------------------
    meta = {
        "generated": datetime.now().isoformat(),
        "n_elites": len(elites),
        "threshold_norm_sum": thr_norm,
        "min_hamming_dist": min_dist,
        "tf_label": tf_label,
        "sequence_length": cfg.sample.init.length,
        #  you can inline the full cfg if you prefer – this keeps it concise
        "config_file": str((out_dir / "config_used.yaml").resolve()),
    }
    yaml_path = save_dir / f"{base_name}.yaml"
    with yaml_path.open("w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    logger.info("Saved metadata → %s", yaml_path.relative_to(out_dir.parent))
