"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/run_set.py

Run a sample set and persist all sampling artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from dnadesign.cruncher.app.progress import progress_adapter
from dnadesign.cruncher.app.run_service import (
    update_run_index_from_manifest,
    update_run_index_from_status,
)
from dnadesign.cruncher.app.sample.artifacts import _elite_parquet_schema, _save_config, _write_parquet_rows
from dnadesign.cruncher.app.sample.diagnostics import (
    _elite_filter_passes,
    _EliteCandidate,
    _norm_map_for_elites,
    resolve_dsdna_mode,
)
from dnadesign.cruncher.app.sample.optimizer_config import (
    _effective_chain_count,
    _resolve_beta_ladder,
    _resolve_optimizer_kind,
)
from dnadesign.cruncher.app.sample.resources import _load_pwms_for_set
from dnadesign.cruncher.app.telemetry import RunTelemetry
from dnadesign.cruncher.artifacts.entries import artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    build_run_dir,
    config_used_path,
    elites_json_path,
    elites_mmr_meta_path,
    elites_path,
    elites_yaml_path,
    ensure_run_dirs,
    live_metrics_path,
    manifest_path,
    run_group_label,
    sequences_path,
    status_path,
    trace_path,
)
from dnadesign.cruncher.artifacts.manifest import build_run_manifest, write_manifest
from dnadesign.cruncher.artifacts.status import RunStatusWriter
from dnadesign.cruncher.config.moves import resolve_move_config
from dnadesign.cruncher.config.schema_v2 import BetaLadderFixed, CruncherConfig, SampleConfig
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.labels import format_regulator_slug
from dnadesign.cruncher.core.optimizers.cooling import make_beta_scheduler
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.selection.mmr import MmrCandidate, select_mmr_elites, tfbs_cores_from_scorer
from dnadesign.cruncher.core.sequence import canon_int
from dnadesign.cruncher.core.state import SequenceState
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path

logger = logging.getLogger(__name__)


def _build_elite_entries(
    candidates: list[_EliteCandidate],
    *,
    scorer: Scorer,
    sample_cfg: SampleConfig,
    want_consensus: bool,
    want_canonical: bool,
    meta_source: str,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for rank, cand in enumerate(candidates, 1):
        seq_arr = cand.seq_arr
        seq_str = SequenceState(seq_arr).to_string()
        canonical_seq = None
        if want_canonical:
            canonical_seq = SequenceState(canon_int(seq_arr)).to_string()
        per_tf_details: dict[str, dict[str, object]] = {}
        norm_map = cand.norm_map
        per_tf_map = cand.per_tf_map

        for tf_name in scorer.tf_names:
            raw_llr, offset, strand = scorer.best_llr(seq_arr, tf_name)
            width = scorer.pwm_width(tf_name)
            if strand == "-":
                offset = len(seq_arr) - width - offset
            start_pos = offset + 1
            strand_label = f"{strand}1"
            motif_diag = f"{start_pos}_[{strand_label}]_{width}"
            if want_consensus:
                consensus = scorer.consensus_sequence(tf_name)
            else:
                consensus = None
            per_tf_details[tf_name] = {
                "raw_llr": float(raw_llr),
                "offset": offset,
                "strand": strand,
                "width": width,
                "motif_diagram": motif_diag,
                "scaled_score": float(per_tf_map[tf_name]),
                "normalized_llr": float(norm_map.get(tf_name, 0.0)),
            }
            if want_consensus:
                per_tf_details[tf_name]["consensus"] = consensus

        entry = {
            "id": str(uuid.uuid4()),
            "sequence": seq_str,
            "rank": rank,
            "norm_sum": cand.sum_norm,
            "min_norm": cand.min_norm,
            "sum_norm": cand.sum_norm,
            "combined_score_final": cand.combined_score,
            "chain": cand.chain_id,
            "chain_1based": cand.chain_id + 1,
            "draw_idx": cand.draw_idx,
            "draw_in_phase": cand.draw_idx - sample_cfg.budget.tune
            if cand.draw_idx >= sample_cfg.budget.tune
            else cand.draw_idx,
            "per_tf": per_tf_details,
            "meta_type": "mcmc-elite",
            "meta_source": meta_source,
            "meta_date": datetime.now(timezone.utc).isoformat(),
        }
        if want_canonical:
            entry["canonical_sequence"] = canonical_seq
        entries.append(entry)
    return entries


def _resolve_final_softmin_beta(optimizer: object, sample_cfg: SampleConfig) -> float | None:
    if hasattr(optimizer, "final_softmin_beta") and callable(getattr(optimizer, "final_softmin_beta")):
        try:
            return getattr(optimizer, "final_softmin_beta")()
        except Exception:
            pass
    if hasattr(optimizer, "stats") and callable(getattr(optimizer, "stats")):
        try:
            stats = getattr(optimizer, "stats")()
            if isinstance(stats, dict):
                value = stats.get("final_softmin_beta")
                if isinstance(value, (int, float)):
                    return float(value)
        except Exception:
            pass
    softmin_cfg = sample_cfg.objective.softmin
    if softmin_cfg.enabled:
        total = sample_cfg.budget.tune + sample_cfg.budget.draws
        softmin_sched = {k: v for k, v in softmin_cfg.model_dump().items() if k in ("kind", "beta", "stages")}
        return make_beta_scheduler(softmin_sched, total)(total - 1)
    return None


def _assert_init_length_fits_pwms(sample_cfg: SampleConfig, pwms: dict[str, PWM]) -> None:
    if not pwms:
        raise ValueError("No PWMs available for sampling.")
    max_w = max(pwm.length for pwm in pwms.values())
    if sample_cfg.init.length < max_w:
        names = ", ".join(f"{tf}:{pwms[tf].length}" for tf in sorted(pwms))
        raise ValueError(
            f"init.length={sample_cfg.init.length} is shorter than the widest PWM (max={max_w}). "
            f"Per-TF lengths: {names}. Increase sample.init.length or trim PWMs via "
            "motif_store.pwm_window_lengths (with pwm_window_strategy=max_info)."
        )


def _run_sample_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
    set_count: int,
    include_set_index: bool,
    tfs: list[str],
    lockmap: dict[str, object],
    sample_cfg: SampleConfig,
    stage: str = "sample",
    run_kind: str | None = None,
    auto_opt_meta: dict[str, object] | None = None,
    init_seeds: list[np.ndarray] | None = None,
) -> Path:
    """
    Run MCMC sampler, save config/meta plus artifacts (trace.nc, sequences.parquet, elites.*).
    Each chain gets its own independent seed (random/consensus/consensus_mix) unless
    init_seeds are provided for warm-started runs.
    """
    base_out = config_path.parent / Path(cfg.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = build_run_dir(
        config_path=config_path,
        out_dir=cfg.out_dir,
        stage=stage,
        tfs=tfs,
        set_index=set_index,
        include_set_index=include_set_index,
    )
    ensure_run_dirs(out_dir, meta=True, artifacts=True, live=sample_cfg.output.live_metrics)
    run_group = run_group_label(tfs, set_index, include_set_index=include_set_index)
    run_logger = logger.debug if stage == "auto_opt" else logger.info
    stage_label = stage.upper().replace("_", "-")
    run_logger("=== RUN %s: %s ===", stage_label, out_dir)
    logger.debug("Full sample config: %s", sample_cfg.model_dump_json())

    metrics_path = live_metrics_path(out_dir) if sample_cfg.output.live_metrics else None
    optimizer_kind = _resolve_optimizer_kind(sample_cfg)
    chain_count = _effective_chain_count(sample_cfg, kind=optimizer_kind)
    status_writer = RunStatusWriter(
        path=status_path(out_dir),
        stage=stage,
        run_dir=out_dir,
        metrics_path=metrics_path,
        payload={
            "config_path": str(config_path.resolve()),
            "status_message": "initializing",
            "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
            "run_group": run_group,
            "run_kind": run_kind,
            "auto_opt": auto_opt_meta,
        },
    )
    telemetry = RunTelemetry(status_writer)
    progress = progress_adapter(sample_cfg.ui.progress_bar)
    update_run_index_from_status(
        config_path,
        out_dir,
        status_writer.payload,
        catalog_root=cfg.motif_store.catalog_root,
    )
    status_writer.update(
        status_message="loading_pwms",
        draws=sample_cfg.budget.draws,
        tune=sample_cfg.budget.tune,
        chains=chain_count,
        optimizer=optimizer_kind,
    )

    # 1) LOAD all required PWMs
    pwms = _load_pwms_for_set(cfg=cfg, config_path=config_path, tfs=tfs, lockmap=lockmap)

    if sample_cfg.init.kind == "consensus":
        regulator = sample_cfg.init.regulator
        if regulator not in tfs:
            raise ValueError(
                f"init.regulator='{regulator}' is not in regulator_sets[{set_index}] ({tfs}). "
                "Use a regulator within the active set or switch to init.kind='consensus_mix'."
            )

    max_w = max(pwm.length for pwm in pwms.values())
    if sample_cfg.init.length < max_w:
        names = ", ".join(f"{tf}:{pwms[tf].length}" for tf in sorted(pwms))
        raise ValueError(
            f"init.length={sample_cfg.init.length} is shorter than the widest PWM (max={max_w}). "
            f"Per-TF lengths: {names}. Increase sample.init.length or trim PWMs via "
            "motif_store.pwm_window_lengths (with pwm_window_strategy=max_info)."
        )

    # 2) INSTANTIATE SCORER and SequenceEvaluator
    scale = sample_cfg.objective.score_scale
    combine_cfg = sample_cfg.objective.combine
    logger.debug("Using score_scale = %r", scale)
    if scale == "llr" and len(tfs) > 1:
        logger.warning(
            "score_scale='llr' is not comparable across PWMs in multi-TF runs. "
            "Consider normalized-llr or logp, or set objective.allow_unscaled_llr=true to silence this warning."
        )

    def _sum_combine(values):
        return float(sum(values))

    combiner = None
    combine_resolved = "min"
    if combine_cfg == "sum":
        combiner = _sum_combine
        combine_resolved = "sum"
        if sample_cfg.objective.softmin.enabled:
            logger.warning("objective.combine='sum' disables softmin; softmin schedule will be ignored.")
    elif combine_cfg == "min":
        if scale == "consensus-neglop-sum":
            combiner = min
        combine_resolved = "min"
    elif scale == "consensus-neglop-sum":
        combiner = _sum_combine
        combine_resolved = "sum"
        if sample_cfg.objective.softmin.enabled:
            logger.warning(
                "score_scale='consensus-neglop-sum' defaults to sum() and disables softmin. "
                "Set objective.combine='min' to enforce weakest-TF optimization."
            )
    logger.debug("Building Scorer and SequenceEvaluator with scale=%r", scale)
    scorer = Scorer(
        pwms,
        bidirectional=sample_cfg.objective.bidirectional,
        scale=scale,
        background=(0.25, 0.25, 0.25, 0.25),
        pseudocounts=sample_cfg.objective.scoring.pwm_pseudocounts,
        log_odds_clip=sample_cfg.objective.scoring.log_odds_clip,
    )
    length_penalty_lambda = sample_cfg.objective.length_penalty_lambda
    length_penalty_ref = None
    if length_penalty_lambda > 0:
        length_penalty_ref = sample_cfg.init.length
    evaluator = SequenceEvaluator(
        pwms=pwms,
        scale=scale,
        combiner=combiner,  # defaults to min(...) for llr/logp/normalized-llr/z
        scorer=scorer,
        bidirectional=sample_cfg.objective.bidirectional,
        background=(0.25, 0.25, 0.25, 0.25),
        pseudocounts=sample_cfg.objective.scoring.pwm_pseudocounts,
        log_odds_clip=sample_cfg.objective.scoring.log_odds_clip,
        length_penalty_lambda=length_penalty_lambda,
        length_penalty_ref=length_penalty_ref,
    )

    logger.debug("Scorer and SequenceEvaluator instantiated")
    logger.debug("  Scorer.scale = %r", scorer.scale)
    logger.debug("  SequenceEvaluator.scale = %r", evaluator._scale)
    logger.debug("  SequenceEvaluator.combiner = %r", evaluator._combiner)

    dsdna_mode_for_opt = resolve_dsdna_mode(
        elites_cfg=sample_cfg.elites,
        bidirectional=sample_cfg.objective.bidirectional,
    )

    # 3) FLATTEN optimizer config for PT
    moves = resolve_move_config(sample_cfg.moves)
    opt_cfg: dict[str, object] = {
        "draws": sample_cfg.budget.draws,
        "tune": sample_cfg.budget.tune,
        "chains": chain_count,
        "min_dist": 0,
        "top_k": sample_cfg.elites.k,
        "bidirectional": sample_cfg.objective.bidirectional,
        "dsdna_hamming": False,
        "dsdna_canonicalize": dsdna_mode_for_opt,
        "score_scale": sample_cfg.objective.score_scale,
        "record_tune": sample_cfg.output.trace.include_tune,
        "progress_bar": sample_cfg.ui.progress_bar,
        "progress_every": sample_cfg.ui.progress_every,
        "early_stop": sample_cfg.early_stop.model_dump(),
        **moves.model_dump(),
        "softmin": sample_cfg.objective.softmin.model_dump(),
    }
    if init_seeds:
        opt_cfg["init_seeds"] = init_seeds

    if optimizer_kind == "pt":
        ladder_cfg = sample_cfg.optimizers.pt.beta_ladder
        betas, ladder_notes = _resolve_beta_ladder(sample_cfg.optimizers.pt)
        if ladder_notes:
            logger.info("PT ladder notes: %s", "; ".join(ladder_notes))
        if isinstance(ladder_cfg, BetaLadderFixed):
            opt_cfg["kind"] = "fixed"
            opt_cfg["beta"] = float(betas[0])
        else:
            opt_cfg["kind"] = "geometric"
            opt_cfg["beta"] = betas
        opt_cfg["swap_prob"] = sample_cfg.optimizers.pt.swap_prob
        opt_cfg["adaptive_swap"] = sample_cfg.optimizers.pt.ladder_adapt.model_dump()
        logger.debug("β-ladder (%d levels): %s", len(betas), ", ".join(f"{b:g}" for b in betas))
    else:
        raise ValueError("Unsupported optimizer; only 'pt' is allowed.")

    logger.debug("Optimizer config flattened: %s", opt_cfg)

    # 4) INSTANTIATE OPTIMIZER (PT), passing in init_cfg and pwms
    from dnadesign.cruncher.core.optimizers.registry import get_optimizer

    optimizer_factory = get_optimizer(optimizer_kind)
    run_logger("Instantiating optimizer: %s", optimizer_kind)
    rng = np.random.default_rng(sample_cfg.rng.seed + set_index - 1)
    optimizer = optimizer_factory(
        evaluator=evaluator,
        cfg=opt_cfg,
        rng=rng,
        pwms=pwms,
        init_cfg=sample_cfg.init,
        telemetry=telemetry,
        progress=progress,
    )

    # 5) RUN the MCMC sampling
    status_writer.update(status_message="sampling")
    run_logger("Starting MCMC sampling …")
    try:
        optimizer.optimise()
    except KeyboardInterrupt as exc:
        status_writer.finish(status="aborted", status_message="aborted", error=str(exc) or "KeyboardInterrupt")
        update_run_index_from_status(
            config_path,
            out_dir,
            status_writer.payload,
            catalog_root=cfg.motif_store.catalog_root,
        )
        raise
    except Exception as exc:  # pragma: no cover - ensures run status is updated on failure
        status_writer.finish(status="failed", status_message="failed", error=str(exc))
        update_run_index_from_status(
            config_path,
            out_dir,
            status_writer.payload,
            catalog_root=cfg.motif_store.catalog_root,
        )
        raise
    if status_writer is not None and hasattr(optimizer, "best_score"):
        best_meta = getattr(optimizer, "best_meta", None)
        status_writer.update(
            best_score=getattr(optimizer, "best_score", None),
            best_chain=(best_meta[0] + 1) if best_meta else None,
            best_draw=(best_meta[1]) if best_meta else None,
        )
    run_logger("MCMC sampling complete.")
    status_writer.update(status_message="sampling complete")
    status_writer.update(status_message="saving_artifacts")

    # 6) SAVE enriched config
    _save_config(cfg, out_dir, config_path, tfs=tfs, set_index=set_index, sample_cfg=sample_cfg, log_fn=run_logger)

    artifacts: list[dict[str, object]] = [
        artifact_entry(
            config_used_path(out_dir),
            out_dir,
            kind="config",
            label="Resolved config (config_used.yaml)",
            stage=stage,
        )
    ]

    # 7) SAVE trace.nc
    if sample_cfg.output.trace.save and hasattr(optimizer, "trace_idata") and optimizer.trace_idata is not None:
        from dnadesign.cruncher.artifacts.traces import save_trace

        save_trace(optimizer.trace_idata, trace_path(out_dir))
        artifacts.append(
            artifact_entry(
                trace_path(out_dir),
                out_dir,
                kind="trace",
                label="Trace (NetCDF)",
                stage=stage,
            )
        )
        logger.debug("Saved MCMC trace -> %s", trace_path(out_dir).relative_to(out_dir.parent))
    elif not sample_cfg.output.trace.save:
        logger.debug("Skipping trace.nc (sample.output.trace.save=false)")

    # Resolve final soft-min beta for elite ranking (from optimizer schedule)
    beta_softmin_final: float | None = _resolve_final_softmin_beta(optimizer, sample_cfg)

    # 8) SAVE sequences.parquet (chain, draw, phase, sequence_string, per-TF scaled scores)
    if (
        sample_cfg.output.save_sequences
        and hasattr(optimizer, "all_samples")
        and hasattr(optimizer, "all_meta")
        and hasattr(optimizer, "all_scores")
    ):
        seq_parquet = sequences_path(out_dir)
        tf_order = sorted(tfs)
        dsdna_mode = resolve_dsdna_mode(
            elites_cfg=sample_cfg.elites,
            bidirectional=sample_cfg.objective.bidirectional,
        )
        want_canonical_all = bool(dsdna_mode)

        def _sequence_rows() -> Iterable[dict[str, object]]:
            for (chain_id, draw_i), seq_arr, per_tf_map in zip(
                optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
            ):
                if draw_i < sample_cfg.budget.tune:
                    phase = "tune"
                    draw_i_to_write = draw_i
                    draw_in_phase = draw_i
                else:
                    phase = "draw"
                    draw_i_to_write = draw_i
                    draw_in_phase = draw_i - sample_cfg.budget.tune
                seq_str = SequenceState(seq_arr).to_string()
                norm_map = _norm_map_for_elites(
                    seq_arr,
                    per_tf_map,
                    scorer=scorer,
                    score_scale=sample_cfg.objective.score_scale,
                )
                row: dict[str, object] = {
                    "chain": int(chain_id),
                    "chain_1based": int(chain_id) + 1,
                    "draw": int(draw_i_to_write),
                    "draw_in_phase": int(draw_in_phase),
                    "phase": phase,
                    "sequence": seq_str,
                }
                if want_canonical_all:
                    row["canonical_sequence"] = SequenceState(canon_int(seq_arr)).to_string()
                row["combined_score_final"] = float(
                    evaluator.combined_from_scores(per_tf_map, beta=beta_softmin_final, length=seq_arr.size)
                )
                row["min_norm"] = float(min(norm_map.values())) if norm_map else 0.0
                for tf in tf_order:
                    row[f"score_{tf}"] = float(per_tf_map[tf])
                yield row

        _write_parquet_rows(seq_parquet, _sequence_rows())
        artifacts.append(
            artifact_entry(
                seq_parquet,
                out_dir,
                kind="table",
                label="Sequences with per-TF scores (Parquet)",
                stage=stage,
            )
        )
        logger.debug(
            "Saved all sequences with per-TF scores -> %s",
            seq_parquet.relative_to(out_dir.parent),
        )

    # 9) BUILD elites list — filter (representativeness), rank (objective), diversify
    filters = sample_cfg.elites.filters
    min_per_tf_norm = filters.min_per_tf_norm
    require_all = filters.require_all_tfs_over_min_norm
    pwm_sum_min = filters.pwm_sum_min
    raw_elites: list[_EliteCandidate] = []
    norm_sums: list[float] = []
    min_norms: list[float] = []
    total_draws_seen = len(optimizer.all_samples)

    for (chain_id, draw_idx), seq_arr, per_tf_map in zip(
        optimizer.all_meta, optimizer.all_samples, optimizer.all_scores
    ):
        norm_map = _norm_map_for_elites(
            seq_arr,
            per_tf_map,
            scorer=scorer,
            score_scale=sample_cfg.objective.score_scale,
        )
        min_norm = min(norm_map.values()) if norm_map else 0.0
        sum_norm = float(sum(norm_map.values()))
        norm_sums.append(sum_norm)
        min_norms.append(min_norm)

        if not _elite_filter_passes(
            norm_map=norm_map,
            min_norm=min_norm,
            sum_norm=sum_norm,
            min_per_tf_norm=min_per_tf_norm,
            require_all_tfs_over_min_norm=require_all,
            pwm_sum_min=pwm_sum_min,
        ):
            continue

        combined_score = evaluator.combined_from_scores(
            per_tf_map,
            beta=beta_softmin_final,
            length=seq_arr.size,
        )
        raw_elites.append(
            _EliteCandidate(
                seq_arr=seq_arr,
                chain_id=chain_id,
                draw_idx=draw_idx,
                combined_score=float(combined_score),
                min_norm=float(min_norm),
                sum_norm=float(sum_norm),
                per_tf_map=per_tf_map,
                norm_map=norm_map,
            )
        )

    # -------- percentile diagnostics ----------------------------------------
    if norm_sums:
        p50, p90 = np.percentile(norm_sums, [50, 90])
        n_tf = scorer.pwm_count
        avg_thr_pct = 100 * pwm_sum_min / n_tf if n_tf else 0.0
        logger.debug("Normalised-sum percentiles  |  median %.2f   90%% %.2f", p50, p90)
        if pwm_sum_min > 0:
            logger.debug(
                "Threshold %.2f ⇒ ~%.0f%%-of-consensus on average per TF (%d regulators)",
                pwm_sum_min,
                avg_thr_pct,
                n_tf,
            )
        logger.debug(
            "Typical draw: med %.2f (≈ %.0f%%/TF); top-10%% %.2f (≈ %.0f%%/TF)",
            p50,
            100 * p50 / n_tf if n_tf else 0.0,
            p90,
            100 * p90 / n_tf if n_tf else 0.0,
        )
    if min_norms and min_per_tf_norm is not None:
        p50_min, p90_min = np.percentile(min_norms, [50, 90])
        logger.debug("Normalised-min percentiles |  median %.2f   90%% %.2f", p50_min, p90_min)

    selection_cfg = sample_cfg.elites.selection
    elite_k = int(sample_cfg.elites.k or 0)
    dsdna_mode = resolve_dsdna_mode(
        elites_cfg=sample_cfg.elites,
        bidirectional=sample_cfg.objective.bidirectional,
    )
    mmr_meta_rows: list[dict[str, object]] | None = None
    mmr_summary: dict[str, object] | None = None
    mmr_meta_path: Path | None = None

    # -------- select elites --------------------------------------------------
    kept_elites: list[_EliteCandidate] = []
    kept_after_mmr = 0

    if elite_k > 0 and raw_elites:
        if selection_cfg.pool_size < elite_k:
            logger.warning(
                "MMR pool_size=%d < elites.k=%d; using pool_size=%d.",
                selection_cfg.pool_size,
                elite_k,
                elite_k,
            )
        mmr_candidates = [
            MmrCandidate(
                seq_arr=cand.seq_arr,
                chain_id=cand.chain_id,
                draw_idx=cand.draw_idx,
                combined_score=cand.combined_score,
                min_norm=cand.min_norm,
                sum_norm=cand.sum_norm,
                per_tf_map=cand.per_tf_map,
                norm_map=cand.norm_map,
            )
            for cand in raw_elites
        ]
        raw_by_id = {f"{cand.chain_id}:{cand.draw_idx}": cand for cand in raw_elites}
        pool_size = max(selection_cfg.pool_size, elite_k)
        if selection_cfg.relevance == "min_per_tf_norm":
            mmr_pool = sorted(
                mmr_candidates,
                key=lambda cand: (cand.min_norm, f"{cand.chain_id}:{cand.draw_idx}"),
                reverse=True,
            )[:pool_size]
        else:
            mmr_pool = sorted(
                mmr_candidates,
                key=lambda cand: (cand.combined_score, f"{cand.chain_id}:{cand.draw_idx}"),
                reverse=True,
            )[:pool_size]
        core_maps = {
            f"{cand.chain_id}:{cand.draw_idx}": tfbs_cores_from_scorer(
                cand.seq_arr,
                scorer=scorer,
                tf_names=scorer.tf_names,
            )
            for cand in mmr_pool
        }
        result = select_mmr_elites(
            mmr_pool,
            k=elite_k,
            pool_size=selection_cfg.pool_size,
            alpha=selection_cfg.alpha,
            relevance=selection_cfg.relevance,
            min_distance=selection_cfg.min_distance,
            dsdna=dsdna_mode,
            tf_names=scorer.tf_names,
            pwms=pwms,
            core_maps=core_maps,
        )
        kept_elites = [raw_by_id[f"{cand.chain_id}:{cand.draw_idx}"] for cand in result.selected]
        kept_after_mmr = len(kept_elites)
        mmr_meta_rows = result.meta
        mmr_summary = {
            "pool_size": selection_cfg.pool_size,
            "k": elite_k,
            "alpha": selection_cfg.alpha,
            "median_relevance_raw": result.median_relevance_raw,
            "mean_pairwise_distance": result.mean_pairwise_distance,
            "min_pairwise_distance": result.min_pairwise_distance,
        }

    if not kept_elites and (pwm_sum_min > 0 or min_per_tf_norm is not None):
        logger.warning("Elite filters removed all candidates; relax min_per_tf_norm/pwm_sum_min or min_distance.")

    # serialise elites
    want_cons = bool(sample_cfg.output.include_consensus_in_elites)
    want_canonical = bool(dsdna_mode)
    elites = _build_elite_entries(
        kept_elites,
        scorer=scorer,
        sample_cfg=sample_cfg,
        want_consensus=want_cons,
        want_canonical=want_canonical,
        meta_source=out_dir.name,
    )

    run_logger(
        "Final elite count: %d (pwm_sum_min=%s, min_per_tf_norm=%s, min_distance=%s)",
        len(elites),
        f"{pwm_sum_min:.2f}" if pwm_sum_min else "off",
        f"{min_per_tf_norm:.2f}" if min_per_tf_norm is not None else "off",
        f"{selection_cfg.min_distance:.2f}" if isinstance(selection_cfg.min_distance, (int, float)) else "off",
    )

    tf_label = format_regulator_slug(tfs)

    # 1)  parquet payload ----------------------------------------------------
    parquet_path = elites_path(out_dir)

    def _elite_rows_from(entries: list[dict[str, object]]) -> Iterable[dict[str, object]]:
        for entry in entries:
            row = dict(entry)
            per_tf = row.pop("per_tf", None)
            if per_tf is not None:
                row["per_tf_json"] = json.dumps(per_tf, sort_keys=True)
                for tf_name, details in per_tf.items():
                    row[f"score_{tf_name}"] = details.get("scaled_score")
                    row[f"norm_{tf_name}"] = details.get("normalized_llr")
            yield row

    elite_schema = _elite_parquet_schema(tfs, include_canonical=want_canonical)
    _write_parquet_rows(parquet_path, _elite_rows_from(elites), chunk_size=2000, schema=elite_schema)
    logger.debug("Saved elites Parquet -> %s", parquet_path.relative_to(out_dir.parent))

    json_path = elites_json_path(out_dir)
    json_path.write_text(json.dumps(elites, indent=2))
    logger.debug("Saved elites JSON -> %s", json_path.relative_to(out_dir.parent))

    if mmr_meta_rows:
        mmr_meta_path = elites_mmr_meta_path(out_dir)
        _write_parquet_rows(mmr_meta_path, mmr_meta_rows, chunk_size=2000)
        logger.debug("Saved elites MMR meta -> %s", mmr_meta_path.relative_to(out_dir.parent))

    # 2)  .yaml run-metadata -----------------------------------------------
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_elites": len(elites),
        "threshold_norm_sum": pwm_sum_min,
        "min_per_tf_norm": min_per_tf_norm,
        "require_all_tfs_over_min_norm": require_all,
        "selection_policy": "mmr",
        "selection": selection_cfg.model_dump(),
        "min_distance": selection_cfg.min_distance,
        "dsdna_canonicalize": want_canonical,
        "total_draws_seen": total_draws_seen,
        "passed_pre_filter": len(raw_elites),
        "kept_after_mmr": kept_after_mmr,
        "objective_combine": combine_resolved,
        "softmin_beta_final_resolved": beta_softmin_final,
        "indexing_note": (
            "chain is 0-based; chain_1based is 1-based; draw_idx is absolute sweep; draw_in_phase is phase-relative"
        ),
        "tf_label": tf_label,
        "sequence_length": sample_cfg.init.length,
        #  you can inline the full cfg if you prefer – this keeps it concise
        "config_file": str(config_used_path(out_dir).resolve()),
        "regulator_set": {"index": set_index, "tfs": tfs},
    }
    if mmr_summary is not None:
        meta["mmr_summary"] = mmr_summary
    yaml_path = elites_yaml_path(out_dir)
    with yaml_path.open("w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    logger.debug("Saved metadata -> %s", yaml_path.relative_to(out_dir.parent))

    artifacts.extend(
        [
            artifact_entry(
                parquet_path,
                out_dir,
                kind="table",
                label="Elite sequences (Parquet)",
                stage=stage,
            ),
            artifact_entry(
                json_path,
                out_dir,
                kind="json",
                label="Elite sequences (JSON)",
                stage=stage,
            ),
            artifact_entry(
                yaml_path,
                out_dir,
                kind="metadata",
                label="Elite metadata (YAML)",
                stage=stage,
            ),
        ]
    )
    if mmr_meta_path is not None:
        artifacts.append(
            artifact_entry(
                mmr_meta_path,
                out_dir,
                kind="table",
                label="Elite MMR selection metadata",
                stage=stage,
            )
        )

    # 10) RUN MANIFEST (for reporting + provenance)
    catalog_root = resolve_catalog_root(config_path, cfg.motif_store.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    lock_path = resolve_lock_path(config_path)
    lock_snapshot_path = manifest_path(out_dir).parent / "lockfile.json"
    lock_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(lock_path, lock_snapshot_path)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        raise ValueError(f"Failed to snapshot lockfile for run: {lock_snapshot_path}") from exc
    manifest = build_run_manifest(
        stage=stage,
        cfg=cfg,
        config_path=config_path,
        lock_path=lock_snapshot_path,
        lockmap={tf: lockmap[tf] for tf in tfs},
        catalog=catalog,
        run_dir=out_dir,
        artifacts=artifacts,
        extra={
            "sequence_length": sample_cfg.init.length,
            "seed": sample_cfg.rng.seed,
            "seed_effective": sample_cfg.rng.seed + set_index - 1,
            "record_tune": sample_cfg.output.trace.include_tune,
            "save_trace": sample_cfg.output.trace.save,
            "tune": sample_cfg.budget.tune,
            "draws": sample_cfg.budget.draws,
            "restarts": sample_cfg.budget.restarts,
            "early_stop": sample_cfg.early_stop.model_dump(),
            "top_k": sample_cfg.elites.k,
            "elites": sample_cfg.elites.model_dump(),
            "regulator_set": {"index": set_index, "tfs": tfs, "count": set_count},
            "run_group": run_group,
            "run_kind": run_kind,
            "auto_opt": auto_opt_meta,
            "objective": {
                "score_scale": sample_cfg.objective.score_scale,
                "combine": combine_resolved,
                "bidirectional": sample_cfg.objective.bidirectional,
                "softmin": sample_cfg.objective.softmin.model_dump(),
                "softmin_beta_final_resolved": beta_softmin_final,
            },
            "optimizer": {
                "kind": optimizer_kind,
                "pt": sample_cfg.optimizers.pt.model_dump(),
            },
            "optimizer_stats": optimizer.stats() if hasattr(optimizer, "stats") else {},
            "objective_schedule_summary": optimizer.objective_schedule_summary()
            if hasattr(optimizer, "objective_schedule_summary")
            else {},
        },
    )
    manifest_path_out = write_manifest(out_dir, manifest)
    update_run_index_from_manifest(
        config_path,
        out_dir,
        manifest,
        catalog_root=cfg.motif_store.catalog_root,
    )
    logger.debug("Wrote run manifest -> %s", manifest_path_out.relative_to(out_dir.parent))
    status_writer.finish(status="completed", artifacts=artifacts)
    update_run_index_from_status(
        config_path,
        out_dir,
        status_writer.payload,
        catalog_root=cfg.motif_store.catalog_root,
    )
    return out_dir
