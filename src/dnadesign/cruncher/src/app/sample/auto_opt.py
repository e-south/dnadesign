"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/auto_opt.py

Auto-optimize sampling configurations and select pilot candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics
from dnadesign.cruncher.analysis.elites import find_elites_parquet
from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.app.sample.artifacts import (
    _auto_opt_best_marker_path,
    _format_run_path,
    _prune_auto_opt_runs,
    _selected_config_payload,
    _write_auto_opt_best_marker,
)
from dnadesign.cruncher.app.sample.auto_opt_candidates import (
    _budget_to_tune_draws,  # noqa: F401
    _build_final_sample_cfg,
    _build_pilot_sample_cfg,
    _pilot_budget_levels,
    _stable_pilot_seed,
)
from dnadesign.cruncher.app.sample.auto_opt_models import AutoOptCandidate, AutoOptSpec
from dnadesign.cruncher.app.sample.auto_opt_scorecard import (
    _assess_candidate_quality,
    _confidence_from_candidates,
    _mmr_scorecard_from_elites,
    _rank_auto_opt_candidates,
    _select_auto_opt_candidate,
    _validate_auto_opt_candidates,
)
from dnadesign.cruncher.app.sample.diagnostics import (
    _bootstrap_seed,
    _bootstrap_seed_payload,
    _bootstrap_top_k_ci,
    _draw_scores_from_sequences,
    _pooled_bootstrap_seed,
    _top_k_median_from_scores,
    resolve_dsdna_mode,
)
from dnadesign.cruncher.app.sample.optimizer_config import (
    _format_auto_opt_config_summary,
    _format_move_probs,
    _resolve_beta_ladder,
)
from dnadesign.cruncher.app.sample.resources import _load_pwms_for_set
from dnadesign.cruncher.app.sample.run_set import _run_sample_for_set
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    run_group_label,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest
from dnadesign.cruncher.config.schema_v2 import AutoOptConfig, CruncherConfig, SampleConfig
from dnadesign.cruncher.core.pwm import PWM

logger = logging.getLogger(__name__)


def _run_auto_optimize_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
    set_count: int,
    include_set_index: bool,
    tfs: list[str],
    lockmap: dict[str, object],
    sample_cfg: SampleConfig,
    auto_cfg: AutoOptConfig,
) -> Path:
    logger.info("Auto-optimize enabled: running PT pilot sweeps.")
    run_group = run_group_label(tfs, set_index, include_set_index=include_set_index)
    pwms = _load_pwms_for_set(cfg=cfg, config_path=config_path, tfs=tfs, lockmap=lockmap)
    length = sample_cfg.init.length
    budgets = _pilot_budget_levels(sample_cfg, auto_cfg)
    move_profiles = list(dict.fromkeys(auto_cfg.move_profiles or [sample_cfg.moves.profile]))
    cooling_boosts = list(dict.fromkeys(auto_cfg.cooling_boosts or [1.0]))
    pt_swap_probs = list(dict.fromkeys(auto_cfg.pt_swap_probs or [sample_cfg.optimizers.pt.swap_prob]))
    ladder, _ = _resolve_beta_ladder(sample_cfg.optimizers.pt)
    ladder_size = len(ladder)
    pt_scales = sorted(set(cooling_boosts))
    logger.info("Auto-opt fixed length: %s", length)
    logger.info("Auto-opt budget levels: %s", ", ".join(str(b) for b in budgets))
    logger.info("Auto-opt move profiles: %s", ", ".join(move_profiles))
    logger.info("Auto-opt pt beta scales: %s", ", ".join(f"{b:g}" for b in pt_scales))
    logger.info("Auto-opt PT swap_probs: %s", ", ".join(f"{p:g}" for p in pt_swap_probs))
    logger.info("Auto-opt PT ladder size: %s", ladder_size)
    logger.info("Auto-opt keep_pilots: %s", auto_cfg.keep_pilots)
    candidate_specs: list[AutoOptSpec] = []
    for move_profile in move_profiles:
        for swap_prob in pt_swap_probs:
            for scale in pt_scales:
                candidate_specs.append(
                    AutoOptSpec(
                        kind="pt",
                        length=length,
                        move_profile=move_profile,
                        cooling_boost=scale,
                        swap_prob=swap_prob,
                        ladder_size=ladder_size,
                    )
                )
    seed_rng = np.random.default_rng(sample_cfg.rng.seed)

    def _run_candidate(
        *,
        kind: str,
        length: int,
        budget: int,
        label: str,
        cooling_boost: float,
        move_profile: str,
        swap_prob: float | None = None,
        ladder_size: int | None = None,
        move_probs: dict[str, float] | None = None,
        move_probs_label: str | None = None,
        init_seeds: list[np.ndarray] | None = None,
    ) -> AutoOptCandidate:
        runs: list[AutoOptCandidate] = []
        for rep in range(auto_cfg.replicates):
            pilot_cfg, notes = _build_pilot_sample_cfg(
                sample_cfg,
                kind=kind,
                total_sweeps=budget,
                cooling_boost=cooling_boost,
                move_profile=move_profile,
                swap_prob=swap_prob,
                move_probs=move_probs,
            )
            seed_label = f"{label}_R{rep + 1}"
            if sample_cfg.rng.deterministic:
                seed = _stable_pilot_seed(pilot_cfg=pilot_cfg, tfs=tfs, lockmap=lockmap, label=seed_label, kind=kind)
            else:
                seed = int(seed_rng.integers(0, 2**31 - 1))
            pilot_cfg.rng.seed = seed
            notes.append(f"set pilot seed={seed}")
            if notes:
                logger.debug(
                    "Auto-opt %s %s (L=%d B=%d R=%d) overrides: %s",
                    label,
                    kind,
                    pilot_cfg.init.length,
                    budget,
                    rep + 1,
                    "; ".join(notes),
                )
            pilot_meta = {
                "mode": "pilot",
                "attempt": label,
                "candidate": kind,
                "length": pilot_cfg.init.length,
                "seed": pilot_cfg.rng.seed,
                "draws": pilot_cfg.budget.draws,
                "tune": pilot_cfg.budget.tune,
                "restarts": pilot_cfg.budget.restarts,
                "budget": budget,
                "replicate": rep + 1,
                "cooling_boost": cooling_boost,
                "beta_scale": cooling_boost,
                "move_profile": move_profile,
                "swap_prob": swap_prob,
                "ladder_size": ladder_size,
                "move_probs_label": move_probs_label,
            }
            pilot_run_dir = _run_sample_for_set(
                cfg,
                config_path,
                set_index=set_index,
                set_count=set_count,
                include_set_index=include_set_index,
                tfs=tfs,
                lockmap=lockmap,
                sample_cfg=pilot_cfg,
                stage="auto_opt",
                run_kind=f"auto_opt_{label}_{kind}_L{pilot_cfg.init.length}_B{budget}_R{rep + 1}",
                auto_opt_meta=pilot_meta,
                init_seeds=init_seeds,
            )
            try:
                candidate = _evaluate_pilot_run(
                    pilot_run_dir,
                    kind,
                    pilot_cfg=pilot_cfg,
                    auto_cfg=auto_cfg,
                    pwms=pwms,
                    tf_names=tfs,
                    budget=budget,
                    mode="auto_opt",
                    cooling_boost=cooling_boost,
                    move_profile=move_profile,
                    swap_prob=swap_prob,
                    ladder_size=ladder_size,
                    move_probs=move_probs,
                    move_probs_label=move_probs_label,
                )
                if candidate.length is None:
                    candidate.length = pilot_cfg.init.length
            except Exception as exc:
                logger.warning("Auto-opt %s %s failed: %s", label, kind, exc)
                candidate = AutoOptCandidate(
                    kind=kind,
                    length=pilot_cfg.init.length,
                    budget=budget,
                    cooling_boost=cooling_boost,
                    move_profile=move_profile,
                    swap_prob=swap_prob,
                    ladder_size=ladder_size,
                    move_probs=move_probs,
                    move_probs_label=move_probs_label,
                    run_dir=pilot_run_dir,
                    run_dirs=[pilot_run_dir],
                    best_score=None,
                    top_k_median_final=None,
                    best_score_final=None,
                    top_k_ci_low=None,
                    top_k_ci_high=None,
                    rhat=None,
                    ess=None,
                    unique_fraction=None,
                    balance_median=None,
                    diversity=None,
                    improvement=None,
                    acceptance_b=None,
                    acceptance_m=None,
                    acceptance_mh=None,
                    swap_rate=None,
                    status="fail",
                    quality="fail",
                    warnings=[str(exc)],
                    diagnostics={},
                    pilot_score=None,
                    median_relevance_raw=None,
                    mean_pairwise_distance=None,
                    unique_successes=None,
                )
            runs.append(candidate)
        return _aggregate_candidate_runs(
            runs,
            budget=budget,
            scorecard_k=auto_cfg.policy.scorecard.k,
        )

    def _log_candidate(candidate: AutoOptCandidate, *, label: str) -> None:
        diag_status = "n/a"
        if isinstance(candidate.diagnostics, dict):
            diag_status = candidate.diagnostics.get("status") or "n/a"
        move_probs_label = "-"
        if candidate.move_probs is not None:
            move_probs_label = _format_move_probs(candidate.move_probs)
        logger.debug(
            (
                "Auto-opt %s %s L=%s B=%s beta=%s move=%s swap=%s ladder=%s moveset=%s: "
                "scorecard=%s diagnostics=%s pilot_score=%s top_k_median=%s best=%s balance=%s diversity=%s "
                "rhat=%s ess=%s unique_fraction=%s"
            ),
            label,
            candidate.kind,
            candidate.length if candidate.length is not None else "n/a",
            candidate.budget if candidate.budget is not None else "n/a",
            f"{candidate.cooling_boost:g}",
            candidate.move_profile,
            f"{candidate.swap_prob:g}" if candidate.swap_prob is not None else "n/a",
            str(candidate.ladder_size) if candidate.ladder_size is not None else "n/a",
            move_probs_label,
            candidate.quality,
            diag_status,
            f"{candidate.pilot_score:.3f}" if candidate.pilot_score is not None else "n/a",
            f"{candidate.top_k_median_final:.3f}" if candidate.top_k_median_final is not None else "n/a",
            f"{candidate.best_score_final:.3f}" if candidate.best_score_final is not None else "n/a",
            f"{candidate.balance_median:.3f}" if candidate.balance_median is not None else "n/a",
            f"{candidate.diversity:.3f}" if candidate.diversity is not None else "n/a",
            f"{candidate.rhat:.3f}" if candidate.rhat is not None else "n/a",
            f"{candidate.ess:.1f}" if candidate.ess is not None else "n/a",
            f"{candidate.unique_fraction:.2f}" if candidate.unique_fraction is not None else "n/a",
        )
        if candidate.warnings:
            is_pilot = label.startswith("pilot_")
            if is_pilot:
                log_fn = logger.info
            else:
                log_fn = logger.warning if candidate.quality == "fail" else logger.info
            log_fn("Auto-opt %s %s warnings: %s", label, candidate.kind, "; ".join(candidate.warnings))

    all_candidates: list[AutoOptCandidate] = []
    final_level_candidates: list[AutoOptCandidate] = []

    def _spec_label(spec: AutoOptSpec, *, idx: int, budget: int) -> str:
        label = f"pilot_{idx}_L{spec.length}_B{budget}_M{spec.move_profile}_C{spec.cooling_boost:g}"
        if spec.kind == "pt":
            swap_label = f"{spec.swap_prob:g}" if spec.swap_prob is not None else "n/a"
            ladder_label = str(spec.ladder_size) if spec.ladder_size is not None else "n/a"
            label = f"{label}_S{swap_label}_T{ladder_label}"
        if spec.move_probs_label:
            label = f"{label}_{spec.move_probs_label}"
        return label

    current_specs = list(candidate_specs)
    for idx, budget in enumerate(budgets):
        level_candidates = []
        for spec in current_specs:
            label = _spec_label(spec, idx=idx + 1, budget=budget)
            candidate = _run_candidate(
                kind=spec.kind,
                length=spec.length,
                budget=budget,
                label=label,
                cooling_boost=spec.cooling_boost,
                move_profile=spec.move_profile,
                swap_prob=spec.swap_prob,
                ladder_size=spec.ladder_size,
                move_probs=spec.move_probs,
                move_probs_label=spec.move_probs_label,
            )
            candidate_notes = _assess_candidate_quality(candidate, auto_cfg, mode="auto_opt")
            if candidate_notes:
                candidate.warnings.extend(candidate_notes)
            _log_candidate(candidate, label=label)
            level_candidates.append(candidate)
            all_candidates.append(candidate)

        confidence_level, _, _ = _confidence_from_candidates(level_candidates)
        if confidence_level == "high":
            final_level_candidates = level_candidates
            if idx < len(budgets) - 1:
                logger.info(
                    "Auto-opt: confident at budget %s; skipping remaining budget levels.",
                    budget,
                )
            break
        if idx < len(budgets) - 1:
            ranked = _rank_auto_opt_candidates(level_candidates)
            current_specs = [
                AutoOptSpec(
                    kind=c.kind,
                    length=int(c.length) if c.length is not None else length,
                    move_profile=c.move_profile,
                    cooling_boost=c.cooling_boost,
                    swap_prob=c.swap_prob,
                    ladder_size=c.ladder_size,
                    move_probs=c.move_probs,
                    move_probs_label=c.move_probs_label,
                )
                for c in ranked
            ]
        else:
            final_level_candidates = level_candidates

    if not final_level_candidates:
        raise ValueError("Auto-optimize failed: no final-level candidates were evaluated.")

    decision_notes: list[str] = []
    viable, _ = _validate_auto_opt_candidates(
        final_level_candidates,
        allow_warn=auto_cfg.policy.allow_warn,
    )

    confidence_level, _best_candidate, _second_candidate = _confidence_from_candidates(viable)
    selection_confident = confidence_level == "high"
    decision_notes.append(f"confidence={confidence_level}")
    if not selection_confident and auto_cfg.policy.allow_warn:
        logger.warning(
            "Auto-optimize: could not confidently separate top candidates at max budget; "
            "selecting best available candidate."
        )
    elif not selection_confident:
        raise ValueError(
            "Auto-optimize failed: could not confidently separate top candidates at the maximum "
            "configured budgets/replicates. Increase auto_opt.budget_levels and/or auto_opt.replicates."
        )

    winner = _select_auto_opt_candidate(viable, allow_fail=False)

    if winner.quality == "ok":
        logger.info(
            (
                "Auto-optimize selected %s L=%s move=%s beta=%s swap=%s ladder=%s "
                "(pilot_score=%s top_k_median=%s, best=%s, balance=%s, rhat=%s, ess=%s, unique_fraction=%s)."
            ),
            winner.kind,
            winner.length if winner.length is not None else "n/a",
            winner.move_profile,
            f"{winner.cooling_boost:g}",
            f"{winner.swap_prob:g}" if winner.swap_prob is not None else "n/a",
            str(winner.ladder_size) if winner.ladder_size is not None else "n/a",
            f"{winner.pilot_score:.3f}" if winner.pilot_score is not None else "n/a",
            f"{winner.top_k_median_final:.3f}" if winner.top_k_median_final is not None else "n/a",
            f"{winner.best_score_final:.3f}" if winner.best_score_final is not None else "n/a",
            f"{winner.balance_median:.3f}" if winner.balance_median is not None else "n/a",
            f"{winner.rhat:.3f}" if winner.rhat is not None else "n/a",
            f"{winner.ess:.1f}" if winner.ess is not None else "n/a",
            f"{winner.unique_fraction:.2f}" if winner.unique_fraction is not None else "n/a",
        )
    else:
        logger.warning(
            (
                "Auto-optimize selected %s L=%s move=%s beta=%s swap=%s ladder=%s "
                "(quality=%s pilot_score=%s top_k_median=%s, best=%s, balance=%s, rhat=%s, "
                "ess=%s, unique_fraction=%s)."
            ),
            winner.kind,
            winner.length if winner.length is not None else "n/a",
            winner.move_profile,
            f"{winner.cooling_boost:g}",
            f"{winner.swap_prob:g}" if winner.swap_prob is not None else "n/a",
            str(winner.ladder_size) if winner.ladder_size is not None else "n/a",
            winner.quality,
            f"{winner.pilot_score:.3f}" if winner.pilot_score is not None else "n/a",
            f"{winner.top_k_median_final:.3f}" if winner.top_k_median_final is not None else "n/a",
            f"{winner.best_score_final:.3f}" if winner.best_score_final is not None else "n/a",
            f"{winner.balance_median:.3f}" if winner.balance_median is not None else "n/a",
            f"{winner.rhat:.3f}" if winner.rhat is not None else "n/a",
            f"{winner.ess:.1f}" if winner.ess is not None else "n/a",
            f"{winner.unique_fraction:.2f}" if winner.unique_fraction is not None else "n/a",
        )
    if winner.cooling_boost > 1:
        decision_notes.append(f"cooling_boost={winner.cooling_boost:g}")
    if winner.move_profile != sample_cfg.moves.profile:
        decision_notes.append(f"move_profile={winner.move_profile}")
    if winner.move_probs is not None:
        decision_notes.append(f"move_probs={_format_move_probs(winner.move_probs)}")
    if winner.kind == "pt" and winner.swap_prob is not None:
        if winner.swap_prob != sample_cfg.optimizers.pt.swap_prob:
            decision_notes.append(f"swap_prob={winner.swap_prob:g}")
        if winner.ladder_size is not None:
            ladder, _ = _resolve_beta_ladder(sample_cfg.optimizers.pt)
            if winner.ladder_size != len(ladder):
                decision_notes.append(f"ladder_size={winner.ladder_size}")
    if decision_notes:
        logger.warning("Auto-opt notes: %s", ", ".join(decision_notes))

    final_cfg, notes = _build_final_sample_cfg(
        sample_cfg,
        kind=winner.kind,
        cooling_boost=winner.cooling_boost,
        move_profile=winner.move_profile,
        swap_prob=winner.swap_prob,
        move_probs=winner.move_probs,
    )
    if notes:
        logger.info("Auto-opt final overrides: %s", "; ".join(notes))
    ranking_label = "pilot_score -> median_relevance_raw -> mean_pairwise_distance -> status"
    logger.info("Auto-opt selection ranking: %s.", ranking_label)
    logger.info("Auto-opt final config: %s", _format_auto_opt_config_summary(final_cfg))
    pilot_root = all_candidates[0].run_dir.parent if all_candidates else None
    best_marker_path = _auto_opt_best_marker_path(pilot_root, run_group=run_group) if pilot_root is not None else None
    if pilot_root is not None:
        logger.info(
            "Auto-opt details: pilot manifests under %s; best marker -> %s; final config in config_used.yaml.",
            _format_run_path(pilot_root, base=config_path.parent),
            _format_run_path(best_marker_path, base=config_path.parent) if best_marker_path else "n/a",
        )

    def _trace_metric(candidate: AutoOptCandidate, key: str) -> object | None:
        if not isinstance(candidate.diagnostics, dict):
            return None
        metrics = candidate.diagnostics.get("metrics")
        if not isinstance(metrics, dict):
            return None
        trace = metrics.get("trace")
        if not isinstance(trace, dict):
            return None
        return trace.get(key)

    decision_payload = {
        "mode": "final",
        "selected": winner.kind,
        "selected_length": winner.length,
        "selected_move_profile": winner.move_profile,
        "selected_cooling_boost": winner.cooling_boost,
        "selected_beta_scale": winner.cooling_boost,
        "selected_swap_prob": winner.swap_prob,
        "selected_ladder_size": winner.ladder_size,
        "selected_move_probs": winner.move_probs,
        "selected_move_probs_label": winner.move_probs_label,
        "selection_quality": winner.quality,
        "selection_confident": selection_confident,
        "selection_confidence": confidence_level,
        "notes": decision_notes,
        "config_summary": _format_auto_opt_config_summary(final_cfg),
        "selected_config": _selected_config_payload(final_cfg),
        "selection_metrics": {
            "pilot_score": winner.pilot_score,
            "median_relevance_raw": winner.median_relevance_raw,
            "mean_pairwise_distance": winner.mean_pairwise_distance,
            "unique_successes": winner.unique_successes,
            "top_k_median_final": winner.top_k_median_final,
            "best_score_final": winner.best_score_final,
            "top_k_ci_low": winner.top_k_ci_low,
            "top_k_ci_high": winner.top_k_ci_high,
        },
        "auto_opt_config": {
            "budget_levels": auto_cfg.budget_levels,
            "replicates": auto_cfg.replicates,
            "keep_pilots": auto_cfg.keep_pilots,
            "scorecard_k": auto_cfg.policy.scorecard.k,
            "scorecard_alpha": auto_cfg.policy.scorecard.alpha,
            "diversity_weight": auto_cfg.policy.diversity_weight,
            "move_profiles": auto_cfg.move_profiles,
            "pt_swap_probs": auto_cfg.pt_swap_probs,
        },
        "pilot_root": str(pilot_root) if pilot_root is not None else None,
        "best_marker": str(best_marker_path) if best_marker_path is not None else None,
        "candidates": [
            {
                "kind": candidate.kind,
                "length": candidate.length,
                "budget": candidate.budget,
                "cooling_boost": candidate.cooling_boost,
                "move_profile": candidate.move_profile,
                "swap_prob": candidate.swap_prob,
                "ladder_size": candidate.ladder_size,
                "move_probs_label": candidate.move_probs_label,
                "runs": [path.name for path in candidate.run_dirs],
                "status": candidate.status,
                "quality": candidate.quality,
                "best_score": candidate.best_score,
                "pilot_score": candidate.pilot_score,
                "median_relevance_raw": candidate.median_relevance_raw,
                "mean_pairwise_distance": candidate.mean_pairwise_distance,
                "unique_successes": candidate.unique_successes,
                "top_k_median_final": candidate.top_k_median_final,
                "best_score_final": candidate.best_score_final,
                "top_k_ci_low": candidate.top_k_ci_low,
                "top_k_ci_high": candidate.top_k_ci_high,
                "balance_median": candidate.balance_median,
                "diversity": candidate.diversity,
                "improvement": candidate.improvement,
                "acceptance_b": candidate.acceptance_b,
                "acceptance_m": candidate.acceptance_m,
                "acceptance_mh": candidate.acceptance_mh,
                "swap_rate": candidate.swap_rate,
                "rhat": candidate.rhat,
                "ess": candidate.ess,
                "ess_ratio": _trace_metric(candidate, "ess_ratio"),
                "trace_draws": _trace_metric(candidate, "draws"),
                "trace_chains": _trace_metric(candidate, "chains"),
                "trace_draws_expected": _trace_metric(candidate, "draws_expected"),
                "trace_chains_expected": _trace_metric(candidate, "chains_expected"),
                "unique_fraction": candidate.unique_fraction,
            }
            for candidate in all_candidates
        ],
    }
    final_run_dir = _run_sample_for_set(
        cfg,
        config_path,
        set_index=set_index,
        set_count=set_count,
        include_set_index=include_set_index,
        tfs=tfs,
        lockmap=lockmap,
        sample_cfg=final_cfg,
        stage="sample",
        run_kind="auto_opt_final",
        auto_opt_meta=decision_payload,
    )
    if pilot_root is not None:
        marker_payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "selected_candidate": {
                "kind": winner.kind,
                "length": winner.length,
                "cooling_boost": winner.cooling_boost,
                "move_profile": winner.move_profile,
                "swap_prob": winner.swap_prob,
                "ladder_size": winner.ladder_size,
                "move_probs_label": winner.move_probs_label,
                "pilot_run": winner.run_dir.name,
            },
            "final_sample_run": final_run_dir.name,
            "config_summary": _format_auto_opt_config_summary(final_cfg),
        }
        marker_path = _write_auto_opt_best_marker(pilot_root, marker_payload, run_group=run_group)
        logger.info(
            "Auto-opt best marker updated -> %s",
            _format_run_path(marker_path, base=config_path.parent),
        )
        removed = _prune_auto_opt_runs(
            config_path=config_path,
            pilot_root=pilot_root,
            candidates=all_candidates,
            winner=winner,
            keep_mode=auto_cfg.keep_pilots,
            catalog_root=cfg.motif_store.catalog_root,
        )
        if removed:
            logger.info(
                "Auto-opt pruned %d pilot run(s) (keep_pilots=%s).",
                len(removed),
                auto_cfg.keep_pilots,
            )
    logger.info(
        "Auto-opt final config saved -> %s",
        _format_run_path(config_used_path(final_run_dir), base=config_path.parent),
    )
    return final_run_dir


def _evaluate_pilot_run(
    run_dir: Path,
    kind: str,
    *,
    pilot_cfg: SampleConfig,
    auto_cfg: AutoOptConfig,
    pwms: dict[str, PWM],
    tf_names: list[str],
    budget: int | None = None,
    mode: str | None = None,
    cooling_boost: float = 1.0,
    move_profile: str = "balanced",
    swap_prob: float | None = None,
    ladder_size: int | None = None,
    move_probs: dict[str, float] | None = None,
    move_probs_label: str | None = None,
) -> AutoOptCandidate:
    manifest = load_manifest(run_dir)
    length = manifest.get("sequence_length")
    trace_file = trace_path(run_dir)
    seq_path = sequences_path(run_dir)
    if not trace_file.exists():
        raise FileNotFoundError(f"Missing trace.nc in {run_dir}")
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing sequences.parquet in {run_dir}")
    elite_path = find_elites_parquet(run_dir)
    seq_df = read_parquet(seq_path)
    elites_df = read_parquet(elite_path)
    tf_names = list(tf_names)
    top_k = int(auto_cfg.policy.scorecard.k)
    draw_scores = _draw_scores_from_sequences(seq_df)
    top_k_median_final = _top_k_median_from_scores(draw_scores, top_k)
    best_score_final = float(np.max(draw_scores)) if draw_scores.size else None
    selection_cfg = pilot_cfg.elites.selection
    dsdna_mode = resolve_dsdna_mode(
        elites_cfg=pilot_cfg.elites,
        bidirectional=pilot_cfg.objective.bidirectional,
    )
    pilot_score = None
    median_relevance_raw = None
    mean_pairwise_distance = None
    if selection_cfg.policy != "mmr":
        raise ValueError("auto_opt requires sample.elites.selection.policy=mmr.")
    pilot_score, median_relevance_raw, mean_pairwise_distance = _mmr_scorecard_from_elites(
        elites_df,
        tf_names=tf_names,
        selection_cfg=selection_cfg,
        auto_cfg=auto_cfg,
        pwms=pwms,
    )
    best_score = pilot_score
    bootstrap_ci: tuple[float, float] | None = None
    if draw_scores.size:
        seed = _bootstrap_seed(manifest=manifest, run_dir=run_dir, kind=kind)
        rng = np.random.default_rng(seed)
        bootstrap_ci = _bootstrap_top_k_ci(draw_scores, k=top_k, rng=rng)

    import arviz as az

    trace_idata = az.from_netcdf(trace_file)
    sample_meta = None
    sample_meta = {"top_k": top_k}
    if mode is not None:
        sample_meta["mode"] = mode
    sample_meta["optimizer_kind"] = kind
    sample_meta["dsdna_canonicalize"] = dsdna_mode
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=trace_idata,
        sequences_df=seq_df,
        elites_df=elites_df,
        tf_names=tf_names,
        optimizer=manifest.get("optimizer", {}),
        optimizer_stats=manifest.get("optimizer_stats", {}),
        sample_meta=sample_meta,
    )
    diagnostics_payload = diagnostics if isinstance(diagnostics, dict) else {}
    metrics = diagnostics_payload.get("metrics", {}) if isinstance(diagnostics_payload, dict) else {}
    trace_metrics = metrics.get("trace", {})
    if isinstance(trace_metrics, dict):
        trace_metrics = dict(trace_metrics)
        expected_draws = manifest.get("draws")
        expected_chains = manifest.get("chains")
        if expected_draws is not None:
            trace_metrics.setdefault("draws_expected", int(expected_draws))
        if expected_chains is not None:
            trace_metrics.setdefault("chains_expected", int(expected_chains))
        metrics = dict(metrics)
        metrics["trace"] = trace_metrics
        diagnostics_payload = dict(diagnostics_payload)
        diagnostics_payload["metrics"] = metrics
    diagnostics = diagnostics_payload if isinstance(diagnostics_payload, dict) else diagnostics
    seq_metrics = metrics.get("sequences", {})
    elites_metrics = metrics.get("elites", {})
    optimizer_metrics = metrics.get("optimizer", {})
    rhat = trace_metrics.get("rhat")
    ess = trace_metrics.get("ess")
    unique_fraction = seq_metrics.get("unique_fraction")
    balance_median = elites_metrics.get("normalized_min_median") or elites_metrics.get("balance_index_median")
    diversity = elites_metrics.get("diversity_hamming")
    improvement = trace_metrics.get("best_so_far_slope") or trace_metrics.get("score_delta")
    acceptance_b = None
    acceptance_m = None
    acceptance_mh = None
    delta_frac_zero_mh = None
    if isinstance(optimizer_metrics, dict):
        acc = optimizer_metrics.get("acceptance_rate") or {}
        if isinstance(acc, dict):
            acceptance_b = acc.get("B")
            acceptance_m = acc.get("M")
        acceptance_mh = optimizer_metrics.get("acceptance_rate_mh")
        delta_frac_zero_mh = optimizer_metrics.get("delta_frac_zero_mh")
    unique_successes = None
    if isinstance(optimizer_metrics, dict):
        raw_unique = optimizer_metrics.get("unique_successes")
        if isinstance(raw_unique, (int, float)):
            unique_successes = int(raw_unique)
    swap_rate = optimizer_metrics.get("swap_acceptance_rate") if isinstance(optimizer_metrics, dict) else None
    warnings = diagnostics.get("warnings", []) if isinstance(diagnostics, dict) else []
    status = "ok"
    if isinstance(diagnostics, dict):
        diag_status = diagnostics.get("status", "ok")
        if diag_status and diag_status != "ok":
            warnings = list(warnings)
            warnings.append(f"Diagnostics status={diag_status}; see diagnostics.json for details.")

    if draw_scores.size == 0:
        warnings = list(warnings)
        warnings.append("Missing pilot metric: combined_score_final draw-phase values.")
        status = "fail"
    if pilot_score is None:
        warnings = list(warnings)
        warnings.append("Missing pilot metric: elites_mmr scorecard unavailable.")
        status = "fail"

    if kind == "pt" and isinstance(optimizer_metrics, dict):
        swap_attempts = optimizer_metrics.get("swap_attempts")
        swap_prob = None
        optimizer_cfg = manifest.get("optimizer", {})
        if isinstance(optimizer_cfg, dict):
            pt_cfg = optimizer_cfg.get("pt", {})
            if isinstance(pt_cfg, dict):
                swap_prob = pt_cfg.get("swap_prob")
        if swap_prob and swap_attempts == 0:
            warnings = list(warnings)
            warnings.append("PT pilot recorded swap_prob>0 but swap_attempts=0.")

    if acceptance_mh is not None and acceptance_mh > 0.95 and delta_frac_zero_mh is not None:
        warnings = list(warnings)
        if delta_frac_zero_mh >= 0.8:
            warnings.append(
                "High MH acceptance likely due to hard-min plateaus (Δscore≈0). "
                "Enable objective.softmin and/or increase target_worst_tf_prob."
            )
        elif delta_frac_zero_mh < 0.5:
            warnings.append(
                "High MH acceptance with nontrivial Δscore suggests beta too low or moves too conservative. "
                "Increase beta schedule or move sizes."
            )

    return AutoOptCandidate(
        kind=kind,
        length=int(length) if isinstance(length, (int, float)) else None,
        budget=budget,
        cooling_boost=cooling_boost,
        move_profile=move_profile,
        swap_prob=swap_prob,
        ladder_size=ladder_size,
        move_probs=move_probs,
        move_probs_label=move_probs_label,
        run_dir=run_dir,
        run_dirs=[run_dir],
        best_score=best_score,
        top_k_median_final=top_k_median_final,
        best_score_final=best_score_final,
        top_k_ci_low=bootstrap_ci[0] if bootstrap_ci else None,
        top_k_ci_high=bootstrap_ci[1] if bootstrap_ci else None,
        rhat=rhat,
        ess=ess,
        unique_fraction=unique_fraction,
        balance_median=balance_median,
        diversity=diversity,
        improvement=improvement,
        acceptance_b=acceptance_b,
        acceptance_m=acceptance_m,
        acceptance_mh=acceptance_mh,
        swap_rate=swap_rate,
        status=status,
        quality=status,
        warnings=warnings,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
        pilot_score=pilot_score,
        median_relevance_raw=median_relevance_raw,
        mean_pairwise_distance=mean_pairwise_distance,
        unique_successes=unique_successes,
    )


def _aggregate_candidate_runs(
    runs: list[AutoOptCandidate],
    *,
    budget: int,
    scorecard_k: int,
) -> AutoOptCandidate:
    if not runs:
        raise ValueError("Auto-opt aggregation requires at least one pilot run")

    def _median(values: list[float | None]) -> float | None:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return None
        return float(np.median(filtered))

    kind = runs[0].kind
    length = runs[0].length
    move_profile = runs[0].move_profile
    swap_prob = runs[0].swap_prob
    ladder_size = runs[0].ladder_size
    move_probs = runs[0].move_probs
    move_probs_label = runs[0].move_probs_label
    run_dirs = [run.run_dir for run in runs]
    status = "ok"
    if all(run.status == "fail" for run in runs):
        status = "fail"

    warnings: list[str] = []
    for run in runs:
        warnings.extend(run.warnings)
    if status != "ok":
        warnings.append("One or more pilot replicates failed")

    diagnostics = {
        "replicates": [run.diagnostics for run in runs],
        "run_dirs": [str(path) for path in run_dirs],
    }
    trace_keys = ("draws", "chains", "draws_expected", "chains_expected", "ess_ratio")
    trace_metrics: dict[str, object] = {}
    for key in trace_keys:
        values: list[float] = []
        for run in runs:
            diag = run.diagnostics if isinstance(run.diagnostics, dict) else {}
            metrics = diag.get("metrics") if isinstance(diag, dict) else None
            if not isinstance(metrics, dict):
                continue
            trace = metrics.get("trace")
            if not isinstance(trace, dict):
                continue
            value = trace.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if values:
            median_val = float(np.median(values))
            if key in {"draws", "chains", "draws_expected", "chains_expected"}:
                trace_metrics[key] = int(round(median_val))
            else:
                trace_metrics[key] = median_val
    if trace_metrics:
        diagnostics["metrics"] = {"trace": trace_metrics}

    def _metric_or_neg(value: float | None) -> float:
        return float(value) if value is not None else float("-inf")

    best_run = max(
        runs,
        key=lambda run: (
            float(run.pilot_score) if run.pilot_score is not None else float("-inf"),
            _metric_or_neg(run.median_relevance_raw),
            _metric_or_neg(run.mean_pairwise_distance),
            str(run.run_dir),
        ),
    )

    pooled_scores: list[np.ndarray] = []
    manifests: list[dict[str, object]] = []
    for run_dir in run_dirs:
        seq_path = sequences_path(run_dir)
        if not seq_path.exists():
            warnings.append(f"Missing sequences.parquet for replicate {run_dir.name}")
            continue
        try:
            seq_df = read_parquet(seq_path)
        except Exception as exc:
            warnings.append(f"Failed to read sequences.parquet for {run_dir.name}: {exc}")
            continue
        scores = _draw_scores_from_sequences(seq_df)
        if scores.size:
            pooled_scores.append(scores)
        try:
            manifest = load_manifest(run_dir)
        except Exception as exc:
            warnings.append(f"Failed to read manifest for {run_dir.name}: {exc}")
        else:
            manifests.append(manifest)

    combined_scores = np.concatenate(pooled_scores) if pooled_scores else np.array([], dtype=float)
    pilot_score = _median([run.pilot_score for run in runs])
    if combined_scores.size:
        top_k_median_final = _top_k_median_from_scores(combined_scores, scorecard_k)
        best_score_final = float(np.max(combined_scores))
    else:
        top_k_median_final = _median([run.top_k_median_final for run in runs])
        best_score_final = max(
            [float(run.best_score_final) for run in runs if run.best_score_final is not None],
            default=None,
        )
    best_score = pilot_score
    bootstrap_ci: tuple[float, float] | None = None
    if combined_scores.size:
        seed = _pooled_bootstrap_seed(manifests=manifests, kind=kind, length=length, budget=budget)
        if seed is None:
            payload = {"kind": kind, "length": length, "budget": budget, "replicate_count": len(run_dirs)}
            seed = _bootstrap_seed_payload(payload)
        rng = np.random.default_rng(seed)
        bootstrap_ci = _bootstrap_top_k_ci(combined_scores, k=scorecard_k, rng=rng)
    else:
        lows = [run.top_k_ci_low for run in runs if run.top_k_ci_low is not None]
        highs = [run.top_k_ci_high for run in runs if run.top_k_ci_high is not None]
        if lows and highs:
            bootstrap_ci = (float(min(lows)), float(max(highs)))

    return AutoOptCandidate(
        kind=kind,
        length=length,
        budget=budget,
        cooling_boost=float(np.median([run.cooling_boost for run in runs])),
        move_profile=move_profile,
        swap_prob=swap_prob,
        ladder_size=ladder_size,
        move_probs=move_probs,
        move_probs_label=move_probs_label,
        run_dir=best_run.run_dir,
        run_dirs=run_dirs,
        best_score=best_score,
        top_k_median_final=top_k_median_final,
        best_score_final=best_score_final,
        top_k_ci_low=bootstrap_ci[0] if bootstrap_ci else None,
        top_k_ci_high=bootstrap_ci[1] if bootstrap_ci else None,
        rhat=_median([run.rhat for run in runs]),
        ess=_median([run.ess for run in runs]),
        unique_fraction=_median([run.unique_fraction for run in runs]),
        balance_median=_median([run.balance_median for run in runs]),
        diversity=_median([run.diversity for run in runs]),
        improvement=_median([run.improvement for run in runs]),
        acceptance_b=_median([run.acceptance_b for run in runs]),
        acceptance_m=_median([run.acceptance_m for run in runs]),
        acceptance_mh=_median([run.acceptance_mh for run in runs]),
        swap_rate=_median([run.swap_rate for run in runs]),
        status=status,
        quality=status,
        warnings=warnings,
        diagnostics=diagnostics,
        pilot_score=_median([run.pilot_score for run in runs]),
        median_relevance_raw=_median([run.median_relevance_raw for run in runs]),
        mean_pairwise_distance=_median([run.mean_pairwise_distance for run in runs]),
        unique_successes=_median([run.unique_successes for run in runs]),
    )
