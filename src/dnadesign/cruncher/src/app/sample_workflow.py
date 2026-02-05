"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample_workflow.py

Coordinate sampling runs across regulator sets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import signal
from contextlib import contextmanager
from pathlib import Path

from dnadesign.cruncher.app.sample.artifacts import (
    _auto_opt_best_marker_path,
    _elite_parquet_schema,
    _format_run_path,
    _prune_auto_opt_runs,
    _save_config,
    _selected_config_payload,
    _write_auto_opt_best_marker,
    _write_length_ladder_table,
    _write_parquet_rows,
)
from dnadesign.cruncher.app.sample.auto_opt import (
    AutoOptCandidate,
    AutoOptSpec,
    _aggregate_candidate_runs,
    _assess_candidate_quality,
    _budget_to_tune_draws,
    _build_final_sample_cfg,
    _build_pilot_sample_cfg,
    _candidate_lengths,
    _confidence_from_candidates,
    _pilot_budget_levels,
    _run_auto_optimize_for_set,
    _select_auto_opt_candidate,
    _stable_pilot_seed,
    _validate_auto_opt_candidates,
    _warm_start_seeds_from_elites,
    _warm_start_seeds_from_sequences,
)
from dnadesign.cruncher.app.sample.diagnostics import (
    _best_score_final_from_sequences,
    _bootstrap_seed,
    _bootstrap_seed_payload,
    _bootstrap_top_k_ci,
    _draw_scores_from_sequences,
    _elite_filter_passes,
    _elite_rank_key,
    _EliteCandidate,
    _filter_elite_candidates,
    _norm_map_for_elites,
    _pooled_bootstrap_seed,
    _top_k_median_from_scores,
    _top_k_median_from_sequences,
)
from dnadesign.cruncher.app.sample.optimizer_config import (
    _boost_beta_ladder,
    _default_beta_ladder,
    _effective_chain_count,
    _format_auto_opt_config_summary,
    _format_beta_ladder_summary,
    _format_move_probs,
    _resolve_beta_ladder,
    _resolve_optimizer_kind,
)
from dnadesign.cruncher.app.sample.resources import _load_pwms_for_set, _lockmap_for, _store
from dnadesign.cruncher.app.sample.run_set import _resolve_final_softmin_beta, _run_sample_for_set
from dnadesign.cruncher.app.target_service import (
    has_blocking_target_errors,
    target_statuses,
)
from dnadesign.cruncher.artifacts.layout import config_used_path
from dnadesign.cruncher.config.schema_v2 import AutoOptConfig, CruncherConfig
from dnadesign.cruncher.core.labels import regulator_sets
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

logger = logging.getLogger(__name__)

__all__ = [
    "AutoOptCandidate",
    "AutoOptSpec",
    "_EliteCandidate",
    "_aggregate_candidate_runs",
    "_assess_candidate_quality",
    "_auto_opt_best_marker_path",
    "_best_score_final_from_sequences",
    "_bootstrap_seed",
    "_bootstrap_seed_payload",
    "_bootstrap_top_k_ci",
    "_boost_beta_ladder",
    "_budget_to_tune_draws",
    "_build_final_sample_cfg",
    "_build_pilot_sample_cfg",
    "_candidate_lengths",
    "_confidence_from_candidates",
    "_default_beta_ladder",
    "_draw_scores_from_sequences",
    "_effective_chain_count",
    "_elite_filter_passes",
    "_elite_parquet_schema",
    "_elite_rank_key",
    "_filter_elite_candidates",
    "_format_auto_opt_config_summary",
    "_format_beta_ladder_summary",
    "_format_move_probs",
    "_format_run_path",
    "_load_pwms_for_set",
    "_lockmap_for",
    "_norm_map_for_elites",
    "_pilot_budget_levels",
    "_pooled_bootstrap_seed",
    "_prune_auto_opt_runs",
    "_resolve_beta_ladder",
    "_resolve_final_softmin_beta",
    "_resolve_optimizer_kind",
    "_run_auto_optimize_for_set",
    "_run_sample_for_set",
    "_save_config",
    "_select_auto_opt_candidate",
    "_selected_config_payload",
    "_sigterm_as_keyboard_interrupt",
    "_stable_pilot_seed",
    "_store",
    "_top_k_median_from_scores",
    "_top_k_median_from_sequences",
    "_validate_auto_opt_candidates",
    "_warm_start_seeds_from_elites",
    "_warm_start_seeds_from_sequences",
    "_write_auto_opt_best_marker",
    "_write_length_ladder_table",
    "_write_parquet_rows",
    "run_sample",
]


@contextmanager
def _sigterm_as_keyboard_interrupt():
    sigterm = signal.SIGTERM
    previous = signal.getsignal(sigterm)

    def _handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(sigterm, _handler)
    try:
        yield
    finally:
        signal.signal(sigterm, previous)


def run_sample(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    auto_opt_override: bool | None = None,
) -> None:
    """
    Run MCMC sampler, save config/meta plus artifacts (trace.nc, sequences.parquet, elites.*).
    Each regulator set is sampled independently for clear provenance.
    """
    if cfg.sample is None:
        raise ValueError("sample section is required for sample")
    with _sigterm_as_keyboard_interrupt():
        ensure_mpl_cache(resolve_catalog_root(config_path, cfg.motif_store.catalog_root))
        lockmap = _lockmap_for(cfg, config_path)
        statuses = target_statuses(cfg=cfg, config_path=config_path)
        sample_cfg = cfg.sample
        auto_cfg = sample_cfg.auto_opt
        if auto_opt_override is not None:
            if auto_opt_override:
                sample_cfg.optimizer.name = "auto"
                if auto_cfg is None:
                    auto_cfg = AutoOptConfig()
                else:
                    auto_cfg = auto_cfg.model_copy(update={"enabled": True})
            else:
                auto_cfg = None
                if sample_cfg.optimizer.name == "auto":
                    raise ValueError(
                        "Auto-opt disabled by flag, but sample.optimizer.name='auto' is not allowed. "
                        "Set sample.optimizer.name to 'pt' (or remove --no-auto-opt)."
                    )
        use_auto_opt = sample_cfg.optimizer.name == "auto" and auto_cfg is not None and auto_cfg.enabled
        groups = regulator_sets(cfg.regulator_sets)
        set_count = len(groups)
        include_set_index = set_count > 1
        for set_index, group in enumerate(groups, start=1):
            if not group:
                raise ValueError(f"regulator_sets[{set_index}] is empty.")
            seen: set[str] = set()
            tfs = [tf for tf in group if not (tf in seen or seen.add(tf))]
            subset = [status for status in statuses if status.set_index == set_index]
            if has_blocking_target_errors(subset):
                details = "; ".join(f"{s.tf_name}:{s.status}" for s in subset if s.status not in {"ready", "warning"})
                raise ValueError(
                    f"Target readiness failed for set {set_index} ({details}). "
                    f"Run `cruncher targets status {config_path.name}` for details."
                )
            if use_auto_opt:
                run_dir = _run_auto_optimize_for_set(
                    cfg,
                    config_path,
                    set_index=set_index,
                    set_count=set_count,
                    include_set_index=include_set_index,
                    tfs=tfs,
                    lockmap=lockmap,
                    sample_cfg=sample_cfg,
                    auto_cfg=auto_cfg,
                )
            else:
                run_dir = _run_sample_for_set(
                    cfg,
                    config_path,
                    set_index=set_index,
                    set_count=set_count,
                    include_set_index=include_set_index,
                    tfs=tfs,
                    lockmap=lockmap,
                    sample_cfg=sample_cfg,
                )
            logger.info(
                "Sample outputs -> %s",
                _format_run_path(run_dir, base=config_path.parent),
            )
            logger.info(
                "Config used -> %s",
                _format_run_path(config_used_path(run_dir), base=config_path.parent),
            )
