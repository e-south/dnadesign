"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_plan_setup.py

Stage-B plan setup and pool-loading helpers used by the orchestrator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import ResolvedPlanItem
from ..artifacts.pool import POOL_MODE_SEQUENCE, POOL_MODE_TFBS, PoolData
from ..input_types import PWM_INPUT_TYPES
from ..runtime_policy import RuntimePolicy
from .deps import PipelineDeps
from .inputs import (
    _budget_attr,
    _build_input_manifest_entry,
    _input_metadata,
    _mining_attr,
    _sampling_attr,
)
from .plan_context import PlanExecutionState, PlanRunContext
from .progress import _summarize_tf_counts
from .progress_runtime import _init_progress_settings
from .stage_b import _fixed_elements_label
from .stage_b_runtime_types import (
    PadSettings,
    PlanInputState,
    PlanRunSettings,
    RuntimeSettings,
    SolverSettings,
)
from .stage_b_solution_rejections import resolve_failed_solution_cap

log = logging.getLogger(__name__)


def _init_plan_settings(
    *,
    source_cfg,
    plan_item: ResolvedPlanItem,
    context: PlanRunContext,
    execution_state: PlanExecutionState,
    one_subsample_only: bool,
    plan_started_at: float | None = None,
) -> PlanRunSettings:
    source_label = source_cfg.name
    plan_name = plan_item.name
    quota = int(plan_item.quota)
    global_cfg = context.global_cfg

    gen = global_cfg.generation
    seq_len = int(gen.sequence_length)
    sampling_cfg = gen.sampling
    pool_strategy = str(sampling_cfg.pool_strategy)

    runtime_cfg = global_cfg.runtime
    effective_max_failed_solutions = resolve_failed_solution_cap(
        max_failed_solutions=int(runtime_cfg.max_failed_solutions),
        max_failed_solutions_per_target=float(runtime_cfg.max_failed_solutions_per_target),
        quota=quota,
    )
    max_per_subsample = int(runtime_cfg.arrays_generated_before_resample)
    if pool_strategy != "iterative_subsample" and not one_subsample_only:
        max_per_subsample = quota
    runtime = RuntimeSettings(
        max_per_subsample=max_per_subsample,
        min_count_per_tf=int(runtime_cfg.min_count_per_tf),
        max_dupes=int(runtime_cfg.max_duplicate_solutions),
        stall_seconds=int(runtime_cfg.stall_seconds_before_resample),
        stall_warn_every=int(runtime_cfg.stall_warning_every_seconds),
        max_consecutive_failures=int(runtime_cfg.max_consecutive_failures),
        max_seconds_per_plan=int(runtime_cfg.max_seconds_per_plan),
        max_failed_solutions=effective_max_failed_solutions,
        leaderboard_every=int(runtime_cfg.leaderboard_every),
        checkpoint_every=int(execution_state.checkpoint_every or 0),
    )

    post = global_cfg.postprocess
    pad_cfg = post.pad
    pad_gc_cfg = pad_cfg.gc
    pad = PadSettings(
        enabled=pad_cfg.mode != "off",
        mode=pad_cfg.mode,
        end=pad_cfg.end,
        gc_mode=pad_gc_cfg.mode,
        gc_min=float(pad_gc_cfg.min),
        gc_max=float(pad_gc_cfg.max),
        gc_target=float(pad_gc_cfg.target),
        gc_tolerance=float(pad_gc_cfg.tolerance),
        gc_min_length=int(pad_gc_cfg.min_pad_length),
        max_tries=int(pad_cfg.max_tries),
    )

    solver_cfg = global_cfg.solver
    solver = SolverSettings(
        strategy=str(solver_cfg.strategy),
        strands=str(solver_cfg.strands),
        time_limit_seconds=(
            float(solver_cfg.time_limit_seconds) if solver_cfg.time_limit_seconds is not None else None
        ),
        threads=int(solver_cfg.threads) if solver_cfg.threads is not None else None,
    )

    extra_library_label = _fixed_elements_label(plan_item.fixed_elements)
    log_cfg = global_cfg.logging
    progress = _init_progress_settings(
        log_cfg=log_cfg,
        source_label=source_label,
        plan_name=plan_name,
        quota=quota,
        max_per_subsample=runtime.max_per_subsample,
        show_tfbs=context.show_tfbs,
        show_solutions=context.show_solutions,
        extra_library_label=extra_library_label,
        shared_dashboard=execution_state.shared_dashboard,
    )

    policy_pad = str(pad.mode)
    policy_sampling = pool_strategy
    policy_solver = solver.strategy
    policy = RuntimePolicy(
        pool_strategy=pool_strategy,
        arrays_generated_before_resample=runtime.max_per_subsample,
        stall_seconds_before_resample=runtime.stall_seconds,
        stall_warning_every_seconds=runtime.stall_warn_every,
        max_consecutive_failures=runtime.max_consecutive_failures,
        max_seconds_per_plan=runtime.max_seconds_per_plan,
    )

    return PlanRunSettings(
        source_label=source_label,
        plan_name=plan_name,
        quota=quota,
        seq_len=seq_len,
        sampling_cfg=sampling_cfg,
        pool_strategy=pool_strategy,
        runtime=runtime,
        pad=pad,
        solver=solver,
        extra_library_label=extra_library_label,
        progress=progress,
        policy=policy,
        policy_pad=policy_pad,
        policy_sampling=policy_sampling,
        policy_solver=policy_solver,
        plan_start=float(plan_started_at) if plan_started_at is not None else time.monotonic(),
    )


def _load_plan_pool(
    *,
    source_cfg,
    cfg_path: Path,
    deps: PipelineDeps,
    np_rng: np.random.Generator,
    outputs_root: Path,
    run_id: str,
    pool_override: PoolData | None,
    source_cache: dict[str, PoolData] | None,
    input_meta_override: dict | None,
    inputs_manifest: dict[str, dict] | None,
    source_label: str,
    plan_name: str,
    progress_style: str,
    display_tf_label,
) -> PlanInputState:
    cache_key = source_label
    cached = source_cache.get(cache_key) if source_cache is not None else None
    if pool_override is not None:
        pool = pool_override
        if source_cache is not None:
            source_cache[cache_key] = pool
    elif cached is None:
        src_obj = deps.source_factory(source_cfg, cfg_path)
        data_entries, meta_df, _summaries = src_obj.load_data(
            rng=np_rng,
            outputs_root=outputs_root,
            run_id=str(run_id),
        )
        if meta_df is not None and isinstance(meta_df, pd.DataFrame):
            sequences = meta_df["tfbs"].tolist() if "tfbs" in meta_df.columns else list(data_entries or [])
            pool = PoolData(
                name=source_label,
                input_type=str(getattr(source_cfg, "type", "")),
                pool_mode=POOL_MODE_TFBS,
                df=meta_df,
                sequences=sequences,
                pool_path=Path("."),
            )
        else:
            pool = PoolData(
                name=source_label,
                input_type=str(getattr(source_cfg, "type", "")),
                pool_mode=POOL_MODE_SEQUENCE,
                df=None,
                sequences=list(data_entries or []),
                pool_path=Path("."),
            )
        if source_cache is not None:
            source_cache[cache_key] = pool
    else:
        pool = cached

    data_entries = pool.sequences
    meta_df = pool.df
    input_meta = dict(input_meta_override) if input_meta_override is not None else _input_metadata(source_cfg, cfg_path)
    input_tf_tfbs_pair_count: int | None = None
    if meta_df is not None and isinstance(meta_df, pd.DataFrame):
        input_row_count = int(len(meta_df))
        input_tf_count = int(meta_df["tf"].nunique()) if "tf" in meta_df.columns else 0
        input_tfbs_count = int(meta_df["tfbs"].nunique()) if "tfbs" in meta_df.columns else 0
        if "tf" in meta_df.columns and "tfbs" in meta_df.columns:
            input_tf_tfbs_pair_count = int(meta_df.drop_duplicates(["tf", "tfbs"]).shape[0])
    else:
        input_row_count = int(len(data_entries))
        input_tf_count = 0
        input_tfbs_count = int(len(set(data_entries))) if data_entries else 0
        input_tf_tfbs_pair_count = None
    input_meta.update(
        {
            "input_row_count": input_row_count,
            "input_tf_count": input_tf_count,
            "input_tfbs_count": input_tfbs_count,
            "input_tf_tfbs_pair_count": input_tf_tfbs_pair_count,
            "sampling_fraction": None,
            "sampling_fraction_pairs": None,
        }
    )
    pair_label = str(input_tf_tfbs_pair_count) if input_tf_tfbs_pair_count is not None else "-"
    if progress_style == "stream":
        log.info(
            "[%s/%s] Input summary: mode=%s rows=%d tfs=%d tfbs=%d pairs=%s",
            source_label,
            plan_name,
            input_meta.get("input_mode"),
            input_row_count,
            input_tf_count,
            input_tfbs_count,
            pair_label,
        )

    source_type = getattr(source_cfg, "type", None)
    if source_type in PWM_INPUT_TYPES and meta_df is not None and "tf" in meta_df.columns:
        input_meta["input_pwm_ids"] = sorted(set(meta_df["tf"].tolist()))
        if inputs_manifest is not None and source_label not in inputs_manifest:
            input_sampling_cfg = getattr(source_cfg, "sampling", None)
            strategy = _sampling_attr(input_sampling_cfg, "strategy")
            length_cfg = _sampling_attr(input_sampling_cfg, "length")
            length_policy = _sampling_attr(length_cfg, "policy")
            length_range = _sampling_attr(length_cfg, "range")
            mining_cfg = _sampling_attr(input_sampling_cfg, "mining")
            mining_batch_size = _mining_attr(mining_cfg, "batch_size")
            mining_log_every = _mining_attr(mining_cfg, "log_every_batches")
            budget_mode = _budget_attr(mining_cfg, "mode")
            budget_candidates = _budget_attr(mining_cfg, "candidates")
            budget_target_tier_fraction = _budget_attr(mining_cfg, "target_tier_fraction")
            budget_max_candidates = _budget_attr(mining_cfg, "max_candidates")
            budget_max_seconds = _budget_attr(mining_cfg, "max_seconds")
            if length_range is not None:
                length_range = list(length_range)
            score_label = "best_hit_score>0"
            tiers_label = "pct_0.1_1_9"
            length_label = str(length_policy)
            if length_policy == "range" and length_range:
                length_label = f"{length_policy}({length_range[0]}..{length_range[1]})"
            counts_label = _summarize_tf_counts([display_tf_label(tf) for tf in meta_df["tf"].tolist()])
            mining_label = "-"
            if mining_cfg is not None:
                parts = []
                if mining_batch_size is not None:
                    parts.append(f"batch={mining_batch_size}")
                if budget_max_seconds is not None:
                    parts.append(f"max_seconds={budget_max_seconds}s")
                if mining_log_every is not None:
                    parts.append(f"log_every={mining_log_every}")
                mining_label = ", ".join(parts) if parts else "enabled"
            budget_label = "-"
            if budget_mode == "fixed_candidates":
                budget_label = f"fixed={budget_candidates}"
            elif budget_mode == "tier_target":
                tier_label = (
                    f"{float(budget_target_tier_fraction) * 100:.3f}%"
                    if budget_target_tier_fraction is not None
                    else "unset"
                )
                budget_label = f"tier={tier_label} max_candidates={budget_max_candidates}"
            if progress_style == "stream":
                log.info(
                    "Stage-A PWM sampling for %s: motifs=%d | sites=%s | strategy=%s | backend=%s | "
                    "eligibility=%s | tiers=%s | mining=%s | budget=%s | length=%s",
                    source_label,
                    len(input_meta.get("input_pwm_ids") or []),
                    counts_label or "-",
                    strategy,
                    "fimo",
                    score_label,
                    tiers_label,
                    mining_label,
                    budget_label,
                    length_label,
                )
            inputs_manifest[source_label] = _build_input_manifest_entry(
                source_cfg=source_cfg,
                cfg_path=cfg_path,
                input_meta=input_meta,
                input_row_count=input_row_count,
                input_tf_count=input_tf_count,
                input_tfbs_count=input_tfbs_count,
                input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
                meta_df=meta_df,
            )
    elif inputs_manifest is not None and source_label not in inputs_manifest:
        inputs_manifest[source_label] = _build_input_manifest_entry(
            source_cfg=source_cfg,
            cfg_path=cfg_path,
            input_meta=input_meta,
            input_row_count=input_row_count,
            input_tf_count=input_tf_count,
            input_tfbs_count=input_tfbs_count,
            input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
            meta_df=meta_df,
        )

    return PlanInputState(
        pool=pool,
        data_entries=data_entries,
        meta_df=meta_df,
        input_meta=input_meta,
        input_row_count=input_row_count,
        input_tf_count=input_tf_count,
        input_tfbs_count=input_tfbs_count,
        input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
    )
