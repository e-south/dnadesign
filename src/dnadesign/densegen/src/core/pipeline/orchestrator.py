"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/orchestrator.py

DenseGen pipeline orchestration (CLI-agnostic).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd
from rich.console import Console

from ...adapters.optimizer import DenseArraysAdapter, OptimizerAdapter
from ...adapters.outputs import OutputRecord, SinkBase, build_sinks, resolve_bio_alphabet
from ...adapters.sources import data_source_factory
from ...adapters.sources.base import BaseDataSource
from ...config import (
    DenseGenConfig,
    LoadedConfig,
    ResolvedPlanItem,
    resolve_run_root,
)
from ...utils import logging_utils
from ...utils.logging_utils import install_native_stderr_filters
from ...utils.sequence_utils import gc_fraction
from ..artifacts.library import LibraryRecord
from ..artifacts.pool import POOL_MODE_SEQUENCE, POOL_MODE_TFBS, PoolData, _hash_file
from ..input_types import PWM_INPUT_TYPES
from ..metadata import build_metadata
from ..motif_labels import motif_display_name
from ..postprocess import generate_pad
from ..run_manifest import PlanManifest, RunManifest
from ..run_metrics import write_run_metrics
from ..run_paths import (
    candidates_root,
    display_path,
    ensure_run_meta_dir,
    has_existing_run_outputs,
    inputs_manifest_path,
    run_manifest_path,
    run_outputs_root,
    run_tables_root,
)
from ..runtime_policy import RuntimePolicy
from ..seeding import derive_seed_map
from .attempts import (
    _append_attempt,
    _flush_attempts,
    _flush_solutions,
    _load_existing_library_index,
    _log_rejection,
)
from .inputs import (
    _budget_attr,
    _build_input_manifest_entry,
    _input_metadata,
    _mining_attr,
    _sampling_attr,
)
from .library_artifacts import prepare_library_source, write_library_artifacts
from .outputs import (
    _assert_sink_alignment,
    _consolidate_parts,
    _emit_event,
    _write_effective_config,
    _write_to_sinks,
)
from .plan_execution import run_plan_schedule
from .plan_pools import PLAN_POOL_INPUT_TYPE, PlanPoolSpec
from .progress import (
    _build_screen_dashboard,
    _format_progress_bar,
    _ScreenDashboard,
    _short_seq,
    _summarize_diversity,
    _summarize_failure_leaderboard,
    _summarize_failure_totals,
    _summarize_leaderboard,
    _summarize_tf_counts,
    _summarize_tfbs_usage_stats,
)
from .resume_state import load_resume_state
from .run_setup import build_display_map_by_input, init_plan_stats, init_state_counts
from .run_state_manager import assert_state_matches_outputs, init_run_state, write_run_state
from .sampling_diagnostics import SamplingDiagnostics
from .sequence_validation import (
    _apply_pad_offsets,
    _find_forbidden_kmer,
    _promoter_windows,
)
from .stage_a_pools import prepare_stage_a_pools
from .stage_b import (
    _compute_sampling_fraction,
    _compute_sampling_fraction_pairs,
    _fixed_elements_dump,
    _merge_min_counts,
    _min_count_by_regulator,
)
from .stage_b_library_builder import LibraryBuilder, LibraryContext
from .stage_b_sampler import LibraryRunResult, SamplingCounters, StageBSampler
from .usage_tracking import _compute_used_tf_info
from .versioning import _resolve_dense_arrays_version

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    total_generated: int
    per_plan: dict[tuple[str, str], int]


def _candidate_logging_enabled(cfg: DenseGenConfig) -> bool:
    for inp in cfg.inputs:
        sampling = getattr(inp, "sampling", None)
        if sampling is None:
            continue
        if getattr(sampling, "keep_all_candidates_debug", False):
            return True
    return False


def _plan_pool_input_meta(spec: PlanPoolSpec) -> dict:
    meta = {
        "input_type": PLAN_POOL_INPUT_TYPE,
        "input_name": spec.pool_name,
        "input_source_names": list(spec.include_inputs),
    }
    if spec.pool.pool_mode == POOL_MODE_TFBS:
        meta["input_mode"] = "binding_sites"
        if spec.pool.df is not None and "tf" in spec.pool.df.columns:
            meta["input_pwm_ids"] = sorted(set(spec.pool.df["tf"].tolist()))
        else:
            meta["input_pwm_ids"] = []
    else:
        meta["input_mode"] = "sequence_library"
        meta["input_pwm_ids"] = []
    return meta


@dataclass(frozen=True)
class PipelineDeps:
    source_factory: Callable[[object, Path], BaseDataSource]
    sink_factory: Callable[[DenseGenConfig, Path], Iterable[SinkBase]]
    optimizer: OptimizerAdapter
    pad: Callable[..., tuple[str, dict] | str]


def default_deps() -> PipelineDeps:
    return PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=build_sinks,
        optimizer=DenseArraysAdapter(),
        pad=generate_pad,
    )


def resolve_plan(loaded: LoadedConfig) -> List[ResolvedPlanItem]:
    return loaded.root.densegen.generation.resolve_plan()


def select_solver(
    preferred: str | None,
    optimizer: OptimizerAdapter,
    *,
    strategy: str,
    test_length: int = 10,
) -> str | None:
    """Probe the requested solver once and fail fast if unavailable."""
    if strategy == "approximate":
        return preferred
    if not preferred:
        raise ValueError("solver.backend is required unless strategy=approximate")
    try:
        optimizer.probe_solver(preferred, test_length=test_length)
        return preferred
    except Exception as exc:
        raise RuntimeError(
            f"Requested solver '{preferred}' failed during probe: {exc}\n"
            "Please install/configure this solver or choose another in solver.backend."
        ) from exc


def _process_plan_for_source(
    source_cfg,
    plan_item: ResolvedPlanItem,
    global_cfg,
    sinks,
    *,
    chosen_solver: str | None,
    deps: PipelineDeps,
    rng: random.Random,
    np_rng: np.random.Generator,
    cfg_path: Path,
    run_id: str,
    run_root: str,
    run_config_path: str,
    run_config_sha256: str,
    random_seed: int,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    show_tfbs: bool,
    show_solutions: bool,
    output_bio_type: str,
    output_alphabet: str,
    one_subsample_only: bool = False,
    already_generated: int = 0,
    inputs_manifest: dict[str, dict] | None = None,
    existing_usage_counts: dict[tuple[str, str], int] | None = None,
    state_counts: dict[tuple[str, str], int] | None = None,
    checkpoint_every: int = 0,
    write_state: Callable[[], None] | None = None,
    site_failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] | None = None,
    source_cache: dict[str, PoolData] | None = None,
    pool_override: PoolData | None = None,
    input_meta_override: dict | None = None,
    attempt_counters: dict[tuple[str, str], int] | None = None,
    library_records: dict[tuple[str, str], list[LibraryRecord]] | None = None,
    library_cursor: dict[tuple[str, str], int] | None = None,
    library_source: str | None = None,
    library_build_rows: list[dict] | None = None,
    library_member_rows: list[dict] | None = None,
    solution_rows: list[dict] | None = None,
    composition_rows: list[dict] | None = None,
    events_path: Path | None = None,
    display_map_by_input: dict[str, dict[str, str]] | None = None,
) -> tuple[int, dict]:
    source_label = source_cfg.name
    plan_name = plan_item.name
    quota = int(plan_item.quota)
    attempt_counters = attempt_counters or {}
    display_map = display_map_by_input.get(source_label, {}) if display_map_by_input else {}

    def _display_tf_label(label: str) -> str:
        if not label:
            return label
        if label in display_map:
            return display_map[label]
        return motif_display_name(label, None)

    def _next_attempt_index() -> int:
        key = (source_label, plan_name)
        current = int(attempt_counters.get(key, 0)) + 1
        attempt_counters[key] = current
        return current

    gen = global_cfg.generation
    seq_len = int(gen.sequence_length)
    sampling_cfg = gen.sampling
    pool_strategy = str(sampling_cfg.pool_strategy)

    runtime_cfg = global_cfg.runtime
    max_per_subsample = int(runtime_cfg.arrays_generated_before_resample)
    min_count_per_tf = int(runtime_cfg.min_count_per_tf)
    max_dupes = int(runtime_cfg.max_duplicate_solutions)
    stall_seconds = int(runtime_cfg.stall_seconds_before_resample)
    stall_warn_every = int(runtime_cfg.stall_warning_every_seconds)
    max_consecutive_failures = int(runtime_cfg.max_consecutive_failures)
    max_seconds_per_plan = int(runtime_cfg.max_seconds_per_plan)
    max_failed_solutions = int(runtime_cfg.max_failed_solutions)
    leaderboard_every = int(runtime_cfg.leaderboard_every)
    checkpoint_every = int(checkpoint_every or 0)

    post = global_cfg.postprocess
    pad_cfg = post.pad
    pad_enabled = pad_cfg.mode != "off"
    pad_mode = pad_cfg.mode
    pad_end = pad_cfg.end
    pad_gc_cfg = pad_cfg.gc
    pad_gc_mode = pad_gc_cfg.mode
    pad_gc_min = float(pad_gc_cfg.min)
    pad_gc_max = float(pad_gc_cfg.max)
    pad_gc_target = float(pad_gc_cfg.target)
    pad_gc_tolerance = float(pad_gc_cfg.tolerance)
    pad_gc_min_length = int(pad_gc_cfg.min_pad_length)
    pad_max_tries = int(pad_cfg.max_tries)
    validate_cfg = getattr(post, "validate_final_sequence", None)
    forbid_kmers_cfg = getattr(validate_cfg, "forbid_kmers_outside_promoter_windows", None) if validate_cfg else None
    forbid_kmers = list(getattr(forbid_kmers_cfg, "kmers", []) or [])

    solver_cfg = global_cfg.solver
    solver_strategy = str(solver_cfg.strategy)
    solver_strands = str(solver_cfg.strands)
    solver_time_limit_seconds = (
        float(solver_cfg.time_limit_seconds) if solver_cfg.time_limit_seconds is not None else None
    )
    solver_threads = int(solver_cfg.threads) if solver_cfg.threads is not None else None

    log_cfg = global_cfg.logging
    print_visual = bool(log_cfg.print_visual)
    progress_style = str(getattr(log_cfg, "progress_style", "stream"))
    progress_every = int(getattr(log_cfg, "progress_every", 1))
    progress_refresh_seconds = float(getattr(log_cfg, "progress_refresh_seconds", 1.0))
    logging_utils.set_progress_style(progress_style)
    logging_utils.set_progress_enabled(progress_style in {"stream", "screen"})
    screen_console = None
    if progress_style == "screen":
        tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        pixi_shell = os.environ.get("PIXI_IN_SHELL") == "1"
        if tty and not pixi_shell:
            screen_console = Console()
        else:
            width = shutil.get_terminal_size(fallback=(140, 40)).columns
            screen_console = Console(file=sys.stdout, width=int(width), force_terminal=False)
            log.warning("progress_style=screen requires an interactive terminal; using static output.")
    last_screen_refresh = 0.0
    latest_failure_totals: str | None = None
    show_tfbs = bool(show_tfbs or getattr(log_cfg, "show_tfbs", False))
    show_solutions = bool(show_solutions or getattr(log_cfg, "show_solutions", False))
    dashboard = (
        _ScreenDashboard(console=screen_console, refresh_seconds=progress_refresh_seconds)
        if progress_style == "screen" and screen_console is not None
        else None
    )
    cr_sum = 0.0
    cr_count = 0

    policy_pad = str(pad_mode)
    policy_sampling = pool_strategy
    policy_solver = solver_strategy

    policy = RuntimePolicy(
        pool_strategy=pool_strategy,
        arrays_generated_before_resample=max_per_subsample,
        stall_seconds_before_resample=stall_seconds,
        stall_warning_every_seconds=stall_warn_every,
        max_consecutive_failures=max_consecutive_failures,
        max_seconds_per_plan=max_seconds_per_plan,
    )

    plan_start = time.monotonic()
    counters = SamplingCounters()
    failed_solutions = 0
    duplicate_records = 0
    stall_events = 0
    failed_min_count_per_tf = 0
    failed_required_regulators = 0
    failed_min_count_by_regulator = 0
    failed_min_required_regulators = 0
    duplicate_solutions = 0
    usage_counts: dict[tuple[str, str], int] = dict(existing_usage_counts or {})
    tf_usage_counts: dict[str, int] = {}
    for (tf, _tfbs), count in usage_counts.items():
        tf_usage_counts[tf] = tf_usage_counts.get(tf, 0) + int(count)
    track_failures = site_failure_counts is not None
    failure_counts = site_failure_counts if site_failure_counts is not None else {}
    attempts_buffer: list[dict] = []
    run_root_path = Path(run_root)
    outputs_root = run_outputs_root(run_root_path)
    tables_root = run_tables_root(run_root_path)
    existing_library_builds = _load_existing_library_index(tables_root)

    # Load source (cache Stage-A PWM sampling results across round-robin passes).
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
    if progress_style != "screen":
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
            counts_label = _summarize_tf_counts([_display_tf_label(tf) for tf in meta_df["tf"].tolist()])
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
            if progress_style != "screen":
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
    fixed_elements = plan_item.fixed_elements
    constraints = plan_item.regulator_constraints
    plan_min_count_by_regulator = dict(constraints.min_count_by_regulator or {})
    fixed_elements_dump = _fixed_elements_dump(fixed_elements)

    libraries_built = existing_library_builds
    libraries_built_start = existing_library_builds
    libraries_used = 0
    library_source_label = str(library_source or getattr(sampling_cfg, "library_source", "build")).lower()
    if library_source_label not in {"build", "artifact"}:
        raise ValueError(f"Unsupported Stage-B sampling.library_source: {library_source_label}")
    if library_source_label == "artifact" and library_cursor is not None:
        prior_used = int(library_cursor.get((source_label, plan_name), 0))
        libraries_built = prior_used
        libraries_built_start = prior_used
    library_sampling_strategy = str(sampling_cfg.library_sampling_strategy)
    iterative_max_libraries = int(sampling_cfg.iterative_max_libraries)
    iterative_min_new_solutions = int(sampling_cfg.iterative_min_new_solutions)

    if pool_strategy != "iterative_subsample" and not one_subsample_only:
        max_per_subsample = quota

    builder = LibraryBuilder(
        source_label=source_label,
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=seq_len,
        min_count_per_tf=min_count_per_tf,
        usage_counts=usage_counts,
        failure_counts=failure_counts if failure_counts else None,
        rng=rng,
        np_rng=np_rng,
        library_source_label=library_source_label,
        library_records=library_records,
        library_cursor=library_cursor,
        events_path=events_path,
        library_build_rows=library_build_rows,
        library_member_rows=library_member_rows,
    )

    def _build_next_library() -> LibraryContext:
        nonlocal libraries_built, libraries_used
        context = builder.build_next(library_index_start=libraries_built)
        libraries_used += 1
        if library_source_label == "artifact":
            libraries_built = libraries_used
        else:
            libraries_built = int(context.sampling_info.get("library_index", libraries_built))
        return context

    diagnostics = SamplingDiagnostics(
        usage_counts=usage_counts,
        tf_usage_counts=tf_usage_counts,
        failure_counts=failure_counts,
        source_label=source_label,
        plan_name=plan_name,
        display_tf_label=_display_tf_label,
        progress_style=progress_style,
        show_tfbs=show_tfbs,
        track_failures=track_failures,
        logger=log,
        library_tfs=[],
        library_tfbs=[],
        library_site_ids=[],
    )

    solver_min_counts: dict[str, int] | None = None

    def _make_generator(
        _library_for_opt: List[str],
        _regulator_labels: List[str],
        *,
        required_regulators_local: list[str],
        min_required_regulators_local: int | None,
    ):
        nonlocal solver_min_counts
        regulator_by_index = list(_regulator_labels) if _regulator_labels else None
        base_min_counts = _min_count_by_regulator(regulator_by_index, min_count_per_tf)
        solver_min_counts = _merge_min_counts(base_min_counts, plan_min_count_by_regulator)
        fe_dict = fixed_elements.model_dump() if hasattr(fixed_elements, "model_dump") else fixed_elements
        solver_required_regs = required_regulators_local or None
        run = deps.optimizer.build(
            library=_library_for_opt,
            sequence_length=seq_len,
            solver=chosen_solver,
            strategy=solver_strategy,
            fixed_elements=fe_dict,
            strands=solver_strands,
            regulator_by_index=regulator_by_index,
            required_regulators=solver_required_regs,
            min_count_by_regulator=solver_min_counts,
            min_required_regulators=min_required_regulators_local,
            solver_time_limit_seconds=solver_time_limit_seconds,
            solver_threads=solver_threads,
        )
        return run

    def _run_library(
        library_context: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
        nonlocal cr_sum
        nonlocal cr_count
        nonlocal duplicate_records
        nonlocal duplicate_solutions
        nonlocal failed_min_count_by_regulator
        nonlocal failed_min_count_per_tf
        nonlocal failed_required_regulators
        nonlocal failed_solutions
        nonlocal last_screen_refresh
        nonlocal latest_failure_totals
        library_for_opt = list(library_context.library_for_opt)
        tfbs_parts = list(library_context.tfbs_parts)
        regulator_labels = list(library_context.regulator_labels)
        sampling_info = dict(library_context.sampling_info)
        sampling_library_index = int(library_context.sampling_library_index)
        sampling_library_hash = str(library_context.sampling_library_hash)
        library_tfbs = list(library_context.library_tfbs)
        library_tfs = list(library_context.library_tfs)
        library_site_ids = list(library_context.library_site_ids)
        library_sources = list(library_context.library_sources)
        required_regulators = list(library_context.required_regulators)
        min_required_regulators = None
        tf_list_from_library = sorted(set(regulator_labels)) if regulator_labels else []
        site_id_by_index = sampling_info.get("site_id_by_index")
        source_by_index = sampling_info.get("source_by_index")
        tfbs_id_by_index = sampling_info.get("tfbs_id_by_index")
        motif_id_by_index = sampling_info.get("motif_id_by_index")

        diagnostics.update_library(
            library_tfs=library_tfs,
            library_tfbs=library_tfbs,
            library_site_ids=library_site_ids,
        )

        sampling_fraction = _compute_sampling_fraction(
            library_for_opt,
            input_tfbs_count=input_tfbs_count,
            pool_strategy=pool_strategy,
        )
        input_meta["sampling_fraction"] = sampling_fraction
        sampling_fraction_pairs = _compute_sampling_fraction_pairs(
            library_for_opt,
            regulator_labels,
            input_pair_count=input_tf_tfbs_pair_count,
            pool_strategy=pool_strategy,
        )
        input_meta["sampling_fraction_pairs"] = sampling_fraction_pairs
        tf_summary = _summarize_tf_counts(
            [_display_tf_label(tf) for tf in regulator_labels] if regulator_labels else []
        )
        library_index = sampling_info.get("library_index")
        strategy_label = sampling_info.get("library_sampling_strategy", library_sampling_strategy)
        pool_label = sampling_info.get("pool_strategy")
        achieved_len = sampling_info.get("achieved_length")
        header = f"Stage-B library for {source_label}/{plan_name}"
        if library_index is not None:
            header = f"{header} (build {library_index})"
        if progress_style != "screen":
            if tf_summary:
                log.info(
                    "%s: %d motifs | TF counts: %s | library_bp=%s pool=%s stage_b_sampling=%s",
                    header,
                    len(library_for_opt),
                    tf_summary,
                    achieved_len,
                    pool_label,
                    strategy_label,
                )
            else:
                log.info(
                    "%s: %d motifs | library_bp=%s pool=%s stage_b_sampling=%s",
                    header,
                    len(library_for_opt),
                    achieved_len,
                    pool_label,
                    strategy_label,
                )

        run = _make_generator(
            library_for_opt,
            regulator_labels,
            required_regulators_local=required_regulators,
            min_required_regulators_local=min_required_regulators,
        )
        opt = run.optimizer
        generator = run.generator
        forbid_each = run.forbid_each

        local_generated = 0
        produced_this_library = 0
        stall_triggered = False
        last_progress = time.monotonic()

        while local_generated < max_per_subsample and global_generated < quota:
            fingerprints = set()
            consecutive_dup = 0
            subsample_started = time.monotonic()
            last_log_warn = subsample_started
            last_progress = subsample_started
            produced_this_library = 0
            stall_triggered = False

            def _mark_stall(now: float) -> None:
                nonlocal stall_events, stall_triggered
                if stall_triggered:
                    return
                log.info(
                    "[%s/%s] Stall (> %ds) with no solutions; will resample.",
                    source_label,
                    plan_name,
                    stall_seconds,
                )
                stall_events += 1
                if events_path is not None:
                    try:
                        _emit_event(
                            events_path,
                            event="STALL_DETECTED",
                            payload={
                                "input_name": source_label,
                                "plan_name": plan_name,
                                "stall_seconds": float(now - last_progress),
                                "library_index": int(sampling_library_index),
                                "library_hash": str(sampling_library_hash),
                            },
                        )
                    except Exception:
                        log.debug("Failed to emit STALL_DETECTED event.", exc_info=True)
                stall_triggered = True

            for sol in generator:
                now = time.monotonic()
                if policy.should_trigger_stall(
                    now=now,
                    last_progress=last_progress,
                ):
                    _mark_stall(now)
                    break
                if policy.should_warn_stall(
                    now=now,
                    last_warn=last_log_warn,
                    last_progress=last_progress,
                ):
                    log.info(
                        "[%s/%s] Still working... %.1fs on current library.",
                        source_label,
                        plan_name,
                        now - subsample_started,
                    )
                    last_log_warn = now
                last_progress = now

                if forbid_each:
                    opt.forbid(sol)
                seq = sol.sequence
                if seq in fingerprints:
                    duplicate_solutions += 1
                    consecutive_dup += 1
                    if consecutive_dup >= max_dupes:
                        log.info(
                            "[%s/%s] Duplicate guard (>= %d in a row); will resample.",
                            source_label,
                            plan_name,
                            max_dupes,
                        )
                        break
                    continue
                consecutive_dup = 0
                fingerprints.add(seq)

                used_tfbs, used_tfbs_detail, used_tf_counts, used_tf_list = _compute_used_tf_info(
                    sol,
                    library_for_opt,
                    regulator_labels,
                    fixed_elements,
                    site_id_by_index,
                    source_by_index,
                    tfbs_id_by_index,
                    motif_id_by_index,
                )
                solver_status = getattr(sol, "status", None)
                if solver_status is not None:
                    solver_status = str(solver_status)
                solver_objective = getattr(sol, "objective", None)
                if solver_objective is None:
                    solver_objective = getattr(sol, "objective_value", None)
                try:
                    solver_objective = float(solver_objective) if solver_objective is not None else None
                except (TypeError, ValueError):
                    solver_objective = None
                solver_solve_time_s = getattr(sol, "_densegen_solve_time_s", None)

                covers_all = True
                covers_required = True
                if min_count_per_tf > 0 and tf_list_from_library:
                    missing = [tf for tf in tf_list_from_library if used_tf_counts.get(tf, 0) < min_count_per_tf]
                    if missing:
                        covers_all = False
                        failed_solutions += 1
                        failed_min_count_per_tf += 1
                        diagnostics.record_site_failures("min_count_per_tf")
                        attempt_index = _next_attempt_index()
                        _log_rejection(
                            tables_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            attempt_index=attempt_index,
                            reason="min_count_per_tf",
                            detail={
                                "min_count_per_tf": min_count_per_tf,
                                "missing_tfs": missing,
                            },
                            sequence=seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                if required_regulators:
                    missing = [tf for tf in required_regulators if used_tf_counts.get(tf, 0) < 1]
                    if missing:
                        covers_required = False
                        failed_solutions += 1
                        failed_required_regulators += 1
                        diagnostics.record_site_failures("required_regulators")
                        attempt_index = _next_attempt_index()
                        _log_rejection(
                            tables_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            attempt_index=attempt_index,
                            reason="required_regulators",
                            detail={
                                "required_regulators": required_regulators,
                                "missing_tfs": missing,
                            },
                            sequence=seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                if plan_min_count_by_regulator:
                    missing = [
                        tf
                        for tf, min_count in plan_min_count_by_regulator.items()
                        if used_tf_counts.get(tf, 0) < int(min_count)
                    ]
                    if missing:
                        failed_solutions += 1
                        failed_min_count_by_regulator += 1
                        diagnostics.record_site_failures("min_count_by_regulator")
                        attempt_index = _next_attempt_index()
                        _log_rejection(
                            tables_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            attempt_index=attempt_index,
                            reason="min_count_by_regulator",
                            detail={
                                "min_count_by_regulator": [
                                    {
                                        "tf": tf,
                                        "min_count": int(plan_min_count_by_regulator[tf]),
                                        "found": int(used_tf_counts.get(tf, 0)),
                                    }
                                    for tf in missing
                                ]
                            },
                            sequence=seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                pad_meta = {"used": False}
                final_seq = seq
                if not pad_enabled and len(final_seq) < seq_len:
                    raise RuntimeError(f"[{source_label}/{plan_name}] Sequence shorter than target and pad.mode=off.")
                if pad_enabled and len(final_seq) < seq_len:
                    gap = seq_len - len(final_seq)
                    rf = deps.pad(
                        length=gap,
                        mode=pad_mode,
                        gc_mode=pad_gc_mode,
                        gc_min=pad_gc_min,
                        gc_max=pad_gc_max,
                        gc_target=pad_gc_target,
                        gc_tolerance=pad_gc_tolerance,
                        gc_min_pad_length=pad_gc_min_length,
                        max_tries=pad_max_tries,
                        rng=rng,
                    )
                    if isinstance(rf, tuple) and len(rf) == 2:
                        pad, pad_info = rf
                        pad_info = pad_info or {}
                    else:
                        pad, pad_info = rf, {}
                    final_seq = (pad + final_seq) if pad_end == "5prime" else (final_seq + pad)
                    pad_meta = {
                        "used": True,
                        "bases": gap,
                        "end": pad_end,
                        "gc_mode": pad_info.get("gc_mode", pad_gc_mode),
                        "gc_min": pad_info.get("final_gc_min"),
                        "gc_max": pad_info.get("final_gc_max"),
                        "gc_target_min": pad_info.get("target_gc_min"),
                        "gc_target_max": pad_info.get("target_gc_max"),
                        "gc_actual": pad_info.get("gc_actual"),
                        "relaxed": pad_info.get("relaxed"),
                        "relaxed_reason": pad_info.get("relaxed_reason"),
                        "attempts": pad_info.get("attempts"),
                    }

                if forbid_kmers:
                    allowed_windows = _promoter_windows(final_seq, fixed_elements_dump)
                    if not allowed_windows:
                        raise RuntimeError(
                            f"[{source_label}/{plan_name}] postprocess validation requires promoter constraints."
                        )
                    hit = _find_forbidden_kmer(final_seq, forbid_kmers, allowed_windows)
                    if hit is not None:
                        failed_solutions += 1
                        attempt_index = _next_attempt_index()
                        _log_rejection(
                            tables_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            attempt_index=attempt_index,
                            reason="postprocess_forbidden_kmer",
                            detail={"kmer": hit[0], "position": int(hit[1])},
                            sequence=final_seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                used_tfbs_detail = _apply_pad_offsets(used_tfbs_detail, pad_meta)
                gc_core = gc_fraction(seq)
                gc_total = gc_fraction(final_seq)
                created_at = datetime.now(timezone.utc).isoformat()
                derived = build_metadata(
                    sol=sol,
                    plan_name=plan_name,
                    tfbs_parts=tfbs_parts,
                    regulator_labels=regulator_labels,
                    library_for_opt=library_for_opt,
                    fixed_elements=fixed_elements,
                    chosen_solver=chosen_solver,
                    solver_strategy=solver_strategy,
                    solver_time_limit_seconds=solver_time_limit_seconds,
                    solver_threads=solver_threads,
                    solver_strands=solver_strands,
                    seq_len=seq_len,
                    actual_length=len(final_seq),
                    pad_meta=pad_meta,
                    sampling_meta=sampling_info,
                    schema_version=str(global_cfg.schema_version),
                    created_at=created_at,
                    run_id=run_id,
                    run_root=run_root,
                    run_config_path=run_config_path,
                    run_config_sha256=run_config_sha256,
                    random_seed=random_seed,
                    policy_pad=policy_pad,
                    policy_sampling=policy_sampling,
                    policy_solver=policy_solver,
                    input_meta=input_meta,
                    fixed_elements_dump=fixed_elements_dump,
                    used_tfbs=used_tfbs,
                    used_tfbs_detail=used_tfbs_detail,
                    used_tf_counts=used_tf_counts,
                    used_tf_list=used_tf_list,
                    min_count_per_tf=min_count_per_tf,
                    covers_all_tfs_in_solution=bool(covers_all),
                    required_regulators=required_regulators,
                    min_required_regulators=min_required_regulators,
                    min_count_by_regulator=plan_min_count_by_regulator,
                    covers_required_regulators=bool(covers_required),
                    gc_core=gc_core,
                    gc_total=gc_total,
                    input_row_count=input_row_count,
                    input_tf_count=input_tf_count,
                    input_tfbs_count=input_tfbs_count,
                    input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
                    sampling_fraction=sampling_fraction,
                    sampling_fraction_pairs=sampling_fraction_pairs,
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                    solver_status=solver_status,
                    solver_objective=solver_objective,
                    solver_solve_time_s=solver_solve_time_s,
                    dense_arrays_version=dense_arrays_version,
                    dense_arrays_version_source=dense_arrays_version_source,
                )
                if not derived:
                    log.info("[%s/%s] Skipping solution; no metadata found", source_label, plan_name)
                    continue
                record = OutputRecord.from_sequence(
                    sequence=final_seq,
                    meta=derived,
                    source=source_label,
                    bio_type=output_bio_type,
                    alphabet=output_alphabet,
                )

                if not _write_to_sinks(sinks, record):
                    duplicate_records += 1
                    continue

                global_generated += 1
                local_generated += 1
                produced_this_library += 1
                attempts_buffer.append(
                    {
                        "created_at": derived.get("created_at"),
                        "input_name": source_label,
                        "plan_name": plan_name,
                        "attempt_index": _next_attempt_index(),
                        "status": "ok",
                        "reason": None,
                        "detail": None,
                        "sequence": final_seq,
                        "used_tf_counts": used_tf_counts,
                        "used_tf_list": used_tf_list,
                        "sampling_library_index": int(sampling_library_index),
                        "sampling_library_hash": str(sampling_library_hash),
                        "solver_status": solver_status,
                        "solver_objective": solver_objective,
                        "solver_solve_time_s": solver_solve_time_s,
                        "dense_arrays_version": dense_arrays_version,
                        "dense_arrays_version_source": dense_arrays_version_source,
                        "library_tfbs": library_tfbs,
                        "library_tfs": library_tfs,
                        "library_site_ids": library_site_ids,
                        "library_sources": library_sources,
                    }
                )
                if solution_rows is not None:
                    solution_rows.append(
                        _append_attempt(
                            tables_root,
                            run_id=run_id,
                            input_name=source_label,
                            plan_name=plan_name,
                            attempt_index=_next_attempt_index(),
                            status="ok",
                            reason=None,
                            detail=None,
                            sequence=final_seq,
                            used_tf_counts=used_tf_counts,
                            used_tf_list=used_tf_list,
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                            solver_status=solver_status,
                            solver_objective=solver_objective,
                            solver_solve_time_s=solver_solve_time_s,
                            dense_arrays_version=dense_arrays_version,
                            dense_arrays_version_source=dense_arrays_version_source,
                            library_tfbs=library_tfbs,
                            library_tfs=library_tfs,
                            library_site_ids=library_site_ids,
                            library_sources=library_sources,
                            attempts_buffer=attempts_buffer,
                        )
                    )
                if composition_rows is not None:
                    composition_rows.append(
                        {
                            "input_name": source_label,
                            "plan_name": plan_name,
                            "library_index": int(sampling_library_index),
                            "library_hash": str(sampling_library_hash),
                            "sequence": final_seq,
                            "used_tfbs": used_tfbs,
                            "used_tfbs_detail": used_tfbs_detail,
                        }
                    )
                if events_path is not None:
                    try:
                        _emit_event(
                            events_path,
                            event="SOLUTION_ACCEPTED",
                            payload={
                                "input_name": source_label,
                                "plan_name": plan_name,
                                "sequence": final_seq,
                                "library_index": int(sampling_library_index),
                                "library_hash": str(sampling_library_hash),
                            },
                        )
                    except Exception:
                        log.debug("Failed to emit SOLUTION_ACCEPTED event.", exc_info=True)

                if checkpoint_every > 0 and global_generated % max(1, checkpoint_every) == 0:
                    _flush_attempts(tables_root, attempts_buffer)
                    if solution_rows is not None:
                        _flush_solutions(tables_root, solution_rows)
                    if state_counts is not None:
                        state_counts[(source_label, plan_name)] = int(global_generated)
                        if write_state is not None:
                            write_state()

                if progress_style == "screen" and dashboard is not None:
                    if (time.monotonic() - last_screen_refresh) >= progress_refresh_seconds:
                        last_screen_refresh = time.monotonic()
                        bar = _format_progress_bar(global_generated, quota)
                        cr_now = float(sol.compression_ratio)
                        cr_sum += cr_now
                        cr_count += 1
                        cr_avg = cr_sum / max(cr_count, 1)
                        diversity_label = _summarize_diversity(
                            usage_counts,
                            tf_usage_counts,
                            library_tfs=library_tfs,
                            library_tfbs=library_tfbs,
                        )
                        renderable = _build_screen_dashboard(
                            source_label=source_label,
                            plan_name=plan_name,
                            bar=bar,
                            generated=int(global_generated),
                            quota=int(quota),
                            pct=float(global_generated) / float(quota) * 100.0,
                            local_generated=int(local_generated),
                            local_target=int(max_per_subsample),
                            library_index=int(sampling_library_index),
                            cr_now=cr_now,
                            cr_avg=cr_avg,
                            resamples=int(counters.total_resamples),
                            dup_out=int(duplicate_records),
                            dup_sol=int(duplicate_solutions),
                            fails=int(failed_solutions),
                            stalls=int(stall_events),
                            failure_totals=latest_failure_totals,
                            tf_usage=diagnostics.map_tf_usage(tf_usage_counts),
                            tfbs_usage=diagnostics.map_tfbs_usage(usage_counts),
                            diversity_label=diversity_label,
                            show_tfbs=show_tfbs,
                            show_solutions=bool(print_visual),
                            sequence_preview=final_seq if print_visual else None,
                        )
                        dashboard.update(renderable)

                if progress_style == "stream" and global_generated % max(1, progress_every) == 0:
                    bar = _format_progress_bar(global_generated, quota)
                    pct = float(global_generated) / float(quota) * 100.0
                    if show_tfbs:
                        tf_label = _short_seq(str(used_tfbs_detail), max_len=80)
                    else:
                        tf_label = _summarize_tf_counts(used_tf_list)
                    cr_now = float(sol.compression_ratio)
                    cr_sum += cr_now
                    cr_count += 1
                    if print_visual:
                        log.info(
                            "[%s/%s] %s %d/%d (%.2f%%) (local %d/%d) CR=%.3f | TFBS %s | seq %s",
                            source_label,
                            plan_name,
                            bar,
                            global_generated,
                            quota,
                            pct,
                            local_generated,
                            max_per_subsample,
                            cr_now if cr_now is not None else float("nan"),
                            tf_label,
                            final_seq,
                        )
                    elif show_solutions:
                        log.info(
                            "[%s/%s] %s %d/%d (%.2f%%) (local %d/%d) CR=%.3f | TFBS %s",
                            source_label,
                            plan_name,
                            bar,
                            global_generated,
                            quota,
                            pct,
                            local_generated,
                            max_per_subsample,
                            cr_now if cr_now is not None else float("nan"),
                            tf_label,
                        )
                    else:
                        log.info(
                            "[%s/%s] %s %d/%d (%.2f%%) (local %d/%d) CR=%.3f",
                            source_label,
                            plan_name,
                            bar,
                            global_generated,
                            quota,
                            pct,
                            local_generated,
                            max_per_subsample,
                            cr_now if cr_now is not None else float("nan"),
                        )

                if leaderboard_every > 0 and global_generated % max(1, leaderboard_every) == 0:
                    failure_totals = _summarize_failure_totals(
                        failure_counts,
                        input_name=source_label,
                        plan_name=plan_name,
                    )
                    latest_failure_totals = failure_totals
                    if progress_style != "screen":
                        log.info(
                            "[%s/%s] Progress %s %d/%d (%.2f%%) | resamples=%d dup_out=%d "
                            "dup_sol=%d fails=%d stalls=%d | %s",
                            source_label,
                            plan_name,
                            bar,
                            global_generated,
                            quota,
                            pct,
                            counters.total_resamples,
                            duplicate_records,
                            duplicate_solutions,
                            failed_solutions,
                            stall_events,
                            failure_totals,
                        )
                        tf_usage_display = diagnostics.map_tf_usage(tf_usage_counts)
                        tfbs_usage_display = diagnostics.map_tfbs_usage(usage_counts) if show_tfbs else usage_counts
                        log.info(
                            "[%s/%s] Leaderboard (TF): %s",
                            source_label,
                            plan_name,
                            _summarize_leaderboard(tf_usage_display, top=5),
                        )
                        if show_tfbs:
                            log.info(
                                "[%s/%s] Leaderboard (TFBS): %s",
                                source_label,
                                plan_name,
                                _summarize_leaderboard(tfbs_usage_display, top=5),
                            )
                        else:
                            log.info(
                                "[%s/%s] TFBS usage: %s",
                                source_label,
                                plan_name,
                                _summarize_tfbs_usage_stats(usage_counts),
                            )
                        if show_tfbs:
                            log.info(
                                "[%s/%s] Failed TFBS: %s",
                                source_label,
                                plan_name,
                                _summarize_failure_leaderboard(
                                    failure_counts,
                                    input_name=source_label,
                                    plan_name=plan_name,
                                    top=5,
                                ),
                            )
                        elif failure_totals:
                            log.info("[%s/%s] Failures: %s", source_label, plan_name, failure_totals)
                        log.info(
                            "[%s/%s] Diversity: %s",
                            source_label,
                            plan_name,
                            _summarize_diversity(
                                usage_counts,
                                tf_usage_counts,
                                library_tfs=library_tfs,
                                library_tfbs=library_tfbs,
                            ),
                        )
                    if show_solutions:
                        log.info(
                            "[%s/%s] Example: %s",
                            source_label,
                            plan_name,
                            final_seq,
                        )

                if local_generated >= max_per_subsample or global_generated >= quota:
                    break

        if produced_this_library == 0 and not stall_triggered and stall_seconds > 0:
            now = time.monotonic()
            if (now - last_progress) >= stall_seconds:
                _mark_stall(now)

        return LibraryRunResult(
            produced=produced_this_library,
            stall_triggered=stall_triggered,
            global_generated=int(global_generated),
        )

    def _on_no_solution(library_context: LibraryContext, reason: str) -> None:
        attempt_index = _next_attempt_index()
        detail = {"stall_seconds": stall_seconds} if reason == "stall_no_solution" else {}
        _append_attempt(
            tables_root,
            run_id=run_id,
            input_name=source_label,
            plan_name=plan_name,
            attempt_index=attempt_index,
            status="failed",
            reason=reason,
            detail=detail,
            sequence=None,
            used_tf_counts=None,
            used_tf_list=[],
            sampling_library_index=int(library_context.sampling_library_index),
            sampling_library_hash=str(library_context.sampling_library_hash),
            solver_status=None,
            solver_objective=None,
            solver_solve_time_s=None,
            dense_arrays_version=dense_arrays_version,
            dense_arrays_version_source=dense_arrays_version_source,
            library_tfbs=list(library_context.library_tfbs),
            library_tfs=list(library_context.library_tfs),
            library_site_ids=list(library_context.library_site_ids),
            library_sources=list(library_context.library_sources),
            attempts_buffer=attempts_buffer,
        )

    def _on_resample(library_context: LibraryContext, reason: str, produced_this_library: int) -> None:
        if events_path is None:
            return
        try:
            _emit_event(
                events_path,
                event="RESAMPLE_TRIGGERED",
                payload={
                    "input_name": source_label,
                    "plan_name": plan_name,
                    "reason": reason,
                    "produced_this_library": int(produced_this_library),
                    "library_index": int(library_context.sampling_library_index),
                    "library_hash": str(library_context.sampling_library_hash),
                },
            )
        except Exception:
            log.debug("Failed to emit RESAMPLE_TRIGGERED event.", exc_info=True)

    sampler = StageBSampler(
        source_label=source_label,
        plan_name=plan_name,
        quota=quota,
        policy=policy,
        max_per_subsample=max_per_subsample,
        pool_strategy=pool_strategy,
        iterative_min_new_solutions=iterative_min_new_solutions,
        iterative_max_libraries=iterative_max_libraries,
        counters=counters,
    )
    sampling_result = sampler.run(
        build_next_library=_build_next_library,
        run_library=_run_library,
        on_no_solution=_on_no_solution,
        on_resample=_on_resample,
        already_generated=already_generated,
        one_subsample_only=one_subsample_only,
        plan_start=plan_start,
    )
    produced_total_this_call = sampling_result.generated
    global_generated = int(already_generated) + int(produced_total_this_call)
    for sink in sinks:
        sink.flush()

        if one_subsample_only:
            _flush_attempts(tables_root, attempts_buffer)
            if solution_rows is not None:
                _flush_solutions(tables_root, solution_rows)
            if state_counts is not None:
                state_counts[(source_label, plan_name)] = int(global_generated)
                if write_state is not None:
                    write_state()
            snapshot = diagnostics.snapshot()
            if global_generated >= quota and (usage_counts or tf_usage_counts or failure_counts):
                diagnostics.log_snapshot()
            if dashboard is not None:
                dashboard.close()
            return produced_total_this_call, {
                "generated": produced_total_this_call,
                "duplicates_skipped": duplicate_records,
                "failed_solutions": failed_solutions,
                "total_resamples": counters.total_resamples,
                "libraries_built": max(0, libraries_built - libraries_built_start),
                "stall_events": stall_events,
                "failed_min_count_per_tf": failed_min_count_per_tf,
                "failed_required_regulators": failed_required_regulators,
                "failed_min_count_by_regulator": failed_min_count_by_regulator,
                "failed_min_required_regulators": failed_min_required_regulators,
                "duplicate_solutions": duplicate_solutions,
                "leaderboard_latest": snapshot,
            }

    _flush_attempts(tables_root, attempts_buffer)
    if solution_rows is not None:
        _flush_solutions(tables_root, solution_rows)
    log.info("Completed %s/%s: %d/%d", source_label, plan_name, global_generated, quota)
    if state_counts is not None:
        state_counts[(source_label, plan_name)] = int(global_generated)
        if write_state is not None:
            write_state()
    snapshot = diagnostics.snapshot()
    if usage_counts or tf_usage_counts or failure_counts:
        diagnostics.log_snapshot()
    if dashboard is not None:
        dashboard.close()
    return produced_total_this_call, {
        "generated": produced_total_this_call,
        "duplicates_skipped": duplicate_records,
        "failed_solutions": failed_solutions,
        "total_resamples": counters.total_resamples,
        "libraries_built": max(0, libraries_built - libraries_built_start),
        "stall_events": stall_events,
        "failed_min_count_per_tf": failed_min_count_per_tf,
        "failed_required_regulators": failed_required_regulators,
        "failed_min_count_by_regulator": failed_min_count_by_regulator,
        "failed_min_required_regulators": failed_min_required_regulators,
        "duplicate_solutions": duplicate_solutions,
        "leaderboard_latest": snapshot,
    }


def run_pipeline(
    loaded: LoadedConfig,
    *,
    resume: bool,
    build_stage_a: bool = False,
    show_tfbs: bool = False,
    show_solutions: bool = False,
    deps: PipelineDeps | None = None,
) -> RunSummary:
    deps = deps or default_deps()
    cfg = loaded.root.densegen
    install_native_stderr_filters(suppress_solver_messages=bool(cfg.logging.suppress_solver_stderr))
    run_root = resolve_run_root(loaded.path, cfg.run.root)
    run_root_str = str(run_root)
    config_sha = hashlib.sha256(loaded.path.read_bytes()).hexdigest()
    try:
        run_cfg_path = str(loaded.path.relative_to(run_root))
    except ValueError:
        run_cfg_path = str(loaded.path)

    outputs_root = run_outputs_root(run_root)
    tables_root = run_tables_root(run_root)
    existing_outputs = has_existing_run_outputs(run_root)
    if resume:
        if not existing_outputs:
            outputs_label = display_path(outputs_root, run_root, absolute=False)
            raise RuntimeError(
                f"resume=True requested but no outputs were found under {outputs_label}. "
                "Start a fresh run or remove resume=True."
            )
    else:
        if existing_outputs:
            outputs_label = display_path(outputs_root, run_root, absolute=False)
            raise RuntimeError(
                f"Existing outputs found under {outputs_label}. Explicit resume is required to continue this run."
            )

    # Seed
    seed = int(cfg.runtime.random_seed)
    seeds = derive_seed_map(seed, ["stage_a", "stage_b", "solver"])
    rng = random.Random(seeds["stage_b"])
    np_rng_stage_a = np.random.default_rng(seeds["stage_a"])
    np_rng_stage_b = np.random.default_rng(seeds["stage_b"])

    # Plan & solver
    pl = cfg.generation.resolve_plan()
    chosen_solver = select_solver(
        cfg.solver.backend,
        deps.optimizer,
        strategy=str(cfg.solver.strategy),
    )
    solver_time_limit_seconds = (
        float(cfg.solver.time_limit_seconds) if cfg.solver.time_limit_seconds is not None else None
    )
    solver_threads = int(cfg.solver.threads) if cfg.solver.threads is not None else None
    dense_arrays_version, dense_arrays_version_source = _resolve_dense_arrays_version(loaded.path)

    # Build sinks
    sinks = list(deps.sink_factory(cfg, loaded.path))
    _assert_sink_alignment(sinks)
    output_bio_type, output_alphabet = resolve_bio_alphabet(cfg)

    total = 0
    per_plan: dict[tuple[str, str], int] = {}
    plan_stats: dict[tuple[str, str], dict[str, int]] = {}
    plan_order: list[tuple[str, str]] = []
    plan_leaderboards: dict[tuple[str, str], dict] = {}
    inputs_manifest_entries: dict[str, dict] = {}
    source_cache: dict[str, PoolData] = {}
    library_build_rows: list[dict] = []
    library_member_rows: list[dict] = []
    solution_rows: list[dict] = []
    composition_rows: list[dict] = []
    outputs_root.mkdir(parents=True, exist_ok=True)
    candidates_dir = candidates_root(outputs_root, cfg.run.id)
    candidate_logging = _candidate_logging_enabled(cfg)
    events_path = outputs_root / "meta" / "events.jsonl"
    try:
        _write_effective_config(
            cfg=cfg, cfg_path=loaded.path, run_root=run_root, seeds=seeds, outputs_root=outputs_root
        )
    except Exception:
        log.debug("Failed to write effective_config.json.", exc_info=True)
    stage_a_state = prepare_stage_a_pools(
        cfg=cfg,
        cfg_path=loaded.path,
        run_root=run_root,
        outputs_root=outputs_root,
        rng=np_rng_stage_a,
        build_stage_a=build_stage_a,
        candidate_logging=candidate_logging,
        candidates_dir=candidates_dir,
        plan_items=pl,
        events_path=events_path,
        run_id=str(cfg.run.id),
        deps=deps,
    )
    pool_data = stage_a_state.pool_data
    plan_pools = stage_a_state.plan_pools
    plan_pool_sources = stage_a_state.plan_pool_sources
    source_cache.update(stage_a_state.source_cache)

    if resume and pool_data is None:
        raise RuntimeError(
            "resume=True requires existing Stage-A pools. Run dense stage-a build-pool first or rerun without resume."
        )
    sampling_cfg = cfg.generation.sampling
    library_state = prepare_library_source(
        sampling_cfg=sampling_cfg,
        cfg_path=loaded.path,
        run_root=run_root,
        plan_items=pl,
        plan_pools=plan_pools,
        tables_root=tables_root,
    )
    library_source = library_state.source
    library_artifact = library_state.artifact
    library_records = library_state.records
    library_cursor = library_state.cursor
    ensure_run_meta_dir(run_root)
    state_ctx = init_run_state(
        run_root=run_root,
        run_id=str(cfg.run.id),
        schema_version=str(cfg.schema_version),
        config_sha256=config_sha,
    )

    resume_state = load_resume_state(
        resume=resume,
        loaded=loaded,
        tables_root=tables_root,
        config_sha=config_sha,
    )
    existing_counts = resume_state.existing_counts
    existing_usage_by_plan = resume_state.existing_usage_by_plan
    site_failure_counts = resume_state.site_failure_counts
    attempt_counters = resume_state.attempt_counters
    if existing_counts:
        total = sum(existing_counts.values())
        per_plan = dict(existing_counts)
        log.info(
            "Resuming from existing outputs: %d sequences across %d plan(s).",
            total,
            len(existing_counts),
        )
    assert_state_matches_outputs(state_path=state_ctx.path, existing_counts=existing_counts)

    plan_stats, plan_order = init_plan_stats(
        plan_items=pl,
        plan_pools=plan_pools,
        existing_counts=existing_counts,
    )

    def _accumulate_stats(key: tuple[str, str], stats: dict) -> None:
        if key not in plan_stats:
            plan_stats[key] = {
                "generated": 0,
                "duplicates_skipped": 0,
                "failed_solutions": 0,
                "total_resamples": 0,
                "libraries_built": 0,
                "stall_events": 0,
                "failed_min_count_per_tf": 0,
                "failed_required_regulators": 0,
                "failed_min_count_by_regulator": 0,
                "failed_min_required_regulators": 0,
                "duplicate_solutions": 0,
            }
            plan_order.append(key)
        dest = plan_stats[key]
        for field in dest:
            dest[field] += int(stats.get(field, 0))

    # Round-robin scheduler
    round_robin = bool(cfg.runtime.round_robin)
    if round_robin and str(cfg.generation.sampling.pool_strategy) == "iterative_subsample":
        log.warning(
            "round_robin=true with pool_strategy=iterative_subsample will rebuild libraries more frequently; "
            "expect higher runtime for multi-plan runs."
        )
    display_map_by_input = build_display_map_by_input(
        plan_items=pl,
        plan_pools=plan_pools,
        inputs=cfg.inputs,
        cfg_path=loaded.path,
    )
    checkpoint_every = int(cfg.runtime.checkpoint_every)
    state_counts = init_state_counts(
        plan_items=pl,
        plan_pools=plan_pools,
        existing_counts=existing_counts,
    )

    def _write_state() -> None:
        write_run_state(
            path=state_ctx.path,
            run_id=str(cfg.run.id),
            schema_version=str(cfg.schema_version),
            config_sha256=config_sha,
            run_root=str(run_root),
            counts=state_counts,
            created_at=state_ctx.created_at,
        )

    _write_state()

    process_plan_args = {
        "global_cfg": cfg,
        "sinks": sinks,
        "chosen_solver": chosen_solver,
        "deps": deps,
        "rng": rng,
        "np_rng": np_rng_stage_b,
        "cfg_path": loaded.path,
        "run_id": cfg.run.id,
        "run_root": run_root_str,
        "run_config_path": run_cfg_path,
        "run_config_sha256": config_sha,
        "random_seed": seed,
        "dense_arrays_version": dense_arrays_version,
        "dense_arrays_version_source": dense_arrays_version_source,
        "show_tfbs": show_tfbs,
        "show_solutions": show_solutions,
        "output_bio_type": output_bio_type,
        "output_alphabet": output_alphabet,
    }
    plan_execution = run_plan_schedule(
        plan_items=pl,
        plan_pools=plan_pools,
        plan_pool_sources=plan_pool_sources,
        existing_counts=existing_counts,
        round_robin=round_robin,
        process_plan=_process_plan_for_source,
        process_plan_args=process_plan_args,
        accumulate_stats=_accumulate_stats,
        plan_pool_input_meta=_plan_pool_input_meta,
        inputs_manifest_entries=inputs_manifest_entries,
        existing_usage_by_plan=existing_usage_by_plan,
        state_counts=state_counts,
        checkpoint_every=checkpoint_every,
        write_state=_write_state,
        site_failure_counts=site_failure_counts,
        source_cache=source_cache,
        attempt_counters=attempt_counters,
        library_records=library_records,
        library_cursor=library_cursor,
        library_source=library_source,
        library_build_rows=library_build_rows,
        library_member_rows=library_member_rows,
        solution_rows=solution_rows,
        composition_rows=composition_rows,
        events_path=events_path,
        display_map_by_input=display_map_by_input,
    )
    per_plan = plan_execution.per_plan
    total = plan_execution.total
    plan_leaderboards = plan_execution.plan_leaderboards

    for sink in sinks:
        sink.finalize()

    outputs_root = run_outputs_root(run_root)
    tables_root = run_tables_root(run_root)
    _consolidate_parts(tables_root, part_glob="attempts_part-*.parquet", final_name="attempts.parquet")
    _consolidate_parts(tables_root, part_glob="solutions_part-*.parquet", final_name="solutions.parquet")

    pool_manifest_hash = None
    pool_manifest_path = outputs_root / "pools" / "pool_manifest.json"
    if pool_manifest_path.exists():
        pool_manifest_hash = _hash_file(pool_manifest_path)
    elif library_artifact is not None and library_artifact.pool_manifest_hash:
        pool_manifest_hash = library_artifact.pool_manifest_hash

    write_library_artifacts(
        library_source=library_source,
        library_artifact=library_artifact,
        library_build_rows=library_build_rows,
        library_member_rows=library_member_rows,
        outputs_root=outputs_root,
        cfg_path=loaded.path,
        run_id=str(cfg.run.id),
        run_root=run_root,
        config_hash=config_sha,
        pool_manifest_hash=pool_manifest_hash,
    )

    if composition_rows:
        composition_path = tables_root / "composition.parquet"
        existing_rows: list[dict] = []
        if composition_path.exists():
            try:
                existing_rows = pd.read_parquet(composition_path).to_dict("records")
            except Exception:
                log.warning("Failed to read existing composition.parquet; overwriting.", exc_info=True)
                existing_rows = []
        existing_keys = {
            (str(row.get("solution_id") or ""), int(row.get("placement_index") or 0)) for row in existing_rows
        }
        new_rows = [
            row
            for row in composition_rows
            if (str(row.get("solution_id") or ""), int(row.get("placement_index") or 0)) not in existing_keys
        ]
        pd.DataFrame(existing_rows + new_rows).to_parquet(composition_path, index=False)

    try:
        write_run_metrics(cfg=cfg, run_root=run_root)
    except Exception as exc:
        raise RuntimeError(f"Failed to write run_metrics.parquet: {exc}") from exc

    manifest_items = [
        PlanManifest(
            input_name=key[0],
            plan_name=key[1],
            generated=plan_stats[key]["generated"],
            duplicates_skipped=plan_stats[key]["duplicates_skipped"],
            failed_solutions=plan_stats[key]["failed_solutions"],
            total_resamples=plan_stats[key]["total_resamples"],
            libraries_built=plan_stats[key]["libraries_built"],
            stall_events=plan_stats[key]["stall_events"],
            failed_min_count_per_tf=plan_stats[key]["failed_min_count_per_tf"],
            failed_required_regulators=plan_stats[key]["failed_required_regulators"],
            failed_min_count_by_regulator=plan_stats[key]["failed_min_count_by_regulator"],
            failed_min_required_regulators=plan_stats[key]["failed_min_required_regulators"],
            duplicate_solutions=plan_stats[key]["duplicate_solutions"],
            leaderboard_latest=plan_leaderboards.get(key),
        )
        for key in plan_order
    ]
    manifest = RunManifest(
        run_id=cfg.run.id,
        created_at=datetime.now(timezone.utc).isoformat(),
        schema_version=str(cfg.schema_version),
        config_sha256=config_sha,
        run_root=run_root_str,
        random_seed=seed,
        seed_stage_a=seeds.get("stage_a"),
        seed_stage_b=seeds.get("stage_b"),
        seed_solver=seeds.get("solver"),
        solver_backend=chosen_solver,
        solver_strategy=str(cfg.solver.strategy),
        solver_time_limit_seconds=solver_time_limit_seconds,
        solver_threads=solver_threads,
        solver_strands=str(cfg.solver.strands),
        dense_arrays_version=dense_arrays_version,
        dense_arrays_version_source=dense_arrays_version_source,
        items=manifest_items,
    )
    manifest_path = run_manifest_path(run_root)
    manifest.write_json(manifest_path)

    if inputs_manifest_entries:
        manifest_inputs: list[dict] = []
        for item in pl:
            spec = plan_pools[item.name]
            entry = inputs_manifest_entries.get(spec.pool_name)
            if entry is not None:
                manifest_inputs.append(entry)
        payload = {
            "schema_version": str(cfg.schema_version),
            "run_id": cfg.run.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha,
            "inputs": manifest_inputs,
            "library_sampling": cfg.generation.sampling.model_dump(),
        }
        inputs_manifest = inputs_manifest_path(run_root)
        inputs_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
        log.info(
            "Inputs manifest written: %s",
            display_path(inputs_manifest, run_root, absolute=False),
        )

    _write_state()

    return RunSummary(total_generated=total, per_plan=per_plan)
