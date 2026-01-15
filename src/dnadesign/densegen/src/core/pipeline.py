"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/pipeline.py

DenseGen pipeline orchestration (CLI-agnostic).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd

from ..adapters.optimizer import DenseArraysAdapter, OptimizerAdapter
from ..adapters.outputs import OutputRecord, SinkBase, build_sinks, resolve_bio_alphabet
from ..adapters.sources import data_source_factory
from ..adapters.sources.base import BaseDataSource
from ..config import DenseGenConfig, LoadedConfig, ResolvedPlanItem, resolve_relative_path, resolve_run_root
from .metadata import build_metadata
from .postprocess import random_fill
from .sampler import TFSampler

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    total_generated: int
    per_plan: dict[tuple[str, str], int]


@dataclass(frozen=True)
class PipelineDeps:
    source_factory: Callable[[object, Path], BaseDataSource]
    sink_factory: Callable[[DenseGenConfig, Path], Iterable[SinkBase]]
    optimizer: OptimizerAdapter
    gap_fill: Callable[..., tuple[str, dict] | str]


def default_deps() -> PipelineDeps:
    return PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=build_sinks,
        optimizer=DenseArraysAdapter(),
        gap_fill=random_fill,
    )


def resolve_plan(loaded: LoadedConfig) -> List[ResolvedPlanItem]:
    return loaded.root.densegen.generation.resolve_plan()


def select_solver_strict(
    preferred: str | None,
    optimizer: OptimizerAdapter,
    *,
    strategy: str,
    test_length: int = 10,
) -> str | None:
    """
    Probe the requested solver once. If it fails, raise with instructions.
    No fallback behavior.
    """
    if strategy == "approximate":
        return preferred
    if not preferred:
        raise ValueError("solver.backend is required unless strategy=approximate")
    optimizer.probe_solver(preferred, test_length=test_length)
    return preferred


def _summarize_tf_counts(labels: List[str], max_items: int = 6) -> str:
    if not labels:
        return ""
    counts = Counter(labels)
    items = [f"{tf}×{n}" for tf, n in counts.most_common(max_items)]
    extra = len(counts) - min(len(counts), max_items)
    return ", ".join(items) + (f" (+{extra} TFs)" if extra > 0 else "")


def _min_count_by_regulator(labels: list[str] | None, min_count_per_tf: int) -> dict[str, int] | None:
    if not labels or min_count_per_tf <= 0:
        return None
    return {tf: int(min_count_per_tf) for tf in sorted(set(labels))}


def _merge_min_counts(base: dict[str, int] | None, override: dict[str, int] | None) -> dict[str, int] | None:
    if not base and not override:
        return None
    merged = dict(base or {})
    for key, val in (override or {}).items():
        merged[key] = max(int(val), merged.get(key, 0))
    return merged


def _extract_side_biases(fixed_elements) -> tuple[list[str], list[str]]:
    if fixed_elements is None:
        return [], []
    if hasattr(fixed_elements, "side_biases"):
        sb = getattr(fixed_elements, "side_biases")
    else:
        sb = (fixed_elements or {}).get("side_biases")
    if sb is None:
        return [], []
    if hasattr(sb, "left") or hasattr(sb, "right"):
        left = list(getattr(sb, "left", []) or [])
        right = list(getattr(sb, "right", []) or [])
    else:
        left = list((sb or {}).get("left") or [])
        right = list((sb or {}).get("right") or [])
    return left, right


def _fixed_elements_dump(fixed_elements) -> dict:
    if fixed_elements is None:
        return {
            "promoter_constraints": [],
            "side_biases": {"left": [], "right": []},
        }
    if hasattr(fixed_elements, "model_dump"):
        dump = fixed_elements.model_dump()
    else:
        dump = dict(fixed_elements or {})

    pcs_raw = dump.get("promoter_constraints") or []
    pcs = []
    keys = ("name", "upstream", "downstream", "spacer_length", "upstream_pos", "downstream_pos")
    for pc in pcs_raw:
        if hasattr(pc, "model_dump"):
            pc = pc.model_dump()
        if isinstance(pc, dict):
            pcs.append({k: pc.get(k) for k in keys})

    sb_raw = dump.get("side_biases") or {}
    if hasattr(sb_raw, "model_dump"):
        sb_raw = sb_raw.model_dump()
    left = list((sb_raw or {}).get("left") or [])
    right = list((sb_raw or {}).get("right") or [])

    return {"promoter_constraints": pcs, "side_biases": {"left": left, "right": right}}


def _input_metadata(source_cfg, cfg_path: Path) -> dict:
    source_type = getattr(source_cfg, "type", "unknown")
    source_name = getattr(source_cfg, "name", "unknown")
    meta = {"input_type": source_type, "input_name": source_name}
    if hasattr(source_cfg, "path"):
        meta["input_path"] = str(resolve_relative_path(cfg_path, getattr(source_cfg, "path")))
    if source_type == "usr_sequences":
        meta["input_dataset"] = getattr(source_cfg, "dataset", None)
        meta["input_root"] = str(resolve_relative_path(cfg_path, getattr(source_cfg, "root")))
        meta["input_mode"] = "sequence_library"
        meta["input_pwm_ids"] = []
    elif source_type == "sequence_library":
        meta["input_mode"] = "sequence_library"
        meta["input_pwm_ids"] = []
    elif source_type == "binding_sites":
        meta["input_mode"] = "binding_sites"
        meta["input_pwm_ids"] = []
    elif source_type in {"pwm_meme", "pwm_jaspar", "pwm_matrix_csv"}:
        meta["input_mode"] = "pwm_sampled"
        if source_type == "pwm_matrix_csv":
            motif_id = getattr(source_cfg, "motif_id", None)
            meta["input_pwm_ids"] = [motif_id] if motif_id else []
        else:
            meta["input_pwm_ids"] = list(getattr(source_cfg, "motif_ids") or [])
        sampling = getattr(source_cfg, "sampling", None)
        if sampling is not None:
            meta["input_pwm_strategy"] = getattr(sampling, "strategy", None)
            meta["input_pwm_score_threshold"] = getattr(sampling, "score_threshold", None)
            meta["input_pwm_score_percentile"] = getattr(sampling, "score_percentile", None)
            meta["input_pwm_n_sites"] = getattr(sampling, "n_sites", None)
            meta["input_pwm_oversample_factor"] = getattr(sampling, "oversample_factor", None)
    else:
        meta["input_mode"] = source_type
        meta["input_pwm_ids"] = []
    return meta


def _compute_used_tf_info(sol, library_for_opt, regulator_labels, fixed_elements):
    promoter_motifs = set()
    if fixed_elements is not None:
        if hasattr(fixed_elements, "promoter_constraints"):
            pcs = getattr(fixed_elements, "promoter_constraints") or []
        else:
            pcs = (fixed_elements or {}).get("promoter_constraints") or []
        if pcs:
            pc = pcs[0]
            if hasattr(pc, "upstream") or hasattr(pc, "downstream"):
                up = getattr(pc, "upstream", None)
                dn = getattr(pc, "downstream", None)
                for v in (up, dn):
                    if isinstance(v, str) and v.strip():
                        promoter_motifs.add(v.strip().upper())
            elif isinstance(pc, dict):
                for k in ("upstream", "downstream"):
                    v = pc.get(k)
                    if isinstance(v, str) and v.strip():
                        promoter_motifs.add(v.strip().upper())

    lib = getattr(sol, "library", [])
    orig_n = len(library_for_opt)
    used_simple: list[str] = []
    used_detail: list[dict] = []
    counts: dict[str, int] = {}
    used_tf_set: set[str] = set()

    for offset, idx in sol.offset_indices_in_order():
        base_idx = idx % len(lib)
        orientation = "fwd" if idx < len(lib) else "rev"
        motif = lib[base_idx]
        if motif in promoter_motifs or base_idx >= orig_n:
            continue
        tf_label = (
            regulator_labels[base_idx] if regulator_labels is not None and base_idx < len(regulator_labels) else ""
        )
        tfbs = motif
        used_simple.append(f"{tf_label}:{tfbs}" if tf_label else tfbs)
        used_detail.append({"tf": tf_label, "tfbs": tfbs, "orientation": orientation, "offset": int(offset)})
        if tf_label:
            counts[tf_label] = counts.get(tf_label, 0) + 1
            used_tf_set.add(tf_label)
    return used_simple, used_detail, counts, sorted(used_tf_set)


def _assert_sink_alignment(sinks: list[SinkBase]) -> None:
    if len(sinks) <= 1:
        return
    sink_types = [type(s).__name__ for s in sinks]
    if len(set(sink_types)) != len(sink_types):
        raise RuntimeError("Duplicate sink types detected; output.targets must be unique.")
    digests = []
    for sink in sinks:
        digest = sink.alignment_digest()
        if digest is None:
            raise RuntimeError(
                f"Sink {type(sink).__name__} does not provide alignment digest; alignment requires digest support."
            )
        digests.append((type(sink).__name__, digest))

    baseline_name, baseline = digests[0]
    mismatches = []
    for name, digest in digests[1:]:
        if digest != baseline:
            mismatches.append((name, digest.id_count, digest.xor_hash))
    if mismatches:
        details = "; ".join(f"{name}: count={cnt} hash={h}" for name, cnt, h in mismatches)
        raise RuntimeError(
            "Output sinks are out of sync before run. "
            f"Baseline={baseline_name} count={baseline.id_count} hash={baseline.xor_hash}. "
            f"Differences: {details}. Remove stale outputs or run with a single target to rebuild."
        )


def _write_to_sinks(sinks: list[SinkBase], record: OutputRecord) -> bool:
    results = [sink.add(record) for sink in sinks]
    if not results:
        raise RuntimeError("No output sinks configured.")
    if all(results):
        return True
    if not any(results):
        return False
    raise RuntimeError("Output sinks are inconsistent (some accepted, some rejected).")


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
    output_bio_type: str,
    output_alphabet: str,
    one_subsample_only: bool = False,
    already_generated: int = 0,
) -> int:
    source_label = source_cfg.name
    plan_name = plan_item.name
    quota = int(plan_item.quota)

    gen = global_cfg.generation
    seq_len = int(gen.sequence_length)
    sampling_cfg = gen.sampling

    pool_strategy = str(sampling_cfg.pool_strategy)
    library_size = int(sampling_cfg.library_size)
    subsample_over = int(sampling_cfg.subsample_over_length_budget_by)
    cover_all_tfs = bool(sampling_cfg.cover_all_regulators)
    unique_binding_sites = bool(sampling_cfg.unique_binding_sites)
    max_sites_per_tf = sampling_cfg.max_sites_per_regulator
    relax_on_exhaustion = bool(sampling_cfg.relax_on_exhaustion)
    allow_incomplete_coverage = bool(sampling_cfg.allow_incomplete_coverage)
    iterative_max_libraries = int(sampling_cfg.iterative_max_libraries)
    iterative_min_new_solutions = int(sampling_cfg.iterative_min_new_solutions)

    runtime_cfg = global_cfg.runtime
    max_per_subsample = int(runtime_cfg.arrays_generated_before_resample)
    min_count_per_tf = int(runtime_cfg.min_count_per_tf)
    max_dupes = int(runtime_cfg.max_duplicate_solutions)
    max_resample_attempts = int(runtime_cfg.max_resample_attempts)
    stall_seconds = int(runtime_cfg.stall_seconds_before_resample)
    stall_warn_every = int(runtime_cfg.stall_warning_every_seconds)
    max_total_resamples = int(runtime_cfg.max_total_resamples)
    max_seconds_per_plan = int(runtime_cfg.max_seconds_per_plan)
    max_failed_solutions = int(runtime_cfg.max_failed_solutions)

    post = global_cfg.postprocess
    gap_cfg = post.gap_fill
    fill_gap = gap_cfg.mode != "off"
    fill_mode = gap_cfg.mode
    fill_end = gap_cfg.end
    fill_gc_min = float(gap_cfg.gc_min)
    fill_gc_max = float(gap_cfg.gc_max)
    fill_max_tries = int(gap_cfg.max_tries)

    solver_cfg = global_cfg.solver
    solver_opts = list(solver_cfg.options)
    solver_strategy = str(solver_cfg.strategy)
    solver_strands = str(solver_cfg.strands)

    log_cfg = global_cfg.logging
    print_visual = bool(log_cfg.print_visual)

    policy_gc_fill = str(fill_mode)
    policy_sampling = pool_strategy
    policy_solver = solver_strategy

    plan_start = time.monotonic()
    total_resamples = 0
    failed_solutions = 0

    # Load source
    src_obj = deps.source_factory(source_cfg, cfg_path)
    data_entries, meta_df = src_obj.load_data(rng=np_rng)
    input_meta = _input_metadata(source_cfg, cfg_path)
    fixed_elements = plan_item.fixed_elements
    required_regulators = list(dict.fromkeys(plan_item.required_regulators or []))
    min_required_regulators = plan_item.min_required_regulators
    plan_min_count_by_regulator = dict(plan_item.min_count_by_regulator or {})
    required_regulators_effective = list(dict.fromkeys([*required_regulators, *plan_min_count_by_regulator.keys()]))
    metadata_min_counts = {tf: max(min_count_per_tf, int(val)) for tf, val in plan_min_count_by_regulator.items()}
    side_left, side_right = _extract_side_biases(fixed_elements)
    required_bias_motifs = list(dict.fromkeys([*side_left, *side_right]))
    fixed_elements_dump = _fixed_elements_dump(fixed_elements)

    # Build initial library
    library_for_opt: List[str]
    tfbs_parts: List[str]
    libraries_built = 0

    if pool_strategy != "iterative_subsample":
        max_per_subsample = quota

    def _build_library() -> tuple[list[str], list[str], list[str], dict]:
        nonlocal libraries_built
        if meta_df is not None and isinstance(meta_df, pd.DataFrame):
            available_tfs = set(meta_df["tf"].tolist())
            missing = [t for t in required_regulators if t not in available_tfs]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Required regulators not found in input: {preview}")
            if plan_min_count_by_regulator:
                missing_counts = [t for t in plan_min_count_by_regulator if t not in available_tfs]
                if missing_counts:
                    preview = ", ".join(missing_counts[:10])
                    raise ValueError(f"min_count_by_regulator TFs not found in input: {preview}")
            if min_required_regulators is not None and min_required_regulators > len(available_tfs):
                raise ValueError(
                    f"min_required_regulators={min_required_regulators} exceeds available regulators "
                    f"({len(available_tfs)})."
                )

            if pool_strategy == "full":
                lib_df = meta_df.copy()
                if unique_binding_sites:
                    lib_df = lib_df.drop_duplicates(["tf", "tfbs"])
                if required_bias_motifs:
                    missing_bias = [m for m in required_bias_motifs if m not in set(lib_df["tfbs"])]
                    if missing_bias:
                        preview = ", ".join(missing_bias[:10])
                        raise ValueError(f"Required side-bias motifs not found in input: {preview}")
                lib_df = lib_df.reset_index(drop=True)
                library = lib_df["tfbs"].tolist()
                reg_labels = lib_df["tf"].tolist()
                parts = [f"{tf}:{tfbs}" for tf, tfbs in zip(reg_labels, lib_df["tfbs"].tolist())]
                info = {
                    "target_length": seq_len + subsample_over,
                    "achieved_length": sum(len(s) for s in library),
                    "relaxed_cap": False,
                    "final_cap": None,
                    "pool_strategy": pool_strategy,
                    "library_size": len(library),
                    "iterative_max_libraries": iterative_max_libraries,
                    "iterative_min_new_solutions": iterative_min_new_solutions,
                }
                libraries_built += 1
                return library, parts, reg_labels, info

            sampler = TFSampler(meta_df, np_rng)
            library, parts, reg_labels, info = sampler.generate_binding_site_subsample(
                seq_len,
                subsample_over,
                required_tfbs=required_bias_motifs,
                required_tfs=required_regulators_effective,
                cover_all_tfs=cover_all_tfs,
                unique_binding_sites=unique_binding_sites,
                max_sites_per_tf=max_sites_per_tf,
                relax_on_exhaustion=relax_on_exhaustion,
                allow_incomplete_coverage=allow_incomplete_coverage,
            )
            info.update(
                {
                    "pool_strategy": pool_strategy,
                    "library_size": library_size,
                    "iterative_max_libraries": iterative_max_libraries,
                    "iterative_min_new_solutions": iterative_min_new_solutions,
                }
            )
            libraries_built += 1
            return library, parts, reg_labels, info

        # Sequence library (no regulators)
        if required_regulators or plan_min_count_by_regulator or min_required_regulators is not None:
            preview = ", ".join(required_regulators[:10]) if required_regulators else "n/a"
            raise ValueError(
                "Regulator constraints are set (required/min_count/min_required) "
                f"but the input does not provide regulators. required_regulators={preview}."
            )
        all_sequences = [s for s in data_entries]
        if not all_sequences:
            raise ValueError(f"No sequences found for source {source_label}")
        pool = list(dict.fromkeys(all_sequences)) if unique_binding_sites else list(all_sequences)
        if pool_strategy == "full":
            if required_bias_motifs:
                missing = [m for m in required_bias_motifs if m not in pool]
                if missing:
                    preview = ", ".join(missing[:10])
                    raise ValueError(f"Required side-bias motifs not found in sequences input: {preview}")
            library = pool
        else:
            if library_size > len(pool):
                raise ValueError(f"library_size={library_size} exceeds available unique sequences ({len(pool)}).")
            take = min(max(1, int(library_size)), len(pool))
            if required_bias_motifs:
                missing = [m for m in required_bias_motifs if m not in pool]
                if missing:
                    preview = ", ".join(missing[:10])
                    raise ValueError(f"Required side-bias motifs not found in sequences input: {preview}")
                if take < len(required_bias_motifs):
                    raise ValueError(
                        f"library_size={take} is smaller than required side_biases ({len(required_bias_motifs)})."
                    )
                required_set = set(required_bias_motifs)
                remaining = [s for s in pool if s not in required_set]
                library = list(required_bias_motifs) + rng.sample(remaining, take - len(required_bias_motifs))
            else:
                library = rng.sample(pool, take)
        tf_parts: list[str] = []
        reg_labels: list[str] = []
        info = {
            "target_length": seq_len + subsample_over,
            "achieved_length": sum(len(s) for s in library),
            "relaxed_cap": False,
            "final_cap": None,
            "pool_strategy": pool_strategy,
            "library_size": len(library) if pool_strategy == "full" else library_size,
            "iterative_max_libraries": iterative_max_libraries,
            "iterative_min_new_solutions": iterative_min_new_solutions,
        }
        libraries_built += 1
        return library, tf_parts, reg_labels, info

    library_for_opt, tfbs_parts, regulator_labels, sampling_info = _build_library()

    # Library summary (succinct)
    tf_summary = _summarize_tf_counts(regulator_labels)
    if tf_summary:
        log.info(
            "Library for %s/%s: %d motifs | TF counts: %s",
            source_label,
            plan_name,
            len(library_for_opt),
            tf_summary,
        )
    else:
        log.info(
            "Library for %s/%s: %d motifs",
            source_label,
            plan_name,
            len(library_for_opt),
        )

    solver_min_counts: dict[str, int] | None = None

    def _make_generator(_library_for_opt: List[str], _regulator_labels: List[str]):
        nonlocal solver_min_counts
        regulator_by_index = list(_regulator_labels) if _regulator_labels else None
        base_min_counts = _min_count_by_regulator(regulator_by_index, min_count_per_tf)
        solver_min_counts = _merge_min_counts(base_min_counts, plan_min_count_by_regulator)
        fe_dict = fixed_elements.model_dump() if hasattr(fixed_elements, "model_dump") else fixed_elements
        run = deps.optimizer.build(
            library=_library_for_opt,
            sequence_length=seq_len,
            solver=chosen_solver,
            strategy=solver_strategy,
            solver_options=solver_opts,
            fixed_elements=fe_dict,
            strands=solver_strands,
            regulator_by_index=regulator_by_index,
            required_regulators=required_regulators,
            min_count_by_regulator=solver_min_counts,
            min_required_regulators=min_required_regulators,
        )
        return run

    run = _make_generator(library_for_opt, regulator_labels)
    opt = run.optimizer
    generator = run.generator
    forbid_each = run.forbid_each

    global_generated = already_generated
    produced_total_this_call = 0

    while global_generated < quota:
        if max_seconds_per_plan > 0 and (time.monotonic() - plan_start) > max_seconds_per_plan:
            raise RuntimeError(f"[{source_label}/{plan_name}] Exceeded max_seconds_per_plan={max_seconds_per_plan}.")
        local_generated = 0
        resamples_in_try = 0

        while local_generated < max_per_subsample and global_generated < quota:
            fingerprints = set()
            consecutive_dup = 0
            subsample_started = time.monotonic()
            last_log_warn = subsample_started
            produced_this_library = 0

            for sol in generator:
                now = time.monotonic()
                if (now - subsample_started >= stall_seconds) and (produced_this_library == 0):
                    log.info(
                        "[%s/%s] Stall (> %ds) with no solutions; will resample.",
                        source_label,
                        plan_name,
                        stall_seconds,
                    )
                    break
                if (now - last_log_warn >= stall_warn_every) and (produced_this_library == 0):
                    log.info(
                        "[%s/%s] Still working... %.1fs on current library.",
                        source_label,
                        plan_name,
                        now - subsample_started,
                    )
                    last_log_warn = now

                if forbid_each:
                    opt.forbid(sol)
                seq = sol.sequence
                if seq in fingerprints:
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
                    sol, library_for_opt, regulator_labels, fixed_elements
                )
                tf_list_from_library = sorted(set(regulator_labels)) if regulator_labels else []

                covers_all = True
                covers_required = True
                if min_count_per_tf > 0 and tf_list_from_library:
                    missing = [tf for tf in tf_list_from_library if used_tf_counts.get(tf, 0) < min_count_per_tf]
                    if missing:
                        covers_all = False
                        failed_solutions += 1
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
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                if min_required_regulators is not None:
                    if len(used_tf_list) < int(min_required_regulators):
                        failed_solutions += 1
                        if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                            raise RuntimeError(
                                f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                            )
                        continue

                gap_meta = {"used": False}
                final_seq = seq
                if not fill_gap and len(final_seq) < seq_len:
                    raise RuntimeError(
                        f"[{source_label}/{plan_name}] Sequence shorter than target and gap_fill.mode=off."
                    )
                if fill_gap and len(final_seq) < seq_len:
                    gap = seq_len - len(final_seq)
                    rf = deps.gap_fill(
                        gap,
                        fill_gc_min,
                        fill_gc_max,
                        max_tries=fill_max_tries,
                        mode=fill_mode,
                        rng=rng,
                    )
                    if isinstance(rf, tuple) and len(rf) == 2:
                        pad, pad_info = rf
                        pad_info = pad_info or {}
                    else:
                        pad, pad_info = rf, {}
                    final_seq = (pad + final_seq) if fill_end == "5prime" else (final_seq + pad)
                    gap_meta = {
                        "used": True,
                        "bases": gap,
                        "end": fill_end,
                        "gc_min": pad_info.get("final_gc_min", fill_gc_min),
                        "gc_max": pad_info.get("final_gc_max", fill_gc_max),
                        "gc_target_min": pad_info.get("target_gc_min", fill_gc_min),
                        "gc_target_max": pad_info.get("target_gc_max", fill_gc_max),
                        "gc_actual": pad_info.get("gc_actual"),
                        "relaxed": pad_info.get("relaxed"),
                        "attempts": pad_info.get("attempts"),
                    }

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
                    solver_options=solver_opts,
                    solver_strands=solver_strands,
                    seq_len=seq_len,
                    actual_length=len(final_seq),
                    gap_meta=gap_meta,
                    sampling_meta=sampling_info,
                    schema_version=str(global_cfg.schema_version),
                    created_at=created_at,
                    run_id=run_id,
                    run_root=run_root,
                    run_config_path=run_config_path,
                    run_config_sha256=run_config_sha256,
                    random_seed=random_seed,
                    policy_gc_fill=policy_gc_fill,
                    policy_sampling=policy_sampling,
                    policy_solver=policy_solver,
                    input_meta=input_meta,
                    fixed_elements_dump=fixed_elements_dump,
                    used_tfbs=used_tfbs,
                    used_tfbs_detail=used_tfbs_detail,
                    used_tf_counts=used_tf_counts,
                    used_tf_list=used_tf_list,
                    min_count_per_tf=min_count_per_tf,
                    covers_all_tfs_in_solution=covers_all,
                    required_regulators=required_regulators,
                    min_required_regulators=min_required_regulators,
                    min_count_by_regulator=metadata_min_counts,
                    covers_required_regulators=covers_required,
                )

                src_label = f"densegen:{source_label}:{plan_name}"
                record = OutputRecord.from_sequence(
                    sequence=final_seq,
                    meta=derived,
                    source=src_label,
                    bio_type=output_bio_type,
                    alphabet=output_alphabet,
                )
                accepted = _write_to_sinks(sinks, record)
                if not accepted:
                    failed_solutions += 1
                    if max_failed_solutions > 0 and failed_solutions > max_failed_solutions:
                        raise RuntimeError(
                            f"[{source_label}/{plan_name}] Exceeded max_failed_solutions={max_failed_solutions}."
                        )
                    log.info(
                        "[%s/%s] Output duplicate detected; skipping.",
                        source_label,
                        plan_name,
                    )
                    continue

                global_generated += 1
                local_generated += 1
                produced_this_library += 1
                produced_total_this_call += 1

                pct = 100.0 * (global_generated / max(1, quota))
                cr = getattr(sol, "compression_ratio", float("nan"))
                if print_visual:
                    log.info(
                        "╭─ %s/%s  %d/%d (%.2f%%) — local %d/%d — CR=%.3f\n"
                        "%s\nsequence %s\n"
                        "╰────────────────────────────────────────────────────────",
                        source_label,
                        plan_name,
                        global_generated,
                        quota,
                        pct,
                        local_generated,
                        max_per_subsample,
                        cr,
                        derived["visual"],
                        final_seq,
                    )
                else:
                    log.info(
                        "[%s/%s] %d/%d (%.2f%%) (local %d/%d) CR=%.3f | seq %s",
                        source_label,
                        plan_name,
                        global_generated,
                        quota,
                        pct,
                        local_generated,
                        max_per_subsample,
                        cr,
                        final_seq,
                    )

                if local_generated >= max_per_subsample or global_generated >= quota:
                    break

            if local_generated >= max_per_subsample or global_generated >= quota:
                break

            if pool_strategy == "iterative_subsample" and iterative_min_new_solutions > 0:
                if produced_this_library < iterative_min_new_solutions:
                    log.info(
                        "[%s/%s] Library produced %d < iterative_min_new_solutions=%d; resampling.",
                        source_label,
                        plan_name,
                        produced_this_library,
                        iterative_min_new_solutions,
                    )

            # Resample
            if pool_strategy != "iterative_subsample":
                raise RuntimeError(
                    f"[{source_label}/{plan_name}] pool_strategy={pool_strategy!r} does not allow resampling. "
                    "Reduce quota or use iterative_subsample."
                )
            resamples_in_try += 1
            total_resamples += 1
            if max_total_resamples > 0 and total_resamples > max_total_resamples:
                raise RuntimeError(f"[{source_label}/{plan_name}] Exceeded max_total_resamples={max_total_resamples}.")
            if resamples_in_try > max_resample_attempts:
                log.info(
                    "[%s/%s] Reached max_resample_attempts (%d) for this subsample try "
                    "(produced %d/%d here). Moving on.",
                    source_label,
                    plan_name,
                    max_resample_attempts,
                    local_generated,
                    max_per_subsample,
                )
                break

            if iterative_max_libraries > 0 and libraries_built >= iterative_max_libraries:
                raise RuntimeError(
                    f"[{source_label}/{plan_name}] Exceeded iterative_max_libraries={iterative_max_libraries}."
                )

            # New library
            library_for_opt, tfbs_parts, regulator_labels, sampling_info = _build_library()
            tf_summary = _summarize_tf_counts(regulator_labels)
            if tf_summary:
                log.info(
                    "Resampled library for %s/%s: %d motifs | TF counts: %s",
                    source_label,
                    plan_name,
                    len(library_for_opt),
                    tf_summary,
                )
            else:
                log.info(
                    "Resampled library for %s/%s: %d motifs",
                    source_label,
                    plan_name,
                    len(library_for_opt),
                )

            run = _make_generator(library_for_opt, regulator_labels)
            opt = run.optimizer
            generator = run.generator
            forbid_each = run.forbid_each

        for sink in sinks:
            sink.flush()

        if one_subsample_only:
            return produced_total_this_call

    log.info("Completed %s/%s: %d/%d", source_label, plan_name, global_generated, quota)
    return produced_total_this_call


def run_pipeline(loaded: LoadedConfig, *, deps: PipelineDeps | None = None) -> RunSummary:
    deps = deps or default_deps()
    cfg = loaded.root.densegen
    run_root = resolve_run_root(loaded.path, cfg.run.root)
    run_root_str = str(run_root)
    config_sha = hashlib.sha256(loaded.path.read_bytes()).hexdigest()
    try:
        run_cfg_path = str(loaded.path.relative_to(run_root))
    except ValueError:
        run_cfg_path = str(loaded.path)

    # Seed
    seed = int(cfg.runtime.random_seed)
    random.seed(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Plan & solver
    pl = cfg.generation.resolve_plan()
    chosen_solver = select_solver_strict(cfg.solver.backend, deps.optimizer, strategy=str(cfg.solver.strategy))

    # Build sinks
    sinks = list(deps.sink_factory(cfg, loaded.path))
    _assert_sink_alignment(sinks)
    output_bio_type, output_alphabet = resolve_bio_alphabet(cfg)

    total = 0
    per_plan: dict[tuple[str, str], int] = {}

    # Round-robin scheduler
    round_robin = bool(cfg.runtime.round_robin)
    inputs = cfg.inputs
    if not round_robin:
        for s in inputs:
            for item in pl:
                produced = _process_plan_for_source(
                    s,
                    item,
                    cfg,
                    sinks,
                    chosen_solver=chosen_solver,
                    deps=deps,
                    rng=rng,
                    np_rng=np_rng,
                    cfg_path=loaded.path,
                    run_id=cfg.run.id,
                    run_root=run_root_str,
                    run_config_path=run_cfg_path,
                    run_config_sha256=config_sha,
                    random_seed=seed,
                    output_bio_type=output_bio_type,
                    output_alphabet=output_alphabet,
                    one_subsample_only=False,
                    already_generated=0,
                )
                per_plan[(s.name, item.name)] = per_plan.get((s.name, item.name), 0) + produced
                total += produced
    else:
        produced_counts: dict[tuple[str, str], int] = {}
        done = False
        while not done:
            done = True
            for s in inputs:
                for item in pl:
                    key = (s.name, item.name)
                    current = produced_counts.get(key, 0)
                    quota = int(item.quota)
                    if current >= quota:
                        continue
                    done = False
                    produced = _process_plan_for_source(
                        s,
                        item,
                        cfg,
                        sinks,
                        chosen_solver=chosen_solver,
                        deps=deps,
                        rng=rng,
                        np_rng=np_rng,
                        cfg_path=loaded.path,
                        run_id=cfg.run.id,
                        run_root=run_root_str,
                        run_config_path=run_cfg_path,
                        run_config_sha256=config_sha,
                        random_seed=seed,
                        output_bio_type=output_bio_type,
                        output_alphabet=output_alphabet,
                        one_subsample_only=True,
                        already_generated=current,
                    )
                    produced_counts[key] = current + produced
        per_plan = produced_counts
        total = sum(per_plan.values())

    for sink in sinks:
        sink.flush()

    return RunSummary(total_generated=total, per_plan=per_plan)
