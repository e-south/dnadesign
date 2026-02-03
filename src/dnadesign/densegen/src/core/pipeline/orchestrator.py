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
import math
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
from ...adapters.outputs import OutputRecord, SinkBase, build_sinks, load_records_from_config, resolve_bio_alphabet
from ...adapters.sources import data_source_factory
from ...adapters.sources.base import BaseDataSource
from ...config import (
    DenseGenConfig,
    LoadedConfig,
    ResolvedPlanItem,
    resolve_outputs_scoped_path,
    resolve_run_root,
)
from ...utils import logging_utils
from ...utils.logging_utils import install_native_stderr_filters
from ..artifacts.candidates import build_candidate_artifact, find_candidate_files, prepare_candidates_dir
from ..artifacts.library import (
    LibraryArtifact,
    LibraryRecord,
    load_library_artifact,
    load_library_records,
    write_library_artifact,
)
from ..artifacts.pool import (
    POOL_MODE_SEQUENCE,
    POOL_MODE_TFBS,
    PoolData,
    _hash_file,
    build_pool_artifact,
    load_pool_data,
    pool_status_by_input,
)
from ..artifacts.records import SolutionRecord
from ..input_types import PWM_INPUT_TYPES
from ..metadata import build_metadata
from ..motif_labels import input_motifs, motif_display_name
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
    run_state_path,
    run_tables_root,
)
from ..run_state import RunState, load_run_state
from ..runtime_policy import RuntimePolicy
from ..seeding import derive_seed_map
from .attempts import (
    SOLUTIONS_CHUNK_SIZE,
    _append_attempt,
    _flush_attempts,
    _flush_solutions,
    _load_existing_attempt_index_by_plan,
    _load_existing_library_index,
    _load_existing_library_index_by_plan,
    _load_failure_counts_from_attempts,
    _log_rejection,
)
from .inputs import (
    _budget_attr,
    _build_input_manifest_entry,
    _input_metadata,
    _mining_attr,
    _sampling_attr,
)
from .outputs import (
    _assert_sink_alignment,
    _consolidate_parts,
    _emit_event,
    _write_effective_config,
    _write_to_sinks,
)
from .plan_pools import PLAN_POOL_INPUT_TYPE, PlanPoolSpec, build_plan_pools
from .progress import (
    _build_screen_dashboard,
    _format_progress_bar,
    _leaderboard_snapshot,
    _ScreenDashboard,
    _short_seq,
    _summarize_diversity,
    _summarize_failure_leaderboard,
    _summarize_failure_totals,
    _summarize_leaderboard,
    _summarize_tf_counts,
    _summarize_tfbs_usage_stats,
)
from .stage_b import (
    _compute_sampling_fraction,
    _compute_sampling_fraction_pairs,
    _fixed_elements_dump,
    _max_fixed_element_len,
    _merge_min_counts,
    _min_count_by_regulator,
    assess_library_feasibility,
    build_library_for_plan,
)
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


def _write_run_state(
    path: Path,
    *,
    run_id: str,
    schema_version: str,
    config_sha256: str,
    run_root: str,
    counts: dict[tuple[str, str], int],
    created_at: str,
) -> None:
    state = RunState.from_counts(
        run_id=run_id,
        schema_version=schema_version,
        config_sha256=config_sha256,
        run_root=run_root,
        counts=counts,
        created_at=created_at,
    )
    state.write_json(path)


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


@dataclass(frozen=True)
class PlanPoolSource:
    name: str
    type: str = PLAN_POOL_INPUT_TYPE


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


def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq)


def _compute_used_tf_info(
    sol,
    library_for_opt,
    regulator_labels,
    fixed_elements,
    site_id_by_index,
    source_by_index,
    tfbs_id_by_index,
    motif_id_by_index,
):
    promoter_motifs = set()
    if fixed_elements is not None:
        if hasattr(fixed_elements, "promoter_constraints"):
            pcs = getattr(fixed_elements, "promoter_constraints") or []
        else:
            pcs = (fixed_elements or {}).get("promoter_constraints") or []
        for pc in pcs:
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
        entry = {
            "tf": tf_label,
            "tfbs": tfbs,
            "orientation": orientation,
            "offset": int(offset),
            "offset_raw": int(offset),
            "length": len(tfbs),
            "end": int(offset) + len(tfbs),
            "pad_left": 0,
        }
        if site_id_by_index is not None and base_idx < len(site_id_by_index):
            site_id = site_id_by_index[base_idx]
            if site_id is not None:
                entry["site_id"] = site_id
        if source_by_index is not None and base_idx < len(source_by_index):
            source = source_by_index[base_idx]
            if source is not None:
                entry["source"] = source
        if tfbs_id_by_index is not None and base_idx < len(tfbs_id_by_index):
            tfbs_id = tfbs_id_by_index[base_idx]
            if tfbs_id is not None:
                entry["tfbs_id"] = tfbs_id
        if motif_id_by_index is not None and base_idx < len(motif_id_by_index):
            motif_id = motif_id_by_index[base_idx]
            if motif_id is not None:
                entry["motif_id"] = motif_id
        used_detail.append(entry)
        if tf_label:
            counts[tf_label] = counts.get(tf_label, 0) + 1
            used_tf_set.add(tf_label)
    return used_simple, used_detail, counts, sorted(used_tf_set)


def _parse_used_tfbs_detail(val) -> list[dict]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                val = json.loads(s)
            except Exception as exc:
                raise ValueError(f"Failed to parse used_tfbs_detail JSON: {s[:120]}") from exc
    if isinstance(val, (list, tuple, np.ndarray)):
        out: list[dict] = []
        for item in list(val):
            if isinstance(item, dict):
                out.append(item)
        return out
    return []


def _update_usage_counts(
    usage_counts: dict[tuple[str, str], int],
    used_tfbs_detail: list[dict],
) -> None:
    for entry in used_tfbs_detail:
        tf = str(entry.get("tf") or "").strip()
        tfbs = str(entry.get("tfbs") or "").strip()
        if not tf or not tfbs:
            continue
        key = (tf, tfbs)
        usage_counts[key] = int(usage_counts.get(key, 0)) + 1


def _apply_pad_offsets(used_tfbs_detail: list[dict], pad_meta: dict) -> list[dict]:
    pad_left = 0
    if pad_meta.get("used") and pad_meta.get("end") == "5prime":
        pad_left = int(pad_meta.get("bases") or 0)
    for entry in used_tfbs_detail:
        offset_raw = int(entry.get("offset_raw", entry.get("offset", 0)))
        length = int(entry.get("length", len(entry.get("tfbs") or "")))
        offset = offset_raw + pad_left
        entry["offset_raw"] = offset_raw
        entry["pad_left"] = pad_left
        entry["length"] = length
        entry["offset"] = offset
        entry["end"] = offset + length
    return used_tfbs_detail


def _find_motif_positions(seq: str, motif: str, bounds: tuple[int, int] | list[int] | None) -> list[int]:
    seq = str(seq)
    motif = str(motif)
    if not motif:
        return []
    lo = None
    hi = None
    if bounds is not None:
        try:
            lo = int(bounds[0])
            hi = int(bounds[1])
        except Exception:
            lo = None
            hi = None
    positions: list[int] = []
    start = 0
    while start < len(seq):
        idx = seq.find(motif, start)
        if idx == -1:
            break
        if lo is not None and idx < lo:
            start = idx + 1
            continue
        if hi is not None and idx > hi:
            start = idx + 1
            continue
        if idx + len(motif) <= len(seq):
            positions.append(idx)
        start = idx + 1
    return positions


def _promoter_windows(seq: str, fixed_elements_dump: dict) -> list[tuple[int, int]]:
    pcs = fixed_elements_dump.get("promoter_constraints") or []
    windows: list[tuple[int, int]] = []
    for pc in pcs:
        if not isinstance(pc, dict):
            continue
        upstream = str(pc.get("upstream") or "").strip().upper()
        downstream = str(pc.get("downstream") or "").strip().upper()
        if not upstream or not downstream:
            continue
        spacer = pc.get("spacer_length")
        if isinstance(spacer, (list, tuple)) and spacer:
            spacer_min = int(min(spacer))
            spacer_max = int(max(spacer))
        elif spacer is not None:
            spacer_min = int(spacer)
            spacer_max = int(spacer)
        else:
            spacer_min = None
            spacer_max = None
        up_positions = _find_motif_positions(seq, upstream, pc.get("upstream_pos"))
        down_positions = _find_motif_positions(seq, downstream, pc.get("downstream_pos"))
        if not up_positions or not down_positions:
            continue
        matched = False
        for up_start in sorted(up_positions):
            up_end = up_start + len(upstream)
            for down_start in sorted(down_positions):
                if down_start < up_end:
                    continue
                spacer_len = down_start - up_end
                if spacer_min is not None and spacer_len < spacer_min:
                    continue
                if spacer_max is not None and spacer_len > spacer_max:
                    continue
                windows.append((up_start, up_end))
                windows.append((down_start, down_start + len(downstream)))
                matched = True
                break
            if matched:
                break
    return windows


def _find_forbidden_kmer(seq: str, kmers: list[str], allowed_windows: list[tuple[int, int]]) -> tuple[str, int] | None:
    if not kmers:
        return None
    for kmer in kmers:
        start = 0
        while start < len(seq):
            idx = seq.find(kmer, start)
            if idx == -1:
                break
            end = idx + len(kmer)
            allowed = False
            for win_start, win_end in allowed_windows:
                if idx >= win_start and end <= win_end:
                    allowed = True
                    break
            if not allowed:
                return kmer, idx
            start = idx + 1
    return None


def _validate_library_constraints(
    record: LibraryRecord,
    *,
    groups: list,
    min_count_by_regulator: dict[str, int],
    input_name: str,
    plan_name: str,
) -> None:
    required_regulators_selected = list(record.required_regulators_selected or [])
    if groups:
        if not required_regulators_selected:
            raise RuntimeError(
                f"Library artifact missing required_regulators_selected for {input_name}/{plan_name} "
                f"(library_index={record.library_index}). Rebuild libraries with the current version."
            )
        library_tf_set = {tf for tf in record.library_tfs if tf}
        missing = [tf for tf in required_regulators_selected if tf not in library_tf_set]
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(
                f"Library artifact required_regulators_selected includes TFs not in the library: {preview}."
            )
        group_members = {m for g in groups for m in g.members}
        invalid = [tf for tf in required_regulators_selected if tf not in group_members]
        if invalid:
            preview = ", ".join(invalid[:10])
            raise RuntimeError(
                f"Library artifact required_regulators_selected includes TFs not in regulator groups: {preview}."
            )
        for group in groups:
            selected = [tf for tf in required_regulators_selected if tf in group.members]
            if len(selected) < int(group.min_required):
                raise RuntimeError(
                    f"Library artifact required_regulators_selected does not satisfy group '{group.name}' "
                    f"min_required={group.min_required}."
                )
    for tf, min_count in min_count_by_regulator.items():
        found = sum(1 for t in record.library_tfs if t == tf)
        if found < int(min_count):
            raise RuntimeError(
                f"Library artifact for {input_name}/{plan_name} (library_index={record.library_index}) "
                f"has tf={tf} count={found} < min_count_by_regulator={min_count}."
            )


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

    def _map_tf_usage(counts: dict[str, int]) -> dict[str, int]:
        mapped: dict[str, int] = {}
        for tf, count in counts.items():
            label = _display_tf_label(str(tf))
            mapped[label] = mapped.get(label, 0) + int(count)
        return mapped

    def _map_tfbs_usage(counts: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
        mapped: dict[tuple[str, str], int] = {}
        for (tf, tfbs), count in counts.items():
            label = _display_tf_label(str(tf))
            mapped[(label, str(tfbs))] = mapped.get((label, str(tfbs)), 0) + int(count)
        return mapped

    def _next_attempt_index() -> int:
        key = (source_label, plan_name)
        current = int(attempt_counters.get(key, 0)) + 1
        attempt_counters[key] = current
        return current

    gen = global_cfg.generation
    seq_len = int(gen.sequence_length)
    sampling_cfg = gen.sampling

    def _record_library_build(
        *,
        sampling_info: dict,
        library_tfbs: list[str],
        library_tfs: list[str],
        library_tfbs_ids: list[str],
        library_motif_ids: list[str],
        library_site_ids: list[str | None],
        library_sources: list[str | None],
        fixed_bp: int,
        min_required_bp: int,
        slack_bp: int,
        infeasible: bool,
        sequence_length: int,
    ) -> None:
        if str(getattr(sampling_cfg, "library_source", "build")).lower() == "artifact":
            return
        if library_build_rows is None or library_member_rows is None:
            return
        library_index = int(sampling_info.get("library_index") or 0)
        library_hash = str(sampling_info.get("library_hash") or "")
        library_id = library_hash or f"{source_label}:{plan_name}:{library_index}"
        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_name": source_label,
            "plan_name": plan_name,
            "library_index": library_index,
            "library_id": library_id,
            "library_hash": library_hash,
            "pool_strategy": sampling_info.get("pool_strategy"),
            "library_sampling_strategy": sampling_info.get("library_sampling_strategy"),
            "library_size": int(sampling_info.get("library_size") or len(library_tfbs)),
            "achieved_length": sampling_info.get("achieved_length"),
            "relaxed_cap": sampling_info.get("relaxed_cap"),
            "final_cap": sampling_info.get("final_cap"),
            "iterative_max_libraries": sampling_info.get("iterative_max_libraries"),
            "iterative_min_new_solutions": sampling_info.get("iterative_min_new_solutions"),
            "required_regulators_selected": sampling_info.get("required_regulators_selected"),
            "fixed_bp": int(fixed_bp),
            "min_required_bp": int(min_required_bp),
            "slack_bp": int(slack_bp),
            "infeasible": bool(infeasible),
            "sequence_length": int(sequence_length),
        }
        library_build_rows.append(row)
        if events_path is not None:
            try:
                _emit_event(
                    events_path,
                    event="LIBRARY_BUILT",
                    payload={
                        "input_name": source_label,
                        "plan_name": plan_name,
                        "library_index": library_index,
                        "library_hash": library_hash,
                        "library_size": int(row.get("library_size") or len(library_tfbs)),
                    },
                )
            except Exception:
                log.debug("Failed to emit LIBRARY_BUILT event.", exc_info=True)
            if sampling_info.get("sampling_weight_by_tf"):
                try:
                    _emit_event(
                        events_path,
                        event="LIBRARY_SAMPLING_PRESSURE",
                        payload={
                            "input_name": source_label,
                            "plan_name": plan_name,
                            "library_index": library_index,
                            "library_hash": library_hash,
                            "sampling_strategy": sampling_info.get("library_sampling_strategy"),
                            "weight_by_tf": sampling_info.get("sampling_weight_by_tf"),
                            "weight_fraction_by_tf": sampling_info.get("sampling_weight_fraction_by_tf"),
                            "usage_count_by_tf": sampling_info.get("sampling_usage_count_by_tf"),
                            "failure_count_by_tf": sampling_info.get("sampling_failure_count_by_tf"),
                        },
                    )
                except Exception:
                    log.debug("Failed to emit LIBRARY_SAMPLING_PRESSURE event.", exc_info=True)
        for idx, tfbs in enumerate(library_tfbs):
            library_member_rows.append(
                {
                    "library_id": library_id,
                    "library_hash": library_hash,
                    "library_index": library_index,
                    "input_name": source_label,
                    "plan_name": plan_name,
                    "position": int(idx),
                    "tf": library_tfs[idx] if idx < len(library_tfs) else "",
                    "tfbs": tfbs,
                    "tfbs_id": library_tfbs_ids[idx] if idx < len(library_tfbs_ids) else None,
                    "motif_id": library_motif_ids[idx] if idx < len(library_motif_ids) else None,
                    "site_id": library_site_ids[idx] if idx < len(library_site_ids) else None,
                    "source": library_sources[idx] if idx < len(library_sources) else None,
                }
            )

    pool_strategy = str(sampling_cfg.pool_strategy)
    library_sampling_strategy = str(sampling_cfg.library_sampling_strategy)
    iterative_max_libraries = int(sampling_cfg.iterative_max_libraries)
    iterative_min_new_solutions = int(sampling_cfg.iterative_min_new_solutions)

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

    policy = RuntimePolicy(
        pool_strategy=pool_strategy,
        arrays_generated_before_resample=max_per_subsample,
        stall_seconds_before_resample=stall_seconds,
        stall_warning_every_seconds=stall_warn_every,
        max_consecutive_failures=max_consecutive_failures,
        max_seconds_per_plan=max_seconds_per_plan,
    )

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

    plan_start = time.monotonic()
    total_resamples = 0
    consecutive_failures = 0
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
    groups = list(constraints.groups or [])
    plan_min_count_by_regulator = dict(constraints.min_count_by_regulator or {})
    metadata_min_counts = {tf: max(min_count_per_tf, int(val)) for tf, val in plan_min_count_by_regulator.items()}
    fixed_elements_dump = _fixed_elements_dump(fixed_elements)
    fixed_elements_max_len = _max_fixed_element_len(fixed_elements_dump)

    # Build initial library
    library_for_opt: List[str]
    tfbs_parts: List[str]
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

    def _select_library_from_artifact() -> tuple[list[str], list[str], list[str], dict]:
        if library_records is None or library_cursor is None:
            raise RuntimeError("Library artifacts requested but no library records were provided.")
        key = (source_label, plan_name)
        records = library_records.get(key) or []
        if not records:
            raise RuntimeError(
                f"No libraries available in artifact for {source_label}/{plan_name}. "
                "Build libraries with `dense stage-b build-libraries` and re-run."
            )
        cursor = int(library_cursor.get(key, 0))
        if cursor >= len(records):
            raise RuntimeError(
                f"Library artifact exhausted for {source_label}/{plan_name} "
                f"(requested index={cursor + 1}, available={len(records)}). "
                "Build more libraries or reduce Stage-B resampling."
            )
        record = records[cursor]
        library_cursor[key] = cursor + 1
        if record.pool_strategy is None or record.library_sampling_strategy is None:
            raise RuntimeError(
                f"Library artifact missing Stage-B sampling metadata for {source_label}/{plan_name} "
                f"(library_index={record.library_index}). Rebuild libraries with the current version."
            )
        if str(record.pool_strategy) != str(pool_strategy):
            raise RuntimeError(
                f"Library artifact pool_strategy mismatch for {source_label}/{plan_name}: "
                f"artifact={record.pool_strategy} config={pool_strategy}."
            )
        if str(record.library_sampling_strategy) != str(library_sampling_strategy):
            raise RuntimeError(
                f"Library artifact Stage-B sampling strategy mismatch for {source_label}/{plan_name}: "
                f"artifact={record.library_sampling_strategy} config={library_sampling_strategy}."
            )
        if pool_strategy != "full" and record.library_size != int(getattr(sampling_cfg, "library_size", 0)):
            raise RuntimeError(
                f"Library artifact size mismatch for {source_label}/{plan_name}: "
                f"artifact={record.library_size} config={sampling_cfg.library_size}."
            )
        _validate_library_constraints(
            record,
            groups=groups,
            min_count_by_regulator=plan_min_count_by_regulator,
            input_name=source_label,
            plan_name=plan_name,
        )
        tfbs_parts_local = []
        for idx, tfbs in enumerate(record.library_tfbs):
            tf = record.library_tfs[idx] if idx < len(record.library_tfs) else ""
            tfbs_parts_local.append(f"{tf}:{tfbs}" if tf else str(tfbs))
        if events_path is not None:
            try:
                _emit_event(
                    events_path,
                    event="LIBRARY_SELECTED",
                    payload={
                        "input_name": source_label,
                        "plan_name": plan_name,
                        "library_index": int(record.library_index),
                        "library_hash": str(record.library_hash),
                        "library_size": int(record.library_size),
                    },
                )
            except Exception:
                log.debug("Failed to emit LIBRARY_SELECTED event.", exc_info=True)
        return record.library_tfbs, tfbs_parts_local, record.library_tfs, record.sampling_info()

    def _build_next_library() -> tuple[list[str], list[str], list[str], dict]:
        nonlocal libraries_built, libraries_used
        if library_source_label == "artifact":
            libraries_used += 1
            libraries_built = libraries_used
            return _select_library_from_artifact()
        library_for_opt_local, tfbs_parts_local, regulator_labels_local, sampling_info_local = build_library_for_plan(
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
            library_index_start=libraries_built,
        )
        libraries_built = int(sampling_info_local.get("library_index", libraries_built))
        libraries_used += 1
        return library_for_opt_local, tfbs_parts_local, regulator_labels_local, sampling_info_local

    if pool_strategy != "iterative_subsample" and not one_subsample_only:
        max_per_subsample = quota
    library_for_opt, tfbs_parts, regulator_labels, sampling_info = _build_next_library()
    if library_source_label != "artifact":
        libraries_built = int(sampling_info.get("library_index", libraries_built))
    site_id_by_index = sampling_info.get("site_id_by_index")
    source_by_index = sampling_info.get("source_by_index")
    tfbs_id_by_index = sampling_info.get("tfbs_id_by_index")
    motif_id_by_index = sampling_info.get("motif_id_by_index")
    sampling_library_index = sampling_info.get("library_index", 0)
    sampling_library_hash = sampling_info.get("library_hash", "")
    library_tfbs = list(library_for_opt)
    library_tfs = list(regulator_labels) if regulator_labels else []
    library_site_ids = list(site_id_by_index) if site_id_by_index else []
    library_sources = list(source_by_index) if source_by_index else []
    library_tfbs_ids = list(tfbs_id_by_index) if tfbs_id_by_index else []
    library_motif_ids = list(motif_id_by_index) if motif_id_by_index else []
    required_regulators = list(dict.fromkeys(sampling_info.get("required_regulators_selected") or []))
    if groups and not required_regulators:
        raise RuntimeError(
            f"Stage-B sampling did not record required_regulators_selected for {source_label}/{plan_name}. "
            "Rebuild libraries with the current version."
        )
    min_required_regulators = None
    min_required_len, min_breakdown, feasibility = assess_library_feasibility(
        library_tfbs=library_tfbs,
        library_tfs=library_tfs,
        fixed_elements=plan_item.fixed_elements,
        groups=groups,
        min_count_by_regulator=plan_min_count_by_regulator,
        min_count_per_tf=min_count_per_tf,
        sequence_length=seq_len,
    )
    fixed_bp = int(feasibility["fixed_bp"])
    min_required_bp = int(feasibility["min_required_bp"])
    slack_bp = int(feasibility["slack_bp"])
    infeasible = bool(feasibility["infeasible"])
    _record_library_build(
        sampling_info=sampling_info,
        library_tfbs=library_tfbs,
        library_tfs=library_tfs,
        library_tfbs_ids=library_tfbs_ids,
        library_motif_ids=library_motif_ids,
        library_site_ids=library_site_ids,
        library_sources=library_sources,
        fixed_bp=fixed_bp,
        min_required_bp=min_required_bp,
        slack_bp=slack_bp,
        infeasible=infeasible,
        sequence_length=seq_len,
    )
    max_tfbs_len = max((len(str(m)) for m in library_tfbs), default=0)
    required_len = max(max_tfbs_len, fixed_elements_max_len)
    if seq_len < required_len:
        raise ValueError(
            "generation.sequence_length is shorter than the widest required motif "
            f"(sequence_length={seq_len}, max_library_motif={max_tfbs_len}, "
            f"max_fixed_element={fixed_elements_max_len}). "
            "Increase densegen.generation.sequence_length or reduce motif lengths "
            "(e.g., adjust Stage-A PWM sampling length.range or fixed-element motifs)."
        )
    if min_required_len > 0 and seq_len < min_required_len:
        raise ValueError(
            "generation.sequence_length is shorter than the minimum required length for constraints "
            f"(sequence_length={seq_len}, min_required_length={min_required_len}, "
            f"fixed_elements_min={min_breakdown['fixed_elements_min']}, "
            f"per_tf_min={min_breakdown['per_tf_min']}, "
            f"min_required_extra={min_breakdown['min_required_extra']}). "
            "Increase densegen.generation.sequence_length or relax regulator_constraints, "
            "min_count_by_regulator, or fixed-element constraints."
        )

    def _current_leaderboard_snapshot() -> dict[str, object]:
        return _leaderboard_snapshot(
            usage_counts,
            tf_usage_counts,
            failure_counts,
            input_name=source_label,
            plan_name=plan_name,
            library_tfs=library_tfs,
            library_tfbs=library_tfbs,
        )

    def _log_leaderboard_snapshot() -> None:
        if progress_style == "screen":
            return
        tf_usage_display = _map_tf_usage(tf_usage_counts)
        tfbs_usage_display = _map_tfbs_usage(usage_counts) if show_tfbs else usage_counts
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
        else:
            log.info(
                "[%s/%s] TFBS usage: %s",
                source_label,
                plan_name,
                _summarize_tfbs_usage_stats(usage_counts),
            )
            failure_totals = _summarize_failure_totals(
                failure_counts,
                input_name=source_label,
                plan_name=plan_name,
            )
            if failure_totals:
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

    def _record_site_failures(reason: str) -> None:
        if not track_failures:
            return
        if not library_tfbs:
            return
        for idx, tfbs in enumerate(library_tfbs):
            tf = library_tfs[idx] if idx < len(library_tfs) else ""
            site_id = None
            if library_site_ids and idx < len(library_site_ids):
                raw = library_site_ids[idx]
                if raw not in (None, "", "None"):
                    site_id = str(raw)
            key = (source_label, plan_name, tf, tfbs, site_id)
            reasons = failure_counts.setdefault(key, {})
            reasons[reason] = reasons.get(reason, 0) + 1

    # Alignment (7): sampling_fraction uses unique TFBS strings and is bounded.
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
    # Library summary (succinct)
    tf_summary = _summarize_tf_counts([_display_tf_label(tf) for tf in regulator_labels] if regulator_labels else [])
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

    solver_min_counts: dict[str, int] | None = None

    def _make_generator(_library_for_opt: List[str], _regulator_labels: List[str]):
        nonlocal solver_min_counts
        regulator_by_index = list(_regulator_labels) if _regulator_labels else None
        base_min_counts = _min_count_by_regulator(regulator_by_index, min_count_per_tf)
        solver_min_counts = _merge_min_counts(base_min_counts, plan_min_count_by_regulator)
        fe_dict = fixed_elements.model_dump() if hasattr(fixed_elements, "model_dump") else fixed_elements
        solver_required_regs = required_regulators or None
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
            min_required_regulators=min_required_regulators,
            solver_time_limit_seconds=solver_time_limit_seconds,
            solver_threads=solver_threads,
        )
        return run

    run = _make_generator(library_for_opt, regulator_labels)
    opt = run.optimizer
    generator = run.generator
    forbid_each = run.forbid_each

    global_generated = already_generated
    produced_total_this_call = 0

    while global_generated < quota:
        if policy.plan_timed_out(now=time.monotonic(), plan_started=plan_start):
            raise RuntimeError(f"[{source_label}/{plan_name}] Exceeded max_seconds_per_plan={max_seconds_per_plan}.")
        local_generated = 0

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
                tf_list_from_library = sorted(set(regulator_labels)) if regulator_labels else []
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
                        _record_site_failures("min_count_per_tf")
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
                        _record_site_failures("required_regulators")
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
                        _record_site_failures("min_count_by_regulator")
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
                gc_core = _gc_fraction(seq)
                gc_total = _gc_fraction(final_seq)
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
                    covers_all_tfs_in_solution=covers_all,
                    required_regulators=required_regulators,
                    min_required_regulators=min_required_regulators,
                    min_count_by_regulator=metadata_min_counts,
                    covers_required_regulators=covers_required,
                    gc_total=gc_total,
                    gc_core=gc_core,
                    input_row_count=input_row_count,
                    input_tf_count=input_tf_count,
                    input_tfbs_count=input_tfbs_count,
                    input_tf_tfbs_pair_count=input_tf_tfbs_pair_count,
                    sampling_fraction=sampling_fraction,
                    sampling_fraction_pairs=sampling_fraction_pairs,
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                    dense_arrays_version=dense_arrays_version,
                    dense_arrays_version_source=dense_arrays_version_source,
                    solver_status=solver_status,
                    solver_objective=solver_objective,
                    solver_solve_time_s=solver_solve_time_s,
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
                    duplicate_records += 1
                    attempt_index = _next_attempt_index()
                    _log_rejection(
                        tables_root,
                        run_id=run_id,
                        input_name=source_label,
                        plan_name=plan_name,
                        attempt_index=attempt_index,
                        reason="output_duplicate",
                        detail={},
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
                    log.info(
                        "[%s/%s] Output duplicate detected; skipping.",
                        source_label,
                        plan_name,
                    )
                    continue

                composition_start = None
                if composition_rows is not None:
                    composition_start = len(composition_rows)
                    for placement_index, entry in enumerate(used_tfbs_detail or []):
                        composition_rows.append(
                            {
                                "solution_id": record.id,
                                "attempt_id": None,
                                "input_name": source_label,
                                "plan_name": plan_name,
                                "library_index": int(sampling_library_index),
                                "library_hash": str(sampling_library_hash),
                                "placement_index": int(placement_index),
                                "tf": entry.get("tf"),
                                "tfbs": entry.get("tfbs"),
                                "motif_id": entry.get("motif_id"),
                                "tfbs_id": entry.get("tfbs_id"),
                                "orientation": entry.get("orientation"),
                                "offset": entry.get("offset"),
                                "length": entry.get("length"),
                                "end": entry.get("end"),
                                "pad_left": entry.get("pad_left"),
                                "site_id": entry.get("site_id"),
                                "source": entry.get("source"),
                            }
                        )

                attempt_index = _next_attempt_index()
                attempt_id = _append_attempt(
                    tables_root,
                    run_id=run_id,
                    input_name=source_label,
                    plan_name=plan_name,
                    attempt_index=attempt_index,
                    status="success",
                    reason="ok",
                    detail={},
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
                    solution_id=record.id,
                    library_tfbs=library_tfbs,
                    library_tfs=library_tfs,
                    library_site_ids=library_site_ids,
                    library_sources=library_sources,
                    attempts_buffer=attempts_buffer,
                )
                if composition_rows is not None and composition_start is not None:
                    for idx in range(composition_start, len(composition_rows)):
                        composition_rows[idx]["attempt_id"] = attempt_id
                if solution_rows is not None:
                    solution_rows.append(
                        SolutionRecord(
                            solution_id=record.id,
                            attempt_id=attempt_id,
                            run_id=str(run_id),
                            input_name=source_label,
                            plan_name=plan_name,
                            created_at=created_at,
                            sequence=final_seq,
                            sequence_hash=hashlib.sha256(final_seq.encode("utf-8")).hexdigest(),
                            sampling_library_index=int(sampling_library_index),
                            sampling_library_hash=str(sampling_library_hash),
                        ).to_dict()
                    )
                    if len(solution_rows) >= SOLUTIONS_CHUNK_SIZE:
                        _flush_solutions(tables_root, solution_rows)

                _update_usage_counts(usage_counts, used_tfbs_detail)
                for tf, count in used_tf_counts.items():
                    tf_usage_counts[tf] = tf_usage_counts.get(tf, 0) + int(count)

                global_generated += 1
                local_generated += 1
                produced_this_library += 1
                produced_total_this_call += 1
                if state_counts is not None:
                    state_counts[(source_label, plan_name)] = int(global_generated)
                    if write_state is not None and checkpoint_every > 0:
                        if global_generated % checkpoint_every == 0:
                            write_state()

                pct = 100.0 * (global_generated / max(1, quota))
                bar = _format_progress_bar(global_generated, quota, width=24)
                cr = getattr(sol, "compression_ratio", float("nan"))
                cr_now = float(cr) if isinstance(cr, (int, float)) and math.isfinite(cr) else None
                if cr_now is not None:
                    cr_sum += cr_now
                    cr_count += 1
                cr_avg = (cr_sum / cr_count) if cr_count else None
                should_log = progress_every > 0 and global_generated % max(1, progress_every) == 0
                if progress_style == "screen":
                    if should_log and dashboard is not None:
                        now = time.monotonic()
                        if (now - last_screen_refresh) >= progress_refresh_seconds:
                            diversity_label = _summarize_diversity(
                                usage_counts,
                                tf_usage_counts,
                                library_tfs=library_tfs,
                                library_tfbs=library_tfbs,
                            )
                            tf_usage_display = _map_tf_usage(tf_usage_counts)
                            tfbs_usage_display = _map_tfbs_usage(usage_counts) if show_tfbs else usage_counts
                            seq_preview = _short_seq(final_seq, max_len=120) if show_solutions else None
                            dashboard.update(
                                _build_screen_dashboard(
                                    source_label=source_label,
                                    plan_name=plan_name,
                                    bar=bar,
                                    generated=global_generated,
                                    quota=quota,
                                    pct=pct,
                                    local_generated=local_generated,
                                    local_target=max_per_subsample,
                                    library_index=int(sampling_library_index),
                                    cr_now=cr_now,
                                    cr_avg=cr_avg,
                                    resamples=total_resamples,
                                    dup_out=duplicate_records,
                                    dup_sol=duplicate_solutions,
                                    fails=failed_solutions,
                                    stalls=stall_events,
                                    failure_totals=latest_failure_totals,
                                    tf_usage=tf_usage_display,
                                    tfbs_usage=tfbs_usage_display,
                                    diversity_label=diversity_label,
                                    show_tfbs=show_tfbs,
                                    show_solutions=show_solutions,
                                    sequence_preview=seq_preview,
                                )
                            )
                            last_screen_refresh = now
                elif progress_style == "summary":
                    pass
                else:
                    if should_log:
                        if show_solutions and print_visual:
                            log.info(
                                "╭─ %s/%s  %s  %d/%d (%.2f%%) — local %d/%d — CR=%.3f\n"
                                "%s\nsequence %s\n"
                                "╰────────────────────────────────────────────────────────",
                                source_label,
                                plan_name,
                                bar,
                                global_generated,
                                quota,
                                pct,
                                local_generated,
                                max_per_subsample,
                                cr_now if cr_now is not None else float("nan"),
                                derived["visual"],
                                final_seq,
                            )
                        elif show_solutions:
                            log.info(
                                "[%s/%s] %s %d/%d (%.2f%%) (local %d/%d) CR=%.3f | seq %s",
                                source_label,
                                plan_name,
                                bar,
                                global_generated,
                                quota,
                                pct,
                                local_generated,
                                max_per_subsample,
                                cr_now if cr_now is not None else float("nan"),
                                final_seq,
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
                            total_resamples,
                            duplicate_records,
                            duplicate_solutions,
                            failed_solutions,
                            stall_events,
                            failure_totals,
                        )
                        tf_usage_display = _map_tf_usage(tf_usage_counts)
                        tfbs_usage_display = _map_tfbs_usage(usage_counts) if show_tfbs else usage_counts
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

            if local_generated >= max_per_subsample or global_generated >= quota:
                break

            if produced_this_library == 0 and not stall_triggered and stall_seconds > 0:
                now = time.monotonic()
                if (now - last_progress) >= stall_seconds:
                    _mark_stall(now)

            if produced_this_library == 0:
                reason = "stall_no_solution" if stall_triggered else "no_solution"
                _record_site_failures(reason)
                attempt_index = _next_attempt_index()
                _append_attempt(
                    tables_root,
                    run_id=run_id,
                    input_name=source_label,
                    plan_name=plan_name,
                    attempt_index=attempt_index,
                    status="failed",
                    reason=reason,
                    detail={"stall_seconds": stall_seconds} if stall_triggered else {},
                    sequence=None,
                    used_tf_counts=None,
                    used_tf_list=[],
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                    solver_status=None,
                    solver_objective=None,
                    solver_solve_time_s=None,
                    dense_arrays_version=dense_arrays_version,
                    dense_arrays_version_source=dense_arrays_version_source,
                    solution_id=None,
                    library_tfbs=library_tfbs,
                    library_tfs=library_tfs,
                    library_site_ids=library_site_ids,
                    library_sources=library_sources,
                    attempts_buffer=attempts_buffer,
                )

            if pool_strategy == "iterative_subsample" and iterative_min_new_solutions > 0:
                if produced_this_library < iterative_min_new_solutions:
                    log.info(
                        "[%s/%s] Library produced %d < iterative_min_new_solutions=%d; Stage-B resampling.",
                        source_label,
                        plan_name,
                        produced_this_library,
                        iterative_min_new_solutions,
                    )

            resample_reason = "resample"
            if produced_this_library == 0:
                resample_reason = "stall_no_solution" if stall_triggered else "no_solution"
            elif pool_strategy == "iterative_subsample" and iterative_min_new_solutions > 0:
                if produced_this_library < iterative_min_new_solutions:
                    resample_reason = "min_new_solutions"

            if produced_this_library == 0:
                consecutive_failures += 1
                if policy.max_consecutive_failures > 0 and consecutive_failures >= policy.max_consecutive_failures:
                    raise RuntimeError(
                        f"[{source_label}/{plan_name}] Exceeded max_consecutive_failures="
                        f"{policy.max_consecutive_failures}."
                    )
            else:
                consecutive_failures = 0

            # Resample
            if not policy.allow_resample():
                raise RuntimeError(
                    f"[{source_label}/{plan_name}] pool_strategy={pool_strategy!r} does not allow Stage-B "
                    "resampling. Reduce quota or use iterative_subsample."
                )
            total_resamples += 1
            if events_path is not None:
                try:
                    _emit_event(
                        events_path,
                        event="RESAMPLE_TRIGGERED",
                        payload={
                            "input_name": source_label,
                            "plan_name": plan_name,
                            "reason": resample_reason,
                            "produced_this_library": int(produced_this_library),
                            "library_index": int(sampling_library_index),
                            "library_hash": str(sampling_library_hash),
                        },
                    )
                except Exception:
                    log.debug("Failed to emit RESAMPLE_TRIGGERED event.", exc_info=True)
            if iterative_max_libraries > 0 and libraries_used >= iterative_max_libraries:
                raise RuntimeError(
                    f"[{source_label}/{plan_name}] Exceeded iterative_max_libraries={iterative_max_libraries}."
                )

            # New library
            library_for_opt, tfbs_parts, regulator_labels, sampling_info = _build_next_library()
            if library_source_label != "artifact":
                libraries_built = int(sampling_info.get("library_index", libraries_built))
            site_id_by_index = sampling_info.get("site_id_by_index")
            source_by_index = sampling_info.get("source_by_index")
            tfbs_id_by_index = sampling_info.get("tfbs_id_by_index")
            motif_id_by_index = sampling_info.get("motif_id_by_index")
            sampling_library_index = sampling_info.get("library_index", sampling_library_index)
            sampling_library_hash = sampling_info.get("library_hash", sampling_library_hash)
            library_tfbs = list(library_for_opt)
            library_tfs = list(regulator_labels) if regulator_labels else []
            library_site_ids = list(site_id_by_index) if site_id_by_index else []
            library_sources = list(source_by_index) if source_by_index else []
            library_tfbs_ids = list(tfbs_id_by_index) if tfbs_id_by_index else []
            library_motif_ids = list(motif_id_by_index) if motif_id_by_index else []
            required_regulators = list(dict.fromkeys(sampling_info.get("required_regulators_selected") or []))
            if groups and not required_regulators:
                raise RuntimeError(
                    f"Stage-B sampling did not record required_regulators_selected for {source_label}/{plan_name}. "
                    "Rebuild libraries with the current version."
                )
            min_required_regulators = None
            min_required_len, min_breakdown, feasibility = assess_library_feasibility(
                library_tfbs=library_tfbs,
                library_tfs=library_tfs,
                fixed_elements=plan_item.fixed_elements,
                groups=groups,
                min_count_by_regulator=plan_min_count_by_regulator,
                min_count_per_tf=min_count_per_tf,
                sequence_length=seq_len,
            )
            fixed_bp = int(feasibility["fixed_bp"])
            min_required_bp = int(feasibility["min_required_bp"])
            slack_bp = int(feasibility["slack_bp"])
            infeasible = bool(feasibility["infeasible"])
            _record_library_build(
                sampling_info=sampling_info,
                library_tfbs=library_tfbs,
                library_tfs=library_tfs,
                library_tfbs_ids=library_tfbs_ids,
                library_motif_ids=library_motif_ids,
                library_site_ids=library_site_ids,
                library_sources=library_sources,
                fixed_bp=fixed_bp,
                min_required_bp=min_required_bp,
                slack_bp=slack_bp,
                infeasible=infeasible,
                sequence_length=seq_len,
            )
            # Alignment (7): sampling_fraction uses unique TFBS strings and is bounded.
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
            tf_summary = _summarize_tf_counts(regulator_labels)
            if progress_style != "screen":
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
            _flush_attempts(tables_root, attempts_buffer)
            if solution_rows is not None:
                _flush_solutions(tables_root, solution_rows)
            if state_counts is not None:
                state_counts[(source_label, plan_name)] = int(global_generated)
                if write_state is not None:
                    write_state()
            snapshot = _current_leaderboard_snapshot()
            if global_generated >= quota and (usage_counts or tf_usage_counts or failure_counts):
                _log_leaderboard_snapshot()
            if dashboard is not None:
                dashboard.close()
            return produced_total_this_call, {
                "generated": produced_total_this_call,
                "duplicates_skipped": duplicate_records,
                "failed_solutions": failed_solutions,
                "total_resamples": total_resamples,
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
    snapshot = _current_leaderboard_snapshot()
    if usage_counts or tf_usage_counts or failure_counts:
        _log_leaderboard_snapshot()
    if dashboard is not None:
        dashboard.close()
    return produced_total_this_call, {
        "generated": produced_total_this_call,
        "duplicates_skipped": duplicate_records,
        "failed_solutions": failed_solutions,
        "total_resamples": total_resamples,
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
    pool_dir = outputs_root / "pools"
    pool_manifest = pool_dir / "pool_manifest.json"
    pool_data: dict[str, PoolData] | None = None

    if build_stage_a:
        if candidate_logging:
            try:
                existed = prepare_candidates_dir(candidates_dir, overwrite=False)
            except Exception as exc:
                raise RuntimeError(f"Failed to prepare candidate artifacts directory: {exc}") from exc
            if existed:
                candidates_label = display_path(candidates_dir, run_root, absolute=False)
                log.info(
                    "Appending candidate artifacts under %s (use dense run --fresh to reset).",
                    candidates_label,
                )
            else:
                candidates_label = display_path(candidates_dir, run_root, absolute=False)
                log.info("Candidate mining artifacts will be written to %s", candidates_label)
        try:
            _pool_artifact, pool_data = build_pool_artifact(
                cfg=cfg,
                cfg_path=loaded.path,
                deps=deps,
                rng=np_rng_stage_a,
                outputs_root=outputs_root,
                out_dir=pool_dir,
                overwrite=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build Stage-A TFBS pools: {exc}") from exc
        try:
            _emit_event(
                events_path,
                event="POOL_BUILT",
                payload={
                    "inputs": [
                        {
                            "name": pool.name,
                            "input_type": pool.input_type,
                            "pool_mode": pool.pool_mode,
                            "rows": int(pool.df.shape[0]) if pool.df is not None else int(len(pool.sequences)),
                        }
                        for pool in pool_data.values()
                    ]
                },
            )
        except Exception:
            log.debug("Failed to emit POOL_BUILT event.", exc_info=True)

    if not pool_manifest.exists():
        raise RuntimeError(
            "Stage-A pools missing or stale. Run `dense stage-a build-pool --fresh` to regenerate pools."
        )
    if not build_stage_a:
        statuses = pool_status_by_input(cfg, loaded.path, run_root)
        stale = [status for status in statuses.values() if status.state != "present"]
        if stale:
            labels = ", ".join(sorted({status.name for status in stale}))
            raise RuntimeError(
                "Stage-A pools missing or stale for: "
                f"{labels}. Run `dense stage-a build-pool --fresh` to regenerate pools."
            )
    try:
        _pool_artifact, pool_data = load_pool_data(pool_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to load existing Stage-A pool artifacts: {exc}") from exc
    pool_label = display_path(pool_dir, run_root, absolute=False)
    log.info("Using Stage-A pools from %s", pool_label)

    if resume and pool_data is None:
        raise RuntimeError(
            "resume=True requires existing Stage-A pools. Run dense stage-a build-pool first or rerun without resume."
        )
    plan_pools = build_plan_pools(plan_items=pl, pool_data=pool_data)
    plan_pool_sources = {plan_name: PlanPoolSource(name=spec.pool_name) for plan_name, spec in plan_pools.items()}
    for spec in plan_pools.values():
        source_cache[spec.pool_name] = spec.pool
    if candidate_logging and build_stage_a:
        candidate_files = find_candidate_files(candidates_dir)
        if candidate_files:
            try:
                build_candidate_artifact(
                    candidates_dir=candidates_dir,
                    cfg_path=loaded.path,
                    run_id=str(cfg.run.id),
                    run_root=run_root,
                    overwrite=True,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to write candidate artifacts: {exc}") from exc
        else:
            candidates_label = display_path(candidates_dir, run_root, absolute=False)
            log.warning(
                "Candidate logging enabled but no candidate records were written under %s. "
                "Check keep_all_candidates_debug and PWM inputs.",
                candidates_label,
            )
    library_records: dict[tuple[str, str], list[LibraryRecord]] | None = None
    library_cursor: dict[tuple[str, str], int] | None = None
    library_artifact: LibraryArtifact | None = None
    sampling_cfg = cfg.generation.sampling
    library_source = str(getattr(sampling_cfg, "library_source", "build")).lower()
    if library_source == "artifact":
        artifact_path = resolve_outputs_scoped_path(
            loaded.path,
            run_root,
            sampling_cfg.library_artifact_path,
            label="sampling.library_artifact_path",
        )
        if not artifact_path.exists():
            artifact_label = display_path(artifact_path, run_root, absolute=False)
            raise RuntimeError(f"Library artifact directory not found: {artifact_label}")
        library_artifact = load_library_artifact(artifact_path)
        library_records = load_library_records(library_artifact)
        library_cursor = {}
        existing_library_by_plan = _load_existing_library_index_by_plan(tables_root)
        for plan_item in pl:
            spec = plan_pools[plan_item.name]
            constraints = plan_item.regulator_constraints
            groups = list(constraints.groups or [])
            plan_min_count_by_regulator = dict(constraints.min_count_by_regulator or {})
            key = (spec.pool_name, plan_item.name)
            records = library_records.get(key)
            if not records:
                raise RuntimeError(
                    f"Library artifact missing libraries for {spec.pool_name}/{plan_item.name}. "
                    "Build libraries with `dense stage-b build-libraries` using this config."
                )
            max_used = existing_library_by_plan.get(key, 0)
            used_count = sum(1 for rec in records if int(rec.library_index) <= int(max_used))
            if max_used and used_count == 0:
                raise RuntimeError(
                    f"Library artifact indices do not cover previously used library_index={max_used} "
                    f"for {spec.pool_name}/{plan_item.name}."
                )
            library_cursor[key] = used_count
            for rec in records:
                if int(rec.library_index) <= 0:
                    raise RuntimeError(
                        f"Library artifact has non-positive library_index={rec.library_index} "
                        f"for {spec.pool_name}/{plan_item.name}."
                    )
                if rec.library_sampling_strategy is None or rec.pool_strategy is None:
                    raise RuntimeError(
                        f"Library artifact missing Stage-B sampling metadata for {spec.pool_name}/{plan_item.name} "
                        f"(library_index={rec.library_index})."
                    )
                _validate_library_constraints(
                    rec,
                    groups=groups,
                    min_count_by_regulator=plan_min_count_by_regulator,
                    input_name=spec.pool_name,
                    plan_name=plan_item.name,
                )
    elif library_source != "build":
        raise RuntimeError(f"Unsupported Stage-B sampling.library_source: {library_source}")
    ensure_run_meta_dir(run_root)
    state_path = run_state_path(run_root)
    state_created_at = datetime.now(timezone.utc).isoformat()
    if state_path.exists():
        try:
            existing_state = load_run_state(state_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read run_state.json: {exc}") from exc
        if existing_state.run_id and existing_state.run_id != cfg.run.id:
            raise RuntimeError(
                "Existing run_state.json was created with a different run_id. "
                "Remove run_state.json or stage a new run root to start fresh."
            )
        if existing_state.config_sha256 and existing_state.config_sha256 != config_sha:
            raise RuntimeError(
                "Existing run_state.json was created with a different config. "
                "Remove run_state.json or stage a new run root to start fresh."
            )
        if existing_state.created_at:
            state_created_at = existing_state.created_at

    existing_counts: dict[tuple[str, str], int] = {}
    existing_usage_by_plan: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    site_failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] = {}
    attempt_counters: dict[tuple[str, str], int] = {}
    if resume:
        site_failure_counts = _load_failure_counts_from_attempts(tables_root)
        attempt_counters = _load_existing_attempt_index_by_plan(tables_root)
        if cfg.output.targets:
            try:
                df_existing, _ = load_records_from_config(
                    loaded.root,
                    loaded.path,
                    columns=[
                        "densegen__run_config_sha256",
                        "densegen__run_id",
                        "densegen__input_name",
                        "densegen__plan",
                        "densegen__used_tfbs_detail",
                    ],
                )
            except Exception:
                df_existing = None
            if df_existing is not None and not df_existing.empty:
                if "densegen__run_config_sha256" in df_existing.columns:
                    mismatched = df_existing["densegen__run_config_sha256"].dropna().unique().tolist()
                    if mismatched and any(val != config_sha for val in mismatched):
                        raise RuntimeError(
                            "Existing outputs were produced with a different config. "
                            "Remove outputs/tables (and outputs/meta if present) "
                            "or stage a new run root to start fresh."
                        )
                if "densegen__run_id" in df_existing.columns:
                    run_ids = df_existing["densegen__run_id"].dropna().unique().tolist()
                    if run_ids and any(val != cfg.run.id for val in run_ids):
                        raise RuntimeError(
                            "Existing outputs were produced with a different run_id. "
                            "Remove outputs/tables (and outputs/meta if present) "
                            "or stage a new run root to start fresh."
                        )
                if {"densegen__input_name", "densegen__plan"} <= set(df_existing.columns):
                    counts = (
                        df_existing.groupby(["densegen__input_name", "densegen__plan"]).size().astype(int).to_dict()
                    )
                    existing_counts = {(str(k[0]), str(k[1])): int(v) for k, v in counts.items()}
                if "densegen__used_tfbs_detail" in df_existing.columns:
                    for _, row in df_existing.iterrows():
                        input_name = str(row.get("densegen__input_name") or "")
                        plan_name = str(row.get("densegen__plan") or "")
                        if not input_name or not plan_name:
                            continue
                        key = (input_name, plan_name)
                        counts = existing_usage_by_plan.setdefault(key, {})
                        used = _parse_used_tfbs_detail(row.get("densegen__used_tfbs_detail"))
                        _update_usage_counts(counts, used)
                if existing_counts:
                    total = sum(existing_counts.values())
                    per_plan = dict(existing_counts)
                    log.info(
                        "Resuming from existing outputs: %d sequences across %d plan(s).",
                        total,
                        len(existing_counts),
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
    inputs = cfg.inputs
    inputs_by_name = {inp.name: inp for inp in inputs}
    display_map_by_input: dict[str, dict[str, str]] = {}
    for item in pl:
        spec = plan_pools[item.name]
        mapping: dict[str, str] = {}
        for input_name in spec.include_inputs:
            inp = inputs_by_name.get(input_name)
            if inp is None:
                continue
            for motif_id, name in input_motifs(inp, loaded.path):
                if motif_id and name and motif_id not in mapping:
                    mapping[motif_id] = name
        if mapping:
            display_map_by_input[spec.pool_name] = mapping
    checkpoint_every = int(cfg.runtime.checkpoint_every)
    state_counts: dict[tuple[str, str], int] = {}
    for item in pl:
        spec = plan_pools[item.name]
        state_counts[(spec.pool_name, item.name)] = int(existing_counts.get((spec.pool_name, item.name), 0))

    if state_path.exists() and not existing_counts:
        # run_state exists but no outputs; avoid accidental double-counting.
        existing_state = load_run_state(state_path)
        if existing_state.items and sum(item.generated for item in existing_state.items) > 0:
            raise RuntimeError(
                "run_state.json indicates prior progress, but no outputs were found. "
                "Restore outputs or delete run_state.json before resuming."
            )

    def _write_state() -> None:
        _write_run_state(
            state_path,
            run_id=cfg.run.id,
            schema_version=str(cfg.schema_version),
            config_sha256=config_sha,
            run_root=str(run_root),
            counts=state_counts,
            created_at=state_created_at,
        )

    _write_state()
    # Seed plan stats with any existing outputs to keep manifests aligned on resume.
    for item in pl:
        spec = plan_pools[item.name]
        key = (spec.pool_name, item.name)
        if key not in plan_stats:
            plan_stats[key] = {
                "generated": int(existing_counts.get(key, 0)),
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

    if not round_robin:
        for item in pl:
            spec = plan_pools[item.name]
            source_cfg = plan_pool_sources[item.name]
            produced, stats = _process_plan_for_source(
                source_cfg,
                item,
                cfg,
                sinks,
                chosen_solver=chosen_solver,
                deps=deps,
                rng=rng,
                np_rng=np_rng_stage_b,
                cfg_path=loaded.path,
                run_id=cfg.run.id,
                run_root=run_root_str,
                run_config_path=run_cfg_path,
                run_config_sha256=config_sha,
                random_seed=seed,
                dense_arrays_version=dense_arrays_version,
                dense_arrays_version_source=dense_arrays_version_source,
                show_tfbs=show_tfbs,
                show_solutions=show_solutions,
                output_bio_type=output_bio_type,
                output_alphabet=output_alphabet,
                one_subsample_only=False,
                already_generated=int(existing_counts.get((spec.pool_name, item.name), 0)),
                inputs_manifest=inputs_manifest_entries,
                existing_usage_counts=existing_usage_by_plan.get((spec.pool_name, item.name)),
                state_counts=state_counts,
                checkpoint_every=checkpoint_every,
                write_state=_write_state,
                site_failure_counts=site_failure_counts,
                source_cache=source_cache,
                pool_override=spec.pool,
                input_meta_override=_plan_pool_input_meta(spec),
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
            per_plan[(spec.pool_name, item.name)] = per_plan.get((spec.pool_name, item.name), 0) + produced
            total += produced
            leaderboard_latest = stats.get("leaderboard_latest")
            if leaderboard_latest is not None:
                plan_leaderboards[(spec.pool_name, item.name)] = leaderboard_latest
            _accumulate_stats((spec.pool_name, item.name), stats)
    else:
        produced_counts: dict[tuple[str, str], int] = dict(existing_counts)
        done = False
        while not done:
            done = True
            for item in pl:
                spec = plan_pools[item.name]
                key = (spec.pool_name, item.name)
                current = produced_counts.get(key, 0)
                quota = int(item.quota)
                if current >= quota:
                    continue
                done = False
                source_cfg = plan_pool_sources[item.name]
                produced, stats = _process_plan_for_source(
                    source_cfg,
                    item,
                    cfg,
                    sinks,
                    chosen_solver=chosen_solver,
                    deps=deps,
                    rng=rng,
                    np_rng=np_rng_stage_b,
                    cfg_path=loaded.path,
                    run_id=cfg.run.id,
                    run_root=run_root_str,
                    run_config_path=run_cfg_path,
                    run_config_sha256=config_sha,
                    random_seed=seed,
                    dense_arrays_version=dense_arrays_version,
                    dense_arrays_version_source=dense_arrays_version_source,
                    show_tfbs=show_tfbs,
                    show_solutions=show_solutions,
                    output_bio_type=output_bio_type,
                    output_alphabet=output_alphabet,
                    one_subsample_only=True,
                    already_generated=current,
                    inputs_manifest=inputs_manifest_entries,
                    existing_usage_counts=existing_usage_by_plan.get((spec.pool_name, item.name)),
                    state_counts=state_counts,
                    checkpoint_every=checkpoint_every,
                    write_state=_write_state,
                    site_failure_counts=site_failure_counts,
                    source_cache=source_cache,
                    pool_override=spec.pool,
                    input_meta_override=_plan_pool_input_meta(spec),
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
                produced_counts[key] = current + produced
                leaderboard_latest = stats.get("leaderboard_latest")
                if leaderboard_latest is not None:
                    plan_leaderboards[key] = leaderboard_latest
                _accumulate_stats(key, stats)
        per_plan = produced_counts
        total = sum(per_plan.values())

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

    libraries_dir = outputs_root / "libraries"
    if library_source == "artifact":
        if library_artifact is None:
            raise RuntimeError("Stage-B sampling.library_source=artifact but no library artifact was loaded.")
        try:
            build_rows = pd.read_parquet(library_artifact.builds_path).to_dict("records")
            member_rows = pd.read_parquet(library_artifact.members_path).to_dict("records")
            write_library_artifact(
                out_dir=libraries_dir,
                builds=build_rows,
                members=member_rows,
                cfg_path=loaded.path,
                run_id=str(cfg.run.id),
                run_root=run_root,
                overwrite=True,
                config_hash=config_sha,
                pool_manifest_hash=pool_manifest_hash,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to write library artifacts: {exc}") from exc
    elif library_build_rows:
        existing_builds: list[dict] = []
        existing_members: list[dict] = []
        builds_path = libraries_dir / "library_builds.parquet"
        members_path = libraries_dir / "library_members.parquet"
        if builds_path.exists():
            try:
                existing_builds = pd.read_parquet(builds_path).to_dict("records")
            except Exception:
                log.warning("Failed to read existing library_builds.parquet; overwriting.", exc_info=True)
                existing_builds = []
        if members_path.exists():
            try:
                existing_members = pd.read_parquet(members_path).to_dict("records")
            except Exception:
                log.warning("Failed to read existing library_members.parquet; overwriting.", exc_info=True)
                existing_members = []

        existing_indices = {
            int(row.get("library_index") or 0) for row in existing_builds if row.get("library_index") is not None
        }
        new_builds = [row for row in library_build_rows if int(row.get("library_index") or 0) not in existing_indices]
        build_rows = existing_builds + new_builds

        existing_member_keys = {
            (
                int(row.get("library_index") or 0),
                int(row.get("position") or 0),
            )
            for row in existing_members
        }
        new_members = [
            row
            for row in library_member_rows
            if (int(row.get("library_index") or 0), int(row.get("position") or 0)) not in existing_member_keys
        ]
        member_rows = existing_members + new_members

        try:
            write_library_artifact(
                out_dir=libraries_dir,
                builds=build_rows,
                members=member_rows,
                cfg_path=loaded.path,
                run_id=str(cfg.run.id),
                run_root=run_root,
                overwrite=True,
                config_hash=config_sha,
                pool_manifest_hash=pool_manifest_hash,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to write library artifacts: {exc}") from exc

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
