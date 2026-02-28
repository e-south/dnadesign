"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_runtime_callbacks.py

Stage-B sampler callback execution with explicit runtime state objects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .progress import _summarize_tf_counts
from .stage_b import (
    _compute_sampling_fraction,
    _compute_sampling_fraction_pairs,
    _merge_min_counts,
    _min_count_by_regulator,
)
from .stage_b_library_builder import LibraryContext
from .stage_b_runtime_checks import (
    _evaluate_sequence_constraints,
    _evaluate_solution_requirements,
    _extract_solver_metrics,
    _maybe_pad_sequence,
)
from .stage_b_sampler import LibraryRunResult
from .stage_b_solution_persistence import persist_candidate_solution
from .stage_b_solution_rejections import (
    handle_duplicate_sequence,
    mark_stall_detected,
    reject_sequence_validation_failure,
    reject_solution_requirement_failure,
)
from .usage_tracking import _compute_used_tf_info


@dataclass
class StageBLibraryRuntimeState:
    libraries_built: int
    libraries_built_start: int
    libraries_used: int = 0
    last_accepted_progress: float | None = None
    failed_solutions: int = 0
    duplicate_records: int = 0
    stall_events: int = 0
    failed_min_count_per_tf: int = 0
    failed_required_regulators: int = 0
    failed_min_count_by_regulator: int = 0
    failed_min_required_regulators: int = 0
    duplicate_solutions: int = 0
    last_no_solution_reason: str | None = None
    last_no_solution_solver_status: str | None = None
    last_no_solution_solver_objective: float | None = None
    last_no_solution_solver_solve_time_s: float | None = None
    last_no_solution_detail: dict | None = None


def build_next_library_callback(
    *,
    builder,
    state: StageBLibraryRuntimeState,
    library_source_label: str,
) -> LibraryContext:
    library_context = builder.build_next(library_index_start=state.libraries_built)
    state.libraries_used += 1
    if str(library_source_label) == "artifact":
        state.libraries_built = int(state.libraries_used)
    else:
        state.libraries_built = int(library_context.sampling_info.get("library_index", state.libraries_built))
    return library_context


@dataclass(frozen=True)
class StageBLibraryRuntimeContext:
    source_label: str
    plan_name: str
    seq_len: int
    pool_strategy: str
    min_count_per_tf: int
    max_dupes: int
    stall_seconds: int
    max_failed_solutions: int
    progress_style: str
    show_solutions: bool
    pad_enabled: bool
    pad_end: str
    pad_mode: str
    pad_gc_mode: str
    pad_gc_min: float
    pad_gc_max: float
    pad_gc_target: float
    pad_gc_tolerance: float
    pad_gc_min_length: int
    pad_max_tries: int
    solver_strategy: str
    solver_strands: str
    solver_attempt_timeout_seconds: float | None
    solver_threads: int | None
    extra_library_label: str | None
    fixed_elements: Any
    fixed_elements_dump: dict
    plan_min_count_by_regulator: dict[str, int]
    input_meta: dict
    input_tfbs_count: int
    input_tf_tfbs_pair_count: int | None
    sequence_constraint_patterns: list[str]
    compiled_sequence_constraints: Any
    library_sampling_strategy: str
    policy: Any
    deps: Any
    rng: Any
    diagnostics: Any
    display_tf_label: Callable[[str], str]
    rejection_context: Any
    solution_output_context: Any
    progress_context: Any
    events_path: Path | None
    tables_root: Path
    run_id: str
    dense_arrays_version: str | None
    dense_arrays_version_source: str | None
    next_attempt_index: Callable[[], int]
    emit_event: Callable[..., None]
    append_attempt: Callable[..., None]
    apply_pad_offsets: Callable[[list[dict], dict], list[dict]]
    logger: Any


class StageBLibraryRuntimeCallbacks:
    def __init__(
        self,
        *,
        builder,
        library_source_label: str,
        context: StageBLibraryRuntimeContext,
        state: StageBLibraryRuntimeState,
    ) -> None:
        self._builder = builder
        self._library_source_label = str(library_source_label)
        self._context = context
        self._state = state

    @property
    def state(self) -> StageBLibraryRuntimeState:
        return self._state

    def build_next_library(self) -> LibraryContext:
        return build_next_library_callback(
            builder=self._builder,
            state=self._state,
            library_source_label=self._library_source_label,
        )

    def _make_generator(
        self,
        library_for_opt: list[str],
        regulator_labels: list[str],
        *,
        required_regulators_local: list[str],
        min_required_regulators_local: int | None,
    ):
        regulator_by_index = list(regulator_labels) if regulator_labels else None
        base_min_counts = _min_count_by_regulator(regulator_by_index, self._context.min_count_per_tf)
        solver_min_counts = _merge_min_counts(base_min_counts, self._context.plan_min_count_by_regulator)
        fe_dict = (
            self._context.fixed_elements.model_dump()
            if hasattr(self._context.fixed_elements, "model_dump")
            else self._context.fixed_elements
        )
        solver_required_regs = required_regulators_local or None
        return self._context.deps.optimizer.build(
            library=library_for_opt,
            sequence_length=self._context.seq_len,
            solver=self._context.solution_output_context.chosen_solver,
            strategy=self._context.solver_strategy,
            fixed_elements=fe_dict,
            strands=self._context.solver_strands,
            regulator_by_index=regulator_by_index,
            required_regulators=solver_required_regs,
            min_count_by_regulator=solver_min_counts,
            min_required_regulators=min_required_regulators_local,
            solver_attempt_timeout_seconds=self._context.solver_attempt_timeout_seconds,
            solver_threads=self._context.solver_threads,
            extra_label=self._context.extra_library_label,
        )

    def run_library(
        self,
        library_context: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
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
        stage_a_best_hit_score_by_index = sampling_info.get("stage_a_best_hit_score_by_index")
        stage_a_rank_within_regulator_by_index = sampling_info.get("stage_a_rank_within_regulator_by_index")
        stage_a_tier_by_index = sampling_info.get("stage_a_tier_by_index")
        stage_a_fimo_start_by_index = sampling_info.get("stage_a_fimo_start_by_index")
        stage_a_fimo_stop_by_index = sampling_info.get("stage_a_fimo_stop_by_index")
        stage_a_fimo_strand_by_index = sampling_info.get("stage_a_fimo_strand_by_index")
        stage_a_selection_rank_by_index = sampling_info.get("stage_a_selection_rank_by_index")
        stage_a_selection_score_norm_by_index = sampling_info.get("stage_a_selection_score_norm_by_index")
        stage_a_tfbs_core_by_index = sampling_info.get("stage_a_tfbs_core_by_index")
        stage_a_score_theoretical_max_by_index = sampling_info.get("stage_a_score_theoretical_max_by_index")
        stage_a_selection_policy_by_index = sampling_info.get("stage_a_selection_policy_by_index")
        stage_a_nearest_selected_similarity_by_index = sampling_info.get("stage_a_nearest_selected_similarity_by_index")
        stage_a_nearest_selected_distance_by_index = sampling_info.get("stage_a_nearest_selected_distance_by_index")
        stage_a_nearest_selected_distance_norm_by_index = sampling_info.get(
            "stage_a_nearest_selected_distance_norm_by_index"
        )

        self._context.diagnostics.update_library(
            library_tfs=library_tfs,
            library_tfbs=library_tfbs,
            library_site_ids=library_site_ids,
        )

        sampling_fraction = _compute_sampling_fraction(
            library_for_opt,
            input_tfbs_count=self._context.input_tfbs_count,
            pool_strategy=self._context.pool_strategy,
        )
        self._context.input_meta["sampling_fraction"] = sampling_fraction
        sampling_fraction_pairs = _compute_sampling_fraction_pairs(
            library_for_opt,
            regulator_labels,
            input_pair_count=self._context.input_tf_tfbs_pair_count,
            pool_strategy=self._context.pool_strategy,
        )
        self._context.input_meta["sampling_fraction_pairs"] = sampling_fraction_pairs
        tf_summary = _summarize_tf_counts(
            [self._context.display_tf_label(tf) for tf in regulator_labels] if regulator_labels else []
        )
        library_index = sampling_info.get("library_index")
        strategy_label = sampling_info.get("library_sampling_strategy", self._context.library_sampling_strategy)
        pool_label = sampling_info.get("pool_strategy")
        achieved_len = sampling_info.get("achieved_length")
        header = f"Stage-B library for {self._context.source_label}/{self._context.plan_name}"
        if library_index is not None:
            header = f"{header} (build {library_index})"
        if self._context.progress_style == "stream":
            if tf_summary:
                self._context.logger.info(
                    "%s: %d motifs | TF counts: %s | library_bp=%s pool=%s stage_b_sampling=%s",
                    header,
                    len(library_for_opt),
                    tf_summary,
                    achieved_len,
                    pool_label,
                    strategy_label,
                )
            else:
                self._context.logger.info(
                    "%s: %d motifs | library_bp=%s pool=%s stage_b_sampling=%s",
                    header,
                    len(library_for_opt),
                    achieved_len,
                    pool_label,
                    strategy_label,
                )

        run = self._make_generator(
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
        last_accepted_progress = self._state.last_accepted_progress
        self._state.last_no_solution_reason = None
        self._state.last_no_solution_solver_status = None
        self._state.last_no_solution_solver_objective = None
        self._state.last_no_solution_solver_solve_time_s = None
        self._state.last_no_solution_detail = None
        subsample_started = time.monotonic()

        if local_generated < max_per_subsample and global_generated < quota:
            fingerprints = set()
            consecutive_dup = 0
            last_log_warn = subsample_started
            if last_accepted_progress is None:
                last_accepted_progress = subsample_started
                self._state.last_accepted_progress = last_accepted_progress
            produced_this_library = 0
            stall_triggered = False

            def _check_stall_after_candidate(now: float) -> bool:
                nonlocal stall_triggered
                if produced_this_library > 0:
                    return False
                if not self._context.policy.should_trigger_stall(
                    now=now,
                    last_progress=float(last_accepted_progress),
                ):
                    return False
                self._state.stall_events, stall_triggered = mark_stall_detected(
                    events_path=self._context.events_path,
                    source_label=self._context.source_label,
                    plan_name=self._context.plan_name,
                    stall_seconds=self._context.stall_seconds,
                    last_progress=float(last_accepted_progress),
                    now=now,
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                    stall_events=self._state.stall_events,
                    stall_triggered=stall_triggered,
                    emit_event=self._context.emit_event,
                    logger=self._context.logger,
                )
                return True

            for sol in generator:
                now = time.monotonic()
                if self._context.policy.should_warn_stall(
                    now=now,
                    last_warn=last_log_warn,
                    last_progress=float(last_accepted_progress),
                ):
                    self._context.logger.info(
                        "[%s/%s] Still working... %.1fs on current library.",
                        self._context.source_label,
                        self._context.plan_name,
                        now - subsample_started,
                    )
                    last_log_warn = now

                if forbid_each:
                    opt.forbid(sol)
                seq = sol.sequence
                should_continue, should_break, self._state.duplicate_solutions, consecutive_dup = (
                    handle_duplicate_sequence(
                        sequence=seq,
                        fingerprints=fingerprints,
                        duplicate_solutions=self._state.duplicate_solutions,
                        consecutive_dup=consecutive_dup,
                        max_dupes=self._context.max_dupes,
                        source_label=self._context.source_label,
                        plan_name=self._context.plan_name,
                        logger=self._context.logger,
                    )
                )
                if should_break:
                    break
                if should_continue:
                    if _check_stall_after_candidate(now):
                        break
                    continue

                used_tfbs, used_tfbs_detail, used_tf_counts, used_tf_list = _compute_used_tf_info(
                    sol,
                    library_for_opt,
                    regulator_labels,
                    self._context.fixed_elements,
                    site_id_by_index,
                    source_by_index,
                    tfbs_id_by_index,
                    motif_id_by_index,
                    stage_a_best_hit_score_by_index,
                    stage_a_rank_within_regulator_by_index,
                    stage_a_tier_by_index,
                    stage_a_fimo_start_by_index,
                    stage_a_fimo_stop_by_index,
                    stage_a_fimo_strand_by_index,
                    stage_a_selection_rank_by_index,
                    stage_a_selection_score_norm_by_index,
                    stage_a_tfbs_core_by_index,
                    stage_a_score_theoretical_max_by_index,
                    stage_a_selection_policy_by_index,
                    stage_a_nearest_selected_similarity_by_index,
                    stage_a_nearest_selected_distance_by_index,
                    stage_a_nearest_selected_distance_norm_by_index,
                )
                solver_status, solver_objective, solver_solve_time_s = _extract_solver_metrics(sol)

                covers_all, covers_required, rejection_reason, rejection_detail = _evaluate_solution_requirements(
                    min_count_per_tf=self._context.min_count_per_tf,
                    tf_list_from_library=tf_list_from_library,
                    required_regulators=required_regulators,
                    plan_min_count_by_regulator=self._context.plan_min_count_by_regulator,
                    used_tf_counts=used_tf_counts,
                )
                if rejection_reason is not None:
                    (
                        self._state.failed_solutions,
                        self._state.failed_min_count_per_tf,
                        self._state.failed_required_regulators,
                        self._state.failed_min_count_by_regulator,
                    ) = reject_solution_requirement_failure(
                        rejection_context=self._context.rejection_context,
                        rejection_reason=rejection_reason,
                        rejection_detail=rejection_detail,
                        sequence=seq,
                        used_tf_counts=used_tf_counts,
                        used_tf_list=used_tf_list,
                        sampling_library_index=int(sampling_library_index),
                        sampling_library_hash=str(sampling_library_hash),
                        solver_status=solver_status,
                        solver_objective=solver_objective,
                        solver_solve_time_s=solver_solve_time_s,
                        library_tfbs=library_tfbs,
                        library_tfs=library_tfs,
                        library_site_ids=library_site_ids,
                        library_sources=library_sources,
                        failed_solutions=self._state.failed_solutions,
                        max_failed_solutions=self._context.max_failed_solutions,
                        source_label=self._context.source_label,
                        plan_name=self._context.plan_name,
                        failed_min_count_per_tf=self._state.failed_min_count_per_tf,
                        failed_required_regulators=self._state.failed_required_regulators,
                        failed_min_count_by_regulator=self._state.failed_min_count_by_regulator,
                        diagnostics=self._context.diagnostics,
                    )
                    if _check_stall_after_candidate(now):
                        break
                    continue

                final_seq, pad_meta = _maybe_pad_sequence(
                    sequence=seq,
                    seq_len=self._context.seq_len,
                    source_label=self._context.source_label,
                    plan_name=self._context.plan_name,
                    pad_enabled=self._context.pad_enabled,
                    pad_end=self._context.pad_end,
                    pad_mode=self._context.pad_mode,
                    pad_gc_mode=self._context.pad_gc_mode,
                    pad_gc_min=self._context.pad_gc_min,
                    pad_gc_max=self._context.pad_gc_max,
                    pad_gc_target=self._context.pad_gc_target,
                    pad_gc_tolerance=self._context.pad_gc_tolerance,
                    pad_gc_min_length=self._context.pad_gc_min_length,
                    pad_max_tries=self._context.pad_max_tries,
                    sequence_constraint_patterns=self._context.sequence_constraint_patterns,
                    pad_builder=self._context.deps.pad,
                    rng=self._context.rng,
                )

                sequence_constraints_eval = _evaluate_sequence_constraints(
                    final_seq=final_seq,
                    compiled_sequence_constraints=self._context.compiled_sequence_constraints,
                    fixed_elements_dump=self._context.fixed_elements_dump,
                    source_label=self._context.source_label,
                    plan_name=self._context.plan_name,
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                )
                promoter_detail = dict(sequence_constraints_eval.promoter_detail)
                sequence_validation = dict(sequence_constraints_eval.sequence_validation)
                if sequence_constraints_eval.rejection_detail is not None:
                    self._state.failed_solutions = reject_sequence_validation_failure(
                        rejection_context=self._context.rejection_context,
                        rejection_detail=dict(sequence_constraints_eval.rejection_detail),
                        rejection_event_payload=sequence_constraints_eval.rejection_event_payload,
                        validation_error=sequence_constraints_eval.error,
                        final_seq=final_seq,
                        used_tf_counts=used_tf_counts,
                        used_tf_list=used_tf_list,
                        sampling_library_index=int(sampling_library_index),
                        sampling_library_hash=str(sampling_library_hash),
                        solver_status=solver_status,
                        solver_objective=solver_objective,
                        solver_solve_time_s=solver_solve_time_s,
                        library_tfbs=library_tfbs,
                        library_tfs=library_tfs,
                        library_site_ids=library_site_ids,
                        library_sources=library_sources,
                        failed_solutions=self._state.failed_solutions,
                        max_failed_solutions=self._context.max_failed_solutions,
                        source_label=self._context.source_label,
                        plan_name=self._context.plan_name,
                        events_path=self._context.events_path,
                        emit_event=self._context.emit_event,
                        logger=self._context.logger,
                    )
                    if _check_stall_after_candidate(now):
                        break
                    continue

                global_generated, local_generated, produced_this_library, self._state.duplicate_records, accepted = (
                    persist_candidate_solution(
                        solution_output_context=self._context.solution_output_context,
                        progress_context=self._context.progress_context,
                        sol=sol,
                        seq=seq,
                        final_seq=final_seq,
                        used_tfbs=used_tfbs,
                        used_tfbs_detail=used_tfbs_detail,
                        used_tf_counts=used_tf_counts,
                        used_tf_list=used_tf_list,
                        pad_meta=pad_meta,
                        covers_all=covers_all,
                        covers_required=covers_required,
                        tfbs_parts=tfbs_parts,
                        regulator_labels=regulator_labels,
                        library_for_opt=library_for_opt,
                        sampling_info=sampling_info,
                        required_regulators=required_regulators,
                        min_required_regulators=min_required_regulators,
                        sampling_fraction=sampling_fraction,
                        sampling_fraction_pairs=sampling_fraction_pairs,
                        sampling_library_index=int(sampling_library_index),
                        sampling_library_hash=str(sampling_library_hash),
                        library_tfbs=library_tfbs,
                        library_tfs=library_tfs,
                        library_site_ids=library_site_ids,
                        library_sources=library_sources,
                        promoter_detail=promoter_detail,
                        sequence_validation=sequence_validation,
                        solver_status=solver_status,
                        solver_objective=solver_objective,
                        solver_solve_time_s=solver_solve_time_s,
                        apply_pad_offsets=self._context.apply_pad_offsets,
                        global_generated=global_generated,
                        local_generated=local_generated,
                        produced_this_library=produced_this_library,
                        duplicate_records=self._state.duplicate_records,
                        duplicate_solutions=self._state.duplicate_solutions,
                        failed_solutions=self._state.failed_solutions,
                        stall_events=self._state.stall_events,
                    )
                )
                if not accepted:
                    if _check_stall_after_candidate(now):
                        break
                    continue
                last_accepted_progress = now
                self._state.last_accepted_progress = last_accepted_progress

                if local_generated >= max_per_subsample or global_generated >= quota:
                    break

        if (
            produced_this_library == 0
            and not stall_triggered
            and self._context.stall_seconds > 0
            and last_accepted_progress is not None
        ):
            now = time.monotonic()
            if (now - float(last_accepted_progress)) >= self._context.stall_seconds:
                self._state.stall_events, stall_triggered = mark_stall_detected(
                    events_path=self._context.events_path,
                    source_label=self._context.source_label,
                    plan_name=self._context.plan_name,
                    stall_seconds=self._context.stall_seconds,
                    last_progress=float(last_accepted_progress),
                    now=now,
                    sampling_library_index=int(sampling_library_index),
                    sampling_library_hash=str(sampling_library_hash),
                    stall_events=self._state.stall_events,
                    stall_triggered=stall_triggered,
                    emit_event=self._context.emit_event,
                    logger=self._context.logger,
                )

        library_elapsed = max(0.0, float(time.monotonic() - subsample_started))

        if produced_this_library == 0:
            no_solution_reason = "stall_no_solution" if stall_triggered else "no_solution"
            no_solution_detail = {
                "solver_status": no_solution_reason,
                "solver_solve_time_s": library_elapsed,
                "library_index": int(sampling_library_index),
                "library_hash": str(sampling_library_hash),
                "library_infeasible": bool(library_context.infeasible),
                "library_slack_bp": int(library_context.slack_bp),
                "library_min_required_len": int(library_context.min_required_len),
            }
            if stall_triggered:
                no_solution_detail["stall_seconds"] = int(self._context.stall_seconds)
            self._state.last_no_solution_reason = no_solution_reason
            self._state.last_no_solution_solver_status = no_solution_reason
            self._state.last_no_solution_solver_objective = None
            self._state.last_no_solution_solver_solve_time_s = library_elapsed
            self._state.last_no_solution_detail = no_solution_detail
        else:
            self._state.last_no_solution_reason = None
            self._state.last_no_solution_solver_status = None
            self._state.last_no_solution_solver_objective = None
            self._state.last_no_solution_solver_solve_time_s = None
            self._state.last_no_solution_detail = None

        return LibraryRunResult(
            produced=produced_this_library,
            stall_triggered=stall_triggered,
            global_generated=int(global_generated),
            active_runtime_seconds=library_elapsed,
        )

    def on_no_solution(self, library_context: LibraryContext, reason: str) -> None:
        attempt_index = self._context.next_attempt_index()
        solver_status = str(self._state.last_no_solution_solver_status or reason)
        solver_objective = self._state.last_no_solution_solver_objective
        solver_solve_time_s = self._state.last_no_solution_solver_solve_time_s
        detail = dict(self._state.last_no_solution_detail or {})
        if reason == "stall_no_solution":
            detail.setdefault("stall_seconds", self._context.stall_seconds)
        detail.setdefault("solver_status", solver_status)
        if solver_solve_time_s is not None:
            detail.setdefault("solver_solve_time_s", float(solver_solve_time_s))
        self._context.append_attempt(
            self._context.tables_root,
            run_id=self._context.run_id,
            input_name=self._context.source_label,
            plan_name=self._context.plan_name,
            attempt_index=attempt_index,
            status="failed",
            reason=reason,
            detail=detail,
            sequence=None,
            used_tf_counts=None,
            used_tf_list=[],
            sampling_library_index=int(library_context.sampling_library_index),
            sampling_library_hash=str(library_context.sampling_library_hash),
            solver_status=solver_status,
            solver_objective=solver_objective,
            solver_solve_time_s=solver_solve_time_s,
            dense_arrays_version=self._context.dense_arrays_version,
            dense_arrays_version_source=self._context.dense_arrays_version_source,
            library_tfbs=list(library_context.library_tfbs),
            library_tfs=list(library_context.library_tfs),
            library_site_ids=list(library_context.library_site_ids),
            library_sources=list(library_context.library_sources),
            attempts_buffer=self._context.rejection_context.attempts_buffer,
        )
        self._state.last_no_solution_reason = None
        self._state.last_no_solution_solver_status = None
        self._state.last_no_solution_solver_objective = None
        self._state.last_no_solution_solver_solve_time_s = None
        self._state.last_no_solution_detail = None

    def on_resample(self, library_context: LibraryContext, reason: str, produced_this_library: int) -> None:
        if self._context.events_path is None:
            return
        try:
            self._context.emit_event(
                self._context.events_path,
                event="RESAMPLE_TRIGGERED",
                payload={
                    "input_name": self._context.source_label,
                    "plan_name": self._context.plan_name,
                    "reason": reason,
                    "produced_this_library": int(produced_this_library),
                    "library_index": int(library_context.sampling_library_index),
                    "library_hash": str(library_context.sampling_library_hash),
                },
            )
        except Exception as exc:
            raise RuntimeError("Failed to emit RESAMPLE_TRIGGERED event.") from exc
