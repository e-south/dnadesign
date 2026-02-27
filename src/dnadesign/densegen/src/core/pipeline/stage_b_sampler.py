"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_sampler.py

Stage-B sampling loop control and resampling policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from ..runtime_policy import RuntimePolicy
from .stage_b_library_builder import LibraryContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LibraryRunResult:
    produced: int
    stall_triggered: bool
    global_generated: int
    no_solution_reason: str | None = None
    active_runtime_seconds: float = 0.0


@dataclass
class SamplingCounters:
    duplicate_records: int = 0
    failed_solutions: int = 0
    total_resamples: int = 0
    stall_events: int = 0
    failed_min_count_per_tf: int = 0
    failed_required_regulators: int = 0
    failed_min_count_by_regulator: int = 0
    failed_min_required_regulators: int = 0
    duplicate_solutions: int = 0


@dataclass(frozen=True)
class SamplingResult:
    generated: int
    total_resamples: int
    consecutive_failures_end: int
    no_progress_seconds_end: float


@dataclass
class StageBSampler:
    source_label: str
    plan_name: str
    quota: int
    policy: RuntimePolicy
    max_per_subsample: int
    pool_strategy: str
    iterative_min_new_solutions: int
    iterative_max_libraries: int
    counters: SamplingCounters

    def run(
        self,
        *,
        build_next_library: Callable[[], LibraryContext],
        run_library: Callable[[LibraryContext, int, int, int], LibraryRunResult],
        on_no_solution: Callable[[LibraryContext, str], None] | None = None,
        on_resample: Callable[[LibraryContext, str, int], None] | None = None,
        already_generated: int = 0,
        one_subsample_only: bool = False,
        initial_consecutive_failures: int = 0,
        initial_no_progress_seconds: float = 0.0,
    ) -> SamplingResult:
        global_generated = int(already_generated)
        produced_total_this_call = 0
        libraries_used = 0
        consecutive_failures = int(initial_consecutive_failures)
        no_progress_seconds = float(max(0.0, initial_no_progress_seconds))

        library = build_next_library()
        libraries_used += 1

        while global_generated < self.quota:
            result = run_library(
                library,
                self.max_per_subsample,
                global_generated,
                self.quota,
            )
            produced_total_this_call += int(result.produced)
            global_generated = int(result.global_generated)
            active_runtime_seconds = float(max(0.0, result.active_runtime_seconds))

            reason = "resample"
            if result.produced == 0:
                no_progress_seconds += active_runtime_seconds
                if result.no_solution_reason is not None:
                    reason = str(result.no_solution_reason)
                else:
                    stall_elapsed = (
                        self.policy.no_progress_seconds_before_resample > 0
                        and no_progress_seconds >= self.policy.no_progress_seconds_before_resample
                    )
                    reason = "stall_no_solution" if result.stall_triggered or stall_elapsed else "no_solution"
                if reason == "stall_no_solution" and not result.stall_triggered:
                    self.counters.stall_events += 1
                if on_no_solution is not None:
                    on_no_solution(library, reason)
            else:
                no_progress_seconds = 0.0
            if self.pool_strategy == "iterative_subsample" and self.iterative_min_new_solutions > 0:
                if result.produced < self.iterative_min_new_solutions:
                    log.info(
                        "[%s/%s] Library produced %d < iterative_min_new_solutions=%d; Stage-B resampling.",
                        self.source_label,
                        self.plan_name,
                        result.produced,
                        self.iterative_min_new_solutions,
                    )

            resample_reason = "resample"
            if result.produced == 0:
                resample_reason = reason
            elif self.pool_strategy == "iterative_subsample" and self.iterative_min_new_solutions > 0:
                if result.produced < self.iterative_min_new_solutions:
                    resample_reason = "min_new_solutions"

            if result.produced > 0:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if (
                    self.policy.max_consecutive_no_progress_resamples > 0
                    and consecutive_failures >= self.policy.max_consecutive_no_progress_resamples
                ):
                    raise RuntimeError(
                        f"[{self.source_label}/{self.plan_name}] Exceeded max_consecutive_no_progress_resamples="
                        f"{self.policy.max_consecutive_no_progress_resamples}."
                    )
            if one_subsample_only:
                if result.produced == 0 and self.policy.allow_resample():
                    self.counters.total_resamples += 1
                    if on_resample is not None:
                        on_resample(library, resample_reason, int(result.produced))
                break
            if global_generated >= self.quota or result.produced >= self.max_per_subsample:
                break

            if not self.policy.allow_resample():
                raise RuntimeError(
                    f"[{self.source_label}/{self.plan_name}] pool_strategy={self.pool_strategy!r} does not allow "
                    "Stage-B resampling. Reduce quota or use iterative_subsample."
                )
            self.counters.total_resamples += 1
            if on_resample is not None:
                on_resample(library, resample_reason, int(result.produced))
            if self.iterative_max_libraries > 0 and libraries_used >= self.iterative_max_libraries:
                raise RuntimeError(
                    f"[{self.source_label}/{self.plan_name}] Exceeded iterative_max_libraries="
                    f"{self.iterative_max_libraries}."
                )

            library = build_next_library()
            libraries_used += 1

        return SamplingResult(
            generated=produced_total_this_call,
            total_resamples=self.counters.total_resamples,
            consecutive_failures_end=int(consecutive_failures),
            no_progress_seconds_end=float(no_progress_seconds),
        )
