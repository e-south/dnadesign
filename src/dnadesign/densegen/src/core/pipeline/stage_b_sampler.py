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
import time
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
        plan_start: float | None = None,
    ) -> SamplingResult:
        global_generated = int(already_generated)
        produced_total_this_call = 0
        libraries_used = 0
        consecutive_failures = 0
        plan_start = float(plan_start) if plan_start is not None else time.monotonic()

        library = build_next_library()
        libraries_used += 1

        while global_generated < self.quota:
            if self.policy.plan_timed_out(now=time.monotonic(), plan_started=plan_start):
                raise RuntimeError(
                    f"[{self.source_label}/{self.plan_name}] Exceeded max_seconds_per_plan="
                    f"{self.policy.max_seconds_per_plan}."
                )

            result = run_library(
                library,
                self.max_per_subsample,
                global_generated,
                self.quota,
            )
            produced_total_this_call += int(result.produced)
            global_generated = int(result.global_generated)

            if result.produced == 0:
                reason = "stall_no_solution" if result.stall_triggered else "no_solution"
                if on_no_solution is not None:
                    on_no_solution(library, reason)
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
                resample_reason = "stall_no_solution" if result.stall_triggered else "no_solution"
            elif self.pool_strategy == "iterative_subsample" and self.iterative_min_new_solutions > 0:
                if result.produced < self.iterative_min_new_solutions:
                    resample_reason = "min_new_solutions"

            if result.produced == 0:
                consecutive_failures += 1
                if (
                    self.policy.max_consecutive_failures > 0
                    and consecutive_failures >= self.policy.max_consecutive_failures
                ):
                    raise RuntimeError(
                        f"[{self.source_label}/{self.plan_name}] Exceeded max_consecutive_failures="
                        f"{self.policy.max_consecutive_failures}."
                    )
            else:
                consecutive_failures = 0

            if one_subsample_only:
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
        )
