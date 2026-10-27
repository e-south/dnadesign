"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_b/test_stage_b_sampler.py

Stage-B sampler control-flow tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.core.pipeline.stage_b_sampler import (
    LibraryContext,
    LibraryRunResult,
    SamplingCounters,
    StageBSampler,
)
from dnadesign.densegen.src.core.runtime_policy import RuntimePolicy


def test_stage_b_sampler_resample_not_allowed_raises() -> None:
    policy = RuntimePolicy(
        pool_strategy="full",
        max_accepted_per_library=1,
        no_progress_seconds_before_resample=0,
        max_consecutive_no_progress_resamples=0,
    )
    counters = SamplingCounters()
    context = LibraryContext(
        library_for_opt=["AAA"],
        tfbs_parts=["TF:AAA"],
        regulator_labels=["TF"],
        sampling_info={"library_index": 1, "library_hash": "hash"},
        library_tfbs=["AAA"],
        library_tfs=["TF"],
        library_site_ids=[None],
        library_sources=[None],
        library_tfbs_ids=[None],
        library_motif_ids=[None],
        sampling_library_index=1,
        sampling_library_hash="hash",
        required_regulators=[],
        fixed_bp=0,
        min_required_bp=0,
        slack_bp=0,
        infeasible=False,
        min_required_len=0,
        min_breakdown={"fixed_elements_min": 0, "per_tf_min": 0, "min_required_extra": 0},
    )

    def _build_next_library() -> LibraryContext:
        return context

    def _run_library(
        library: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
        _ = (library, max_per_subsample, global_generated, quota)
        return LibraryRunResult(produced=0, stall_triggered=False, global_generated=global_generated)

    sampler = StageBSampler(
        source_label="demo",
        plan_name="demo",
        quota=1,
        policy=policy,
        max_per_subsample=1,
        pool_strategy="full",
        iterative_min_new_solutions=0,
        iterative_max_libraries=0,
        counters=counters,
    )

    with pytest.raises(RuntimeError, match="does not allow Stage-B resampling"):
        sampler.run(
            build_next_library=_build_next_library,
            run_library=_run_library,
        )


def _library_context() -> LibraryContext:
    return LibraryContext(
        library_for_opt=["AAA"],
        tfbs_parts=["TF:AAA"],
        regulator_labels=["TF"],
        sampling_info={"library_index": 1, "library_hash": "hash"},
        library_tfbs=["AAA"],
        library_tfs=["TF"],
        library_site_ids=[None],
        library_sources=[None],
        library_tfbs_ids=[None],
        library_motif_ids=[None],
        sampling_library_index=1,
        sampling_library_hash="hash",
        required_regulators=[],
        fixed_bp=0,
        min_required_bp=0,
        slack_bp=0,
        infeasible=False,
        min_required_len=0,
        min_breakdown={"fixed_elements_min": 0, "per_tf_min": 0, "min_required_extra": 0},
    )


def test_stage_b_sampler_one_subsample_emits_resample_telemetry_for_no_solution() -> None:
    policy = RuntimePolicy(
        pool_strategy="subsample",
        max_accepted_per_library=1,
        no_progress_seconds_before_resample=0,
        max_consecutive_no_progress_resamples=0,
    )
    counters = SamplingCounters()
    context = _library_context()
    resample_calls: list[tuple[str, int]] = []

    def _build_next_library() -> LibraryContext:
        return context

    def _run_library(
        library: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
        _ = (library, max_per_subsample, global_generated, quota)
        return LibraryRunResult(produced=0, stall_triggered=False, global_generated=global_generated)

    sampler = StageBSampler(
        source_label="demo",
        plan_name="demo",
        quota=1,
        policy=policy,
        max_per_subsample=1,
        pool_strategy="subsample",
        iterative_min_new_solutions=0,
        iterative_max_libraries=0,
        counters=counters,
    )
    sampler.run(
        build_next_library=_build_next_library,
        run_library=_run_library,
        one_subsample_only=True,
        on_resample=lambda _library, reason, produced: resample_calls.append((reason, produced)),
    )

    assert counters.total_resamples == 1
    assert resample_calls == [("no_solution", 0)]


def test_stage_b_sampler_carries_consecutive_failures_across_calls() -> None:
    policy = RuntimePolicy(
        pool_strategy="subsample",
        max_accepted_per_library=1,
        no_progress_seconds_before_resample=0,
        max_consecutive_no_progress_resamples=2,
    )
    counters = SamplingCounters()
    context = _library_context()

    def _build_next_library() -> LibraryContext:
        return context

    def _run_library(
        library: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
        _ = (library, max_per_subsample, global_generated, quota)
        return LibraryRunResult(produced=0, stall_triggered=False, global_generated=global_generated)

    sampler = StageBSampler(
        source_label="demo",
        plan_name="demo",
        quota=1,
        policy=policy,
        max_per_subsample=1,
        pool_strategy="subsample",
        iterative_min_new_solutions=0,
        iterative_max_libraries=0,
        counters=counters,
    )
    first = sampler.run(
        build_next_library=_build_next_library,
        run_library=_run_library,
        one_subsample_only=True,
        initial_consecutive_failures=0,
    )
    assert first.consecutive_failures_end == 1

    with pytest.raises(RuntimeError, match="max_consecutive_no_progress_resamples=2"):
        sampler.run(
            build_next_library=_build_next_library,
            run_library=_run_library,
            one_subsample_only=True,
            initial_consecutive_failures=first.consecutive_failures_end,
        )


def test_stage_b_sampler_uses_explicit_no_solution_reason() -> None:
    policy = RuntimePolicy(
        pool_strategy="subsample",
        max_accepted_per_library=1,
        no_progress_seconds_before_resample=0,
        max_consecutive_no_progress_resamples=0,
    )
    counters = SamplingCounters()
    context = _library_context()
    no_solution_reasons: list[str] = []

    def _build_next_library() -> LibraryContext:
        return context

    def _run_library(
        library: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
        _ = (library, max_per_subsample, global_generated, quota)
        return LibraryRunResult(
            produced=0,
            stall_triggered=False,
            global_generated=global_generated,
            no_solution_reason="infeasible_library",
        )

    sampler = StageBSampler(
        source_label="demo",
        plan_name="demo",
        quota=1,
        policy=policy,
        max_per_subsample=1,
        pool_strategy="subsample",
        iterative_min_new_solutions=0,
        iterative_max_libraries=0,
        counters=counters,
    )
    sampler.run(
        build_next_library=_build_next_library,
        run_library=_run_library,
        one_subsample_only=True,
        on_no_solution=lambda _library, reason: no_solution_reasons.append(reason),
    )
    assert no_solution_reasons == ["infeasible_library"]


def test_stage_b_sampler_resets_no_progress_counter_after_success() -> None:
    policy = RuntimePolicy(
        pool_strategy="subsample",
        max_accepted_per_library=2,
        no_progress_seconds_before_resample=0,
        max_consecutive_no_progress_resamples=0,
    )
    counters = SamplingCounters()
    context = _library_context()
    run_results = [
        LibraryRunResult(produced=1, stall_triggered=False, global_generated=1),
        LibraryRunResult(produced=1, stall_triggered=False, global_generated=2),
    ]

    def _build_next_library() -> LibraryContext:
        return context

    def _run_library(
        library: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
        _ = (library, max_per_subsample, global_generated, quota)
        return run_results.pop(0)

    sampler = StageBSampler(
        source_label="demo",
        plan_name="demo",
        quota=2,
        policy=policy,
        max_per_subsample=2,
        pool_strategy="subsample",
        iterative_min_new_solutions=0,
        iterative_max_libraries=0,
        counters=counters,
    )
    result = sampler.run(
        build_next_library=_build_next_library,
        run_library=_run_library,
    )
    assert result.generated == 2
    assert result.no_progress_seconds_end == 0.0


def test_stage_b_sampler_counts_inferred_stall_no_solution() -> None:
    policy = RuntimePolicy(
        pool_strategy="subsample",
        max_accepted_per_library=1,
        no_progress_seconds_before_resample=10,
        max_consecutive_no_progress_resamples=0,
    )
    counters = SamplingCounters()
    context = _library_context()
    no_solution_reasons: list[str] = []

    def _build_next_library() -> LibraryContext:
        return context

    def _run_library(
        library: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
        _ = (library, max_per_subsample, global_generated, quota)
        return LibraryRunResult(
            produced=0,
            stall_triggered=False,
            global_generated=global_generated,
            active_runtime_seconds=2.0,
        )

    sampler = StageBSampler(
        source_label="demo",
        plan_name="demo",
        quota=1,
        policy=policy,
        max_per_subsample=1,
        pool_strategy="subsample",
        iterative_min_new_solutions=0,
        iterative_max_libraries=0,
        counters=counters,
    )
    sampler.run(
        build_next_library=_build_next_library,
        run_library=_run_library,
        one_subsample_only=True,
        initial_no_progress_seconds=9.0,
        on_no_solution=lambda _library, reason: no_solution_reasons.append(reason),
    )
    assert no_solution_reasons == ["stall_no_solution"]
    assert counters.stall_events == 1


def test_stage_b_sampler_does_not_double_count_explicit_stall() -> None:
    policy = RuntimePolicy(
        pool_strategy="subsample",
        max_accepted_per_library=1,
        no_progress_seconds_before_resample=10,
        max_consecutive_no_progress_resamples=0,
    )
    counters = SamplingCounters()
    context = _library_context()

    def _build_next_library() -> LibraryContext:
        return context

    def _run_library(
        library: LibraryContext,
        max_per_subsample: int,
        global_generated: int,
        quota: int,
    ) -> LibraryRunResult:
        _ = (library, max_per_subsample, global_generated, quota)
        return LibraryRunResult(
            produced=0,
            stall_triggered=True,
            global_generated=global_generated,
            active_runtime_seconds=12.0,
        )

    sampler = StageBSampler(
        source_label="demo",
        plan_name="demo",
        quota=1,
        policy=policy,
        max_per_subsample=1,
        pool_strategy="subsample",
        iterative_min_new_solutions=0,
        iterative_max_libraries=0,
        counters=counters,
    )
    sampler.run(
        build_next_library=_build_next_library,
        run_library=_run_library,
        one_subsample_only=True,
    )
    assert counters.stall_events == 0
