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
        arrays_generated_before_resample=1,
        stall_seconds_before_resample=0,
        stall_warning_every_seconds=0,
        max_consecutive_failures=0,
        max_seconds_per_plan=0,
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
