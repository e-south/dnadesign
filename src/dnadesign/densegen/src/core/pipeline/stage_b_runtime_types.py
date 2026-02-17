"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_runtime_types.py

Stage-B runtime dataclasses and local dashboard lifecycle helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..artifacts.pool import PoolData
from ..runtime_policy import RuntimePolicy
from .progress import PlanProgressReporter, _ScreenDashboard


def _close_plan_dashboard(*, dashboard, shared_dashboard) -> None:
    if dashboard is not None and dashboard is not shared_dashboard:
        dashboard.close()


@dataclass(frozen=True)
class RuntimeSettings:
    max_per_subsample: int
    min_count_per_tf: int
    max_dupes: int
    stall_seconds: int
    stall_warn_every: int
    max_consecutive_failures: int
    max_seconds_per_plan: int
    max_failed_solutions: int
    leaderboard_every: int
    checkpoint_every: int


@dataclass(frozen=True)
class PadSettings:
    enabled: bool
    mode: str
    end: str
    gc_mode: str
    gc_min: float
    gc_max: float
    gc_target: float
    gc_tolerance: float
    gc_min_length: int
    max_tries: int


@dataclass(frozen=True)
class SolverSettings:
    strategy: str
    strands: str
    time_limit_seconds: float | None
    threads: int | None


@dataclass(frozen=True)
class ProgressSettings:
    progress_style: str
    progress_every: int
    progress_refresh_seconds: float
    print_visual: bool
    tf_colors: dict[str, str] | None
    show_tfbs: bool
    show_solutions: bool
    dashboard: _ScreenDashboard | None
    reporter: PlanProgressReporter


@dataclass(frozen=True)
class PlanRunSettings:
    source_label: str
    plan_name: str
    quota: int
    seq_len: int
    sampling_cfg: object
    pool_strategy: str
    runtime: RuntimeSettings
    pad: PadSettings
    solver: SolverSettings
    extra_library_label: str | None
    progress: ProgressSettings
    policy: RuntimePolicy
    policy_pad: str
    policy_sampling: str
    policy_solver: str
    plan_start: float


@dataclass(frozen=True)
class PlanInputState:
    pool: PoolData
    data_entries: list
    meta_df: pd.DataFrame | None
    input_meta: dict
    input_row_count: int
    input_tf_count: int
    input_tfbs_count: int
    input_tf_tfbs_pair_count: int | None


@dataclass(frozen=True)
class SequenceConstraintEvaluation:
    promoter_detail: dict
    sequence_validation: dict
    rejection_detail: dict | None
    rejection_event_payload: dict | None
    error: Exception | None = None
