"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_solution_types.py

Shared Stage-B dataclasses for rejection, persistence, and progress contexts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class StageBRejectionContext:
    source_label: str
    plan_name: str
    tables_root: Path
    run_id: str
    dense_arrays_version: str | None
    dense_arrays_version_source: str | None
    attempts_buffer: list[dict]
    next_attempt_index: Callable[[], int]


@dataclass(frozen=True)
class StageBSolutionOutputContext:
    source_label: str
    plan_name: str
    fixed_elements: Any
    fixed_elements_dump: dict
    chosen_solver: str | None
    solver_strategy: str
    solver_attempt_timeout_seconds: float | None
    solver_threads: int | None
    solver_strands: str
    seq_len: int
    schema_version: str
    run_id: str
    run_root: str
    run_config_path: str
    run_config_sha256: str
    random_seed: int
    policy_pad: str
    policy_sampling: str
    policy_solver: str
    input_meta: dict
    min_count_per_tf: int
    plan_min_count_by_regulator: dict[str, int]
    input_row_count: int
    input_tf_count: int
    input_tfbs_count: int
    input_tf_tfbs_pair_count: int | None
    output_bio_type: str
    output_alphabet: str
    tables_root: Path
    next_attempt_index: Callable[[], int]
    dense_arrays_version: str | None
    dense_arrays_version_source: str | None
    sinks: list[Any]
    attempts_buffer: list[dict]
    solution_rows: list[dict] | None
    composition_rows: list[dict] | None
    events_path: Path | None


@dataclass(frozen=True)
class StageBProgressContext:
    source_label: str
    plan_name: str
    checkpoint_every: int
    tables_root: Path
    attempts_buffer: list[dict]
    solution_rows: list[dict] | None
    state_counts: dict[tuple[str, str], int] | None
    write_state: Callable[[], None] | None
    total_quota: int | None
    progress_reporter: Any
    counters: Any
    failure_counts: dict
    leaderboard_every: int
    show_solutions: bool
    usage_counts: dict[tuple[str, str], int]
    tf_usage_counts: dict[str, int]
    diagnostics: Any
    logger: Any
