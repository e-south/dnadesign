"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/plan_context.py

Shared context objects for plan execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from ...adapters.outputs import SinkBase
from ...config import DenseGenConfig
from .deps import PipelineDeps

if TYPE_CHECKING:
    from ..artifacts.library import LibraryRecord
    from ..artifacts.pool import PoolData
    from .plan_execution import PlanKey


@dataclass(frozen=True)
class PlanRunContext:
    global_cfg: DenseGenConfig
    sinks: list[SinkBase]
    chosen_solver: str | None
    deps: PipelineDeps
    rng: random.Random
    np_rng: np.random.Generator
    cfg_path: Path
    run_id: str
    run_root: str
    run_config_path: str
    run_config_sha256: str
    random_seed: int | None
    dense_arrays_version: str | None
    dense_arrays_version_source: str
    show_tfbs: bool
    show_solutions: bool
    output_bio_type: str
    output_alphabet: str


@dataclass
class PlanExecutionState:
    inputs_manifest: dict[str, dict]
    existing_usage_counts: dict[tuple[str, str], int] | None = None
    state_counts: dict["PlanKey", int] | None = None
    checkpoint_every: int = 0
    write_state: Callable[[], None] | None = None
    site_failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] | None = None
    source_cache: dict[str, "PoolData"] | None = None
    pool_override: "PoolData" | None = None
    input_meta_override: dict | None = None
    attempt_counters: dict["PlanKey", int] | None = None
    library_records: dict["PlanKey", list["LibraryRecord"]] | None = None
    library_cursor: dict["PlanKey", int] | None = None
    library_source: str | None = None
    library_build_rows: list[dict] | None = None
    library_member_rows: list[dict] | None = None
    solution_rows: list[dict] | None = None
    composition_rows: list[dict] | None = None
    events_path: Path | None = None
    display_map_by_input: dict[str, dict[str, str]] | None = None
