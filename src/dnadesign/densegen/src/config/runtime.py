"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/runtime.py

DenseGen runtime configuration schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from math import isfinite

from pydantic import BaseModel, ConfigDict, field_validator


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round_robin: bool = False
    max_accepted_per_library: int = 1
    min_count_per_tf: int = 0
    max_duplicate_solutions: int = 3
    no_progress_seconds_before_resample: int = 30
    max_consecutive_no_progress_resamples: int = 25
    max_failed_solutions: int = 0
    max_failed_solutions_per_target: float = 0.0
    leaderboard_every: int = 50
    checkpoint_every: int = 50
    random_seed: int = 1337

    @field_validator(
        "max_accepted_per_library",
        "min_count_per_tf",
        "max_duplicate_solutions",
        "no_progress_seconds_before_resample",
        "max_consecutive_no_progress_resamples",
        "max_failed_solutions",
        "leaderboard_every",
        "checkpoint_every",
    )
    @classmethod
    def _non_negative(cls, v: int, info):
        if v < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return v

    @field_validator("max_accepted_per_library")
    @classmethod
    def _arrays_positive(cls, v: int):
        if v <= 0:
            raise ValueError("max_accepted_per_library must be > 0")
        return v

    @field_validator("max_failed_solutions_per_target")
    @classmethod
    def _failed_solutions_per_target_non_negative(cls, v: float):
        value = float(v)
        if not isfinite(value):
            raise ValueError("max_failed_solutions_per_target must be finite")
        if value < 0:
            raise ValueError("max_failed_solutions_per_target must be >= 0")
        return value
