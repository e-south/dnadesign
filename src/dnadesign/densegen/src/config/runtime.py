"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/runtime.py

DenseGen runtime configuration schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round_robin: bool = False
    arrays_generated_before_resample: int = 1
    min_count_per_tf: int = 0
    max_duplicate_solutions: int = 3
    stall_seconds_before_resample: int = 30
    stall_warning_every_seconds: int = 15
    max_consecutive_failures: int = 25
    max_seconds_per_plan: int = 0
    max_failed_solutions: int = 0
    leaderboard_every: int = 50
    checkpoint_every: int = 50
    random_seed: int = 1337

    @field_validator(
        "arrays_generated_before_resample",
        "min_count_per_tf",
        "max_duplicate_solutions",
        "stall_seconds_before_resample",
        "stall_warning_every_seconds",
        "max_consecutive_failures",
        "max_seconds_per_plan",
        "max_failed_solutions",
        "leaderboard_every",
        "checkpoint_every",
    )
    @classmethod
    def _non_negative(cls, v: int, info):
        if v < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return v

    @field_validator("arrays_generated_before_resample")
    @classmethod
    def _arrays_positive(cls, v: int):
        if v <= 0:
            raise ValueError("arrays_generated_before_resample must be > 0")
        return v
