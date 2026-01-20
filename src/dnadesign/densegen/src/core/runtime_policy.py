"""
Runtime policy helpers for resampling and stall detection.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePolicy:
    pool_strategy: str
    arrays_generated_before_resample: int
    stall_seconds_before_resample: int
    stall_warning_every_seconds: int
    max_resample_attempts: int
    max_total_resamples: int
    max_seconds_per_plan: int

    def allow_resample(self) -> bool:
        return self.pool_strategy in {"iterative_subsample", "subsample"}

    def should_trigger_stall(self, *, now: float, subsample_started: float, produced_this_library: int) -> bool:
        if self.stall_seconds_before_resample <= 0:
            return False
        return (produced_this_library == 0) and (now - subsample_started >= self.stall_seconds_before_resample)

    def should_warn_stall(self, *, now: float, last_warn: float, produced_this_library: int) -> bool:
        if self.stall_warning_every_seconds <= 0:
            return False
        return (produced_this_library == 0) and (now - last_warn >= self.stall_warning_every_seconds)

    def plan_timed_out(self, *, now: float, plan_started: float) -> bool:
        if self.max_seconds_per_plan <= 0:
            return False
        return (now - plan_started) >= float(self.max_seconds_per_plan)
