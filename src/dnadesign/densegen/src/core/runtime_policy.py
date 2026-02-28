"""
Runtime policy helpers for resampling and stall detection.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePolicy:
    pool_strategy: str
    max_accepted_per_library: int
    no_progress_seconds_before_resample: int
    max_consecutive_no_progress_resamples: int

    def allow_resample(self) -> bool:
        return self.pool_strategy in {"iterative_subsample", "subsample"}

    def should_trigger_stall(self, *, now: float, last_progress: float) -> bool:
        if self.no_progress_seconds_before_resample <= 0:
            return False
        return (now - last_progress) >= self.no_progress_seconds_before_resample

    def should_warn_stall(self, *, now: float, last_warn: float, last_progress: float) -> bool:
        if self.no_progress_seconds_before_resample <= 0:
            return False
        if (now - last_progress) < self.no_progress_seconds_before_resample:
            return False
        return (now - last_warn) >= self.no_progress_seconds_before_resample
