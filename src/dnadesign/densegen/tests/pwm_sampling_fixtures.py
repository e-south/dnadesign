"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/pwm_sampling_fixtures.py

Stage-A PWM sampling config helpers for tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable, Optional

from dnadesign.densegen.src.config import (
    PWMMiningBudgetConfig,
    PWMMiningConfig,
    PWMSamplingConfig,
    PWMSelectionConfig,
    PWMSelectionTierWidening,
)


def fixed_candidates_mining(
    *,
    batch_size: int,
    candidates: int,
    log_every_batches: int = 1,
) -> PWMMiningConfig:
    budget = PWMMiningBudgetConfig(mode="fixed_candidates", candidates=int(candidates))
    return PWMMiningConfig(batch_size=int(batch_size), budget=budget, log_every_batches=int(log_every_batches))


def tier_target_mining(
    *,
    batch_size: int,
    target_tier_fraction: float,
    max_candidates: Optional[int] = None,
    min_candidates: Optional[int] = None,
    max_seconds: Optional[float] = None,
    growth_factor: float = 1.25,
    log_every_batches: int = 1,
) -> PWMMiningConfig:
    budget = PWMMiningBudgetConfig(
        mode="tier_target",
        target_tier_fraction=float(target_tier_fraction),
        max_candidates=max_candidates,
        min_candidates=min_candidates,
        max_seconds=max_seconds,
        growth_factor=float(growth_factor),
    )
    return PWMMiningConfig(batch_size=int(batch_size), budget=budget, log_every_batches=int(log_every_batches))


def selection_top_score() -> PWMSelectionConfig:
    return PWMSelectionConfig(policy="top_score")


def selection_mmr(
    *,
    alpha: float = 0.9,
    min_score_norm: Optional[float] = None,
    max_candidates: Optional[int] = None,
    relevance_norm: str = "minmax_raw_score",
    tier_widening: Optional[Iterable[float]] = None,
) -> PWMSelectionConfig:
    tier_config = None
    if tier_widening is not None:
        tier_config = PWMSelectionTierWidening(
            enabled=True,
            ladder=[float(val) for val in tier_widening],
        )
    return PWMSelectionConfig(
        policy="mmr",
        alpha=float(alpha),
        pool={
            "min_score_norm": min_score_norm,
            "max_candidates": max_candidates,
            "relevance_norm": str(relevance_norm),
        },
        tier_widening=tier_config,
    )


def sampling_config(
    *,
    n_sites: int,
    mining: PWMMiningConfig,
    selection: Optional[PWMSelectionConfig] = None,
    strategy: str = "stochastic",
    keep_all_candidates_debug: bool = False,
    include_matched_sequence: bool = True,
    uniqueness_key: str = "core",
    length_policy: str = "exact",
    length_range: Optional[tuple[int, int]] = None,
    trim_window_length: Optional[int] = None,
    trim_window_strategy: str = "max_info",
) -> PWMSamplingConfig:
    return PWMSamplingConfig(
        n_sites=int(n_sites),
        strategy=str(strategy),
        mining=mining,
        selection=selection or selection_top_score(),
        keep_all_candidates_debug=bool(keep_all_candidates_debug),
        include_matched_sequence=bool(include_matched_sequence),
        uniqueness={"key": str(uniqueness_key)},
        length={"policy": str(length_policy), "range": length_range},
        trimming={"window_length": trim_window_length, "window_strategy": str(trim_window_strategy)},
    )
