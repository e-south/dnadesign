"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/auto_opt_models.py

Data models for auto-opt sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AutoOptCandidate:
    kind: str
    length: int | None
    budget: int | None
    cooling_boost: float
    move_profile: str
    swap_prob: float | None
    ladder_size: int | None
    move_probs: dict[str, float] | None
    move_probs_label: str | None
    run_dir: Path
    run_dirs: list[Path]
    best_score: float | None
    top_k_median_final: float | None
    best_score_final: float | None
    top_k_ci_low: float | None
    top_k_ci_high: float | None
    rhat: float | None
    ess: float | None
    unique_fraction: float | None
    balance_median: float | None
    diversity: float | None
    improvement: float | None
    acceptance_b: float | None
    acceptance_m: float | None
    acceptance_mh: float | None
    swap_rate: float | None
    status: str
    quality: str
    warnings: list[str]
    diagnostics: dict[str, object]
    pilot_score: float | None
    median_relevance_raw: float | None
    mean_pairwise_distance: float | None
    unique_successes: int | None


@dataclass(frozen=True)
class AutoOptSpec:
    kind: str
    length: int
    move_profile: str
    cooling_boost: float
    swap_prob: float | None = None
    ladder_size: int | None = None
    move_probs: dict[str, float] | None = None
    move_probs_label: str | None = None
