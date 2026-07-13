"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/core/contracts.py

Contracts for the stress-study response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

STRESS_STATE_IDS = ("00", "10", "01", "11")
STRESS_RMF_GREEDY_CAMPAIGN_SLUG = "secg_rmf_greedy"
EXPECTED_STRESS_TARGET_VIEW_IDS = ("ethanol", "ciprofloxacin", "and")


@dataclass(frozen=True)
class StressTargetView:
    id: str
    label: str
    target_mask: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("stress target view id must be non-empty.")
        if len(self.target_mask) != len(STRESS_STATE_IDS) or set(self.target_mask) - {0.0, 1.0}:
            raise ValueError(
                f"stress target view {self.id!r} must define one binary mask in state order {STRESS_STATE_IDS}."
            )
        if not any(self.target_mask) or all(self.target_mask):
            raise ValueError(f"stress target view {self.id!r} must contain at least one ON and one OFF state.")


@dataclass(frozen=True)
class SfxiSourceProvenance:
    source_id: str
    source_campaign_slug: str
    expected_run_id: str
    target_view_id: str
    lifecycle: Literal["provenance_only"] = "provenance_only"

    def __post_init__(self) -> None:
        if self.target_view_id not in EXPECTED_STRESS_TARGET_VIEW_IDS:
            raise ValueError(f"SFXI source provenance has unknown target view {self.target_view_id!r}.")


@dataclass(frozen=True)
class StressCampaignContract:
    slug: str
    config_path: Path
    target_views: tuple[StressTargetView, ...]
    candidate_records_path: Path
    x_column_name: str


@dataclass(frozen=True)
class SfxiEvidenceFrame:
    source: SfxiSourceProvenance
    target_view: StressTargetView
    predictions: pd.DataFrame
    y_hat: np.ndarray
    denom: float
    run_id: str
    scaling_percentile: int = 95
    scaling_min_n: int = 5
    scaling_eps: float = 1.0e-8
    intensity_log2_offset_delta: float = 0.0
    records_path: Path | None = None
    x_column_name: str = ""
    model_params: Mapping[str, object] = field(default_factory=dict)
    yops_eps: float = 1.0e-8
    stats_n_train: int = 0
    stats_n_scored: int = 0


PolicyKind = Literal[
    "multiplicative",
    "logic_gate",
    "lexicographic",
    "off_state_logic_penalty",
]


@dataclass(frozen=True)
class PolicySpec:
    id: str
    label: str
    kind: PolicyKind
    beta: float = 1.0
    gamma: float = 1.0
    logic_gate: float | None = None
    off_state_logic_eta: float = 0.0
    tier: str = "sweep"
    plain_rule: str = ""


@dataclass(frozen=True)
class MetastudyPaths:
    repo_root: Path
    reader_bundle_root: Path
    out_dir: Path
    campaign_root: Path


@dataclass(frozen=True)
class RecommendationThresholds:
    min_eligible_count: int = 1000
    min_effective_topk: int = 6
    min_target_view_median_logic: float = 0.45
    max_all_target_views_overlap: int = 1
    max_mean_pairwise_score_spearman: float = 0.85
    min_target_view_cv_score_spearman: float = 0.30


DEFAULT_RECOMMENDATION_THRESHOLDS = RecommendationThresholds()


SFXI_SOURCE_PROVENANCE = (
    SfxiSourceProvenance(
        source_id="stress_sfxi_round0_ethanol",
        source_campaign_slug="secg_ethanol_rf_sfxi_topn",
        expected_run_id="r0-2026-07-09T18:37:10+00:00",
        target_view_id="ethanol",
    ),
    SfxiSourceProvenance(
        source_id="stress_sfxi_round0_ciprofloxacin",
        source_campaign_slug="secg_cipro_rf_sfxi_topn",
        expected_run_id="r0-2026-07-09T18:37:49+00:00",
        target_view_id="ciprofloxacin",
    ),
    SfxiSourceProvenance(
        source_id="stress_sfxi_round0_and",
        source_campaign_slug="secg_and_rf_sfxi_topn",
        expected_run_id="r0-2026-07-09T18:38:31+00:00",
        target_view_id="and",
    ),
)
