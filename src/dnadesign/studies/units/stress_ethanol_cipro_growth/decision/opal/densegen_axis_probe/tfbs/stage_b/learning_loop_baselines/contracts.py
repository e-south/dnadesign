"""Contracts for TFBS learning-loop baseline reviews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEARNING_LOOP_BASELINE_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_learning_loop_baseline.v1"
LEARNING_LOOP_BASELINE_PLOT_MANIFEST_SCHEMA_VERSION = (
    "stress_ethanol_cipro_growth.tfbs_stage_b_learning_loop_baseline_plots.v1"
)
COUNT_FRACTION_PROFILE_ID = "tfbs_count_fraction_probe_v1"
COUNT_FIXED_SLOT_POSITION_PROFILE_IDS = frozenset(
    {
        "tfbs_slot_position_count_fixed_sentinel_probe_v1",
        "tfbs_slot_position_count_fixed_baer_middle_probe_v1",
    }
)
LEARNING_LOOP_BASELINE_SURFACE_KIND = "study_learning_loop_baseline"


@dataclass(frozen=True)
class LearningLoopBaselineSpec:
    """Contract for one learning-loop baseline review surface."""

    review_id: str
    comparison_set_key: str
    comparison_set_label: str
    visual_tier: str
    accepted_profile_ids: frozenset[str]
    claim_boundary: str
    interpretation_boundary: str


COUNT_FRACTION_LEARNING_LOOP_SPEC = LearningLoopBaselineSpec(
    review_id="count_fraction_learning_loop",
    comparison_set_key="count_fraction_learning_loop_baseline",
    comparison_set_label="Count-fraction learning-loop baseline",
    visual_tier="current_claim",
    accepted_profile_ids=frozenset({COUNT_FRACTION_PROFILE_ID}),
    claim_boundary=(
        "This supports a harness-level active-learning claim for synthetic DenseGen count-fraction metadata. "
        "It does not claim stress-growth response, TF binding, or biological mechanism."
    ),
    interpretation_boundary=(
        "Frozen round-0 replay tests whether iterative retraining adds cumulative selected label enrichment "
        "beyond the initial X-based ranking. The same-budget top-label reference shows how much enrichment was "
        "achievable under the same acquired-budget constraint."
    ),
)

COUNT_FIXED_SLOT_POSITION_LEARNING_LOOP_SPEC = LearningLoopBaselineSpec(
    review_id="count_fixed_slot_position_learning_loop",
    comparison_set_key="count_fixed_slot_position_learning_loop_baseline",
    comparison_set_label="Count-fixed placement learning-loop baseline",
    visual_tier="current_boundary",
    accepted_profile_ids=COUNT_FIXED_SLOT_POSITION_PROFILE_IDS,
    claim_boundary=(
        "This is boundary evidence for synthetic DenseGen TFBS placement metadata under count-fixed candidate "
        "scopes. It should not be generalized to all slot geometry, biology, TF binding, or stress response."
    ),
    interpretation_boundary=(
        "Frozen round-0 replay tests whether count-fixed placement enrichment required adaptive retraining or "
        "was already available from the initial X-based ranking. The same-budget top-label reference gives the "
        "achievable placement-label ceiling."
    ),
)


@dataclass(frozen=True)
class FrozenReplayResult:
    """Materialized frozen replay artifact paths."""

    status: str
    review_dir: Path
    manifest_json_path: Path
    trajectory_csv_path: Path
    endpoint_summary_csv_path: Path
    claim_interpretation_csv_path: Path
    plot_manifest_json_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "review_dir": str(self.review_dir),
            "manifest_json_path": str(self.manifest_json_path),
            "trajectory_csv_path": str(self.trajectory_csv_path),
            "endpoint_summary_csv_path": str(self.endpoint_summary_csv_path),
            "claim_interpretation_csv_path": str(self.claim_interpretation_csv_path),
            "plot_manifest_json_path": str(self.plot_manifest_json_path),
        }
