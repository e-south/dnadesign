"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/learning_loop_baselines/contracts.py

Contracts for TFBS learning-loop baseline reviews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...profiles import (
    CANONICAL_COUNT_FRACTION_PROFILE_ID,
    SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID,
    SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID,
)

LEARNING_LOOP_BASELINE_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_learning_loop_baseline.v1"
LEARNING_LOOP_BASELINE_PLOT_MANIFEST_SCHEMA_VERSION = (
    "stress_ethanol_cipro_growth.tfbs_stage_b_learning_loop_baseline_plots.v1"
)
COUNT_FIXED_SLOT_POSITION_PROFILE_IDS = frozenset(
    {
        SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID,
        SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID,
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
    comparison_set_label="Composition learning loop",
    visual_tier="composition_learning_loop",
    accepted_profile_ids=frozenset({CANONICAL_COUNT_FRACTION_PROFILE_ID}),
    claim_boundary=(
        "This supports a harness-level active-learning claim for Dense Array count-fraction metadata. "
        "It does not claim stress-growth response, TF binding, or biological mechanism."
    ),
    interpretation_boundary=(
        "Frozen round-0 replay tests whether iterative retraining adds cumulative selected label enrichment "
        "beyond the initial X-based ranking. The same-budget known-label ranking shows how much label enrichment "
        "was available under the same acquired-budget constraint."
    ),
)

COUNT_FIXED_SLOT_POSITION_LEARNING_LOOP_SPEC = LearningLoopBaselineSpec(
    review_id="count_fixed_slot_position_learning_loop",
    comparison_set_key="count_fixed_slot_position_learning_loop_baseline",
    comparison_set_label="Placement learning loop",
    visual_tier="placement_learning_loop",
    accepted_profile_ids=COUNT_FIXED_SLOT_POSITION_PROFILE_IDS,
    claim_boundary=(
        "This is boundary evidence for TFBS placement construction metadata under count-fixed candidate "
        "scopes. It should not be generalized to all slot geometry, biology, TF binding, or stress response."
    ),
    interpretation_boundary=(
        "Frozen round-0 replay tests whether count-fixed placement enrichment required adaptive retraining or "
        "was already available from the initial X-based ranking. The same-budget known-label ranking gives a "
        "finite-budget placement-label reference."
    ),
)

LEARNING_LOOP_SPECS_BY_VISUAL_TIER = {
    COUNT_FRACTION_LEARNING_LOOP_SPEC.visual_tier: COUNT_FRACTION_LEARNING_LOOP_SPEC,
    COUNT_FIXED_SLOT_POSITION_LEARNING_LOOP_SPEC.visual_tier: COUNT_FIXED_SLOT_POSITION_LEARNING_LOOP_SPEC,
}


def learning_loop_spec_for_visual_tier(visual_tier: object) -> LearningLoopBaselineSpec:
    """Return the learning-loop evidence contract for a visual tier, or fail fast."""

    tier = str(visual_tier or "").strip()
    try:
        return LEARNING_LOOP_SPECS_BY_VISUAL_TIER[tier]
    except KeyError as exc:
        raise ValueError(f"unsupported TFBS learning-loop visual_tier: {visual_tier!r}") from exc


def validate_learning_loop_source_profiles(
    *,
    visual_tier: object,
    source_profile_ids: object,
    path: Path,
) -> tuple[str, ...]:
    """Validate source profiles before a learning-loop manifest can enter a portfolio."""

    spec = learning_loop_spec_for_visual_tier(visual_tier)
    if not isinstance(source_profile_ids, list):
        raise ValueError(f"TFBS learning-loop manifest source_profile_ids must be a list: {path}")
    normalized = tuple(str(value).strip() for value in source_profile_ids)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"TFBS learning-loop manifest source_profile_ids must contain non-empty ids: {path}")
    unsupported = sorted(set(normalized) - set(spec.accepted_profile_ids))
    if unsupported:
        raise ValueError(
            "TFBS learning-loop manifest source_profile_ids do not match its evidence tier: "
            f"tier={spec.visual_tier!r} unsupported={unsupported} accepted={sorted(spec.accepted_profile_ids)} "
            f"source={path}"
        )
    return normalized


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
