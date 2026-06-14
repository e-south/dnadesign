"""Contracts for replicated DenseGen TFBS Stage B review artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPLICATED_REVIEW_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_replicated_review.v1"
REPLICATED_REVIEW_ENDPOINT_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_replicated_endpoints.v1"
TFBS_STAGE_B_DETERMINISTIC_REPLICATE_SEEDS = (7, 17, 29)

CLAIM_READY_REPLICATED = "READY_AS_REPLICATED_VALID_NULL_LEARNABILITY_SIGNAL"
CLAIM_BLOCKED_INCOMPLETE_REPLICATES = "BLOCKED_INCOMPLETE_REPLICATE_SET"
CLAIM_BLOCKED_REPLICATE_NOT_READY = "BLOCKED_REPLICATE_NOT_READY"
CLAIM_BLOCKED_NONPOSITIVE_REPLICATED_ENDPOINT = "BLOCKED_NONPOSITIVE_REPLICATED_ENDPOINT"
CLAIM_LIMITED_INVALID_NEGATIVE_CONTROL = "LIMITED_INVALID_NEGATIVE_CONTROL_REPLICATE"


@dataclass(frozen=True)
class TfbsStageBReplicateManifest:
    """Validated config-manifest payload for one deterministic replicate seed."""

    path: Path
    seed: int
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class TfbsStageBReplicatedReviewResult:
    """Paths for a replicated realized-label Stage B review."""

    status: str
    review_dir: Path
    trajectory_csv_path: Path
    replicate_pair_summary_csv_path: Path
    endpoint_summary_csv_path: Path
    claim_assessment_csv_path: Path
    plot_manifest_json_path: Path
    notebook_visual_registration: Mapping[str, Any]
    summary_json_path: Path
    replicate_count: int
    replicate_seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "review_dir": str(self.review_dir),
            "trajectory_csv_path": str(self.trajectory_csv_path),
            "replicate_pair_summary_csv_path": str(self.replicate_pair_summary_csv_path),
            "endpoint_summary_csv_path": str(self.endpoint_summary_csv_path),
            "claim_assessment_csv_path": str(self.claim_assessment_csv_path),
            "plot_manifest_json_path": str(self.plot_manifest_json_path),
            "notebook_visual_registration": dict(self.notebook_visual_registration),
            "summary_json_path": str(self.summary_json_path),
            "replicate_count": int(self.replicate_count),
            "replicate_seeds": list(self.replicate_seeds),
        }
