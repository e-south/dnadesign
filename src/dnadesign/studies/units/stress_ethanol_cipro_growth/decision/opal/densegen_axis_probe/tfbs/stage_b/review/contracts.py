"""Contracts for DenseGen TFBS Stage B realized-label review artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REALIZED_REVIEW_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_realized_review.v1"
VALID_NEGATIVE_CONTROL = "VALID_AS_NEGATIVE_CONTROL"


@dataclass(frozen=True)
class TfbsStageBRealizedReviewResult:
    """Paths for a materialized realized-label Stage B review."""

    status: str
    review_dir: Path
    trajectory_csv_path: Path
    pair_summary_csv_path: Path
    claim_assessment_csv_path: Path
    plot_manifest_json_path: Path
    slot_diagnostics_summary_json_path: Path | None
    slot_diagnostics_plot_manifest_json_path: Path | None
    notebook_visual_registration: Mapping[str, Any]
    summary_json_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "review_dir": str(self.review_dir),
            "trajectory_csv_path": str(self.trajectory_csv_path),
            "pair_summary_csv_path": str(self.pair_summary_csv_path),
            "claim_assessment_csv_path": str(self.claim_assessment_csv_path),
            "plot_manifest_json_path": str(self.plot_manifest_json_path),
            "slot_diagnostics_summary_json_path": (
                str(self.slot_diagnostics_summary_json_path)
                if self.slot_diagnostics_summary_json_path is not None
                else None
            ),
            "slot_diagnostics_plot_manifest_json_path": (
                str(self.slot_diagnostics_plot_manifest_json_path)
                if self.slot_diagnostics_plot_manifest_json_path is not None
                else None
            ),
            "notebook_visual_registration": dict(self.notebook_visual_registration),
            "summary_json_path": str(self.summary_json_path),
        }
