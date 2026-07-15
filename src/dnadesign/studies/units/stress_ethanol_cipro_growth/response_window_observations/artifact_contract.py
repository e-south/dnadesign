"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/artifact_contract.py

Identity and result contracts for response-window observation bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCHEMA_ID = "stress_ethanol_cipro_growth.response_window_observations.v1"
SCHEMA_VERSION = "1"
STUDY_ID = "stress_ethanol_cipro_growth"
RECORD_FILES = {
    "observations": "observations.parquet",
    "contributions": "contributions.parquet",
    "hierarchical_bootstrap_draws": "hierarchical_bootstrap_draws.parquet",
    "uncertainty": "uncertainty.parquet",
    "repeat_diagnostics": "repeat_diagnostics.parquet",
    "reduction_sensitivity": "reduction_sensitivity.parquet",
    "event_time_sensitivity": "event_time_sensitivity.parquet",
}


class ResponseWindowObservationArtifactError(ValueError):
    """Raised when an observation bundle is blocked, incomplete, or inconsistent."""


@dataclass(frozen=True)
class ResponseWindowObservationWriteResult:
    manifest_json: Path
    observations_parquet: Path
    candidate_count: int


@dataclass(frozen=True)
class ResponseWindowObservationVerification:
    manifest_json: Path
    manifest_sha256: str
    observations_parquet: Path
    candidate_count: int
    policy_id: str
    y_space: str


__all__ = [
    "RECORD_FILES",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "STUDY_ID",
    "ResponseWindowObservationArtifactError",
    "ResponseWindowObservationVerification",
    "ResponseWindowObservationWriteResult",
]
