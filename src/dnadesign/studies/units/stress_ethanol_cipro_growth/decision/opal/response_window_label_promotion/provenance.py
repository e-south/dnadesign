"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/provenance.py

Study-owned provenance construction for promoted OPAL labels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact import (
    SCHEMA_ID as OBSERVATION_SCHEMA_ID,
)

from .contracts import PROVENANCE_SCHEMA_ID, STUDY_ID, Y_SPACE
from .exclusions import build_candidate_selection_exclusion_provenance


def build_study_provenance(
    *,
    observation_manifest: dict[str, object],
    source_observation_manifest_path: PurePosixPath,
    source_observation_manifest_sha256: str,
    candidate_records_sha256: str,
    candidate_record_count: int,
    label_frame: pd.DataFrame,
    appended_label_event_count: int,
    observed_round: int,
    batch_id: str,
    candidate_exclusion_entries: list[dict[str, str]],
    prior_promotion: dict[str, object] | None,
) -> dict[str, object]:
    appended = label_frame.loc[
        label_frame["observed_round"].astype(int).eq(observed_round) & label_frame["batch_id"].astype(str).eq(batch_id)
    ]
    return {
        "schema_id": PROVENANCE_SCHEMA_ID,
        "schema_version": "5",
        "study_id": STUDY_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "observation_bundle": {
            "schema_id": OBSERVATION_SCHEMA_ID,
            "manifest_path": source_observation_manifest_path.as_posix(),
            "manifest_sha256": source_observation_manifest_sha256,
            "policy": observation_manifest["policy"],
            "source_manifests": observation_manifest["source_manifests"],
        },
        "candidate_table": {
            "path": "records.parquet",
            "records_sha256": candidate_records_sha256,
            "record_count": candidate_record_count,
        },
        "candidate_selection_exclusions": build_candidate_selection_exclusion_provenance(candidate_exclusion_entries),
        "prior_promotion": prior_promotion,
        "label_contract": {
            "y_space": Y_SPACE,
            "value_order": list(VALUE_COLUMNS),
            "observed_round": observed_round,
            "batch_id": batch_id,
            "label_event_count": len(label_frame),
            "unique_candidate_count": int(label_frame["id"].astype(str).nunique()),
            "appended_label_event_count": appended_label_event_count,
            "appended_unique_candidate_count": int(appended["id"].astype(str).nunique()),
            "observed_rounds": sorted(label_frame["observed_round"].astype(int).unique().tolist()),
            "batch_ids": list(dict.fromkeys(label_frame["batch_id"].astype(str).tolist())),
        },
    }


__all__ = ["build_study_provenance"]
