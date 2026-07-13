"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_local_structure_summary.py

Local-structure review summary tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
    LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM,
    build_local_structure_review_by_candidate,
)


def test_local_structure_review_summary_does_not_gate_on_distal_review_only_region() -> None:
    rows = [
        {
            "candidate_id": "candidate_a",
            "region_id": region_id,
            "status": "available",
            "local_ca_rmsd_angstrom": 1.0,
        }
        for region_id in LOCAL_STRUCTURE_REGION_IDS
    ]
    rows[-1]["status"] = "model_structure_missing"

    summary = build_local_structure_review_by_candidate(rows)["candidate_a"]

    assert summary["local_structure_gate_status"] == "passed"
    assert summary["local_structure_unavailable_region_count"] == 0
    assert summary["local_structure_distal_scaffold_control_ca_rmsd_angstrom"] is None
    assert summary["local_structure_gate_failure_reasons_json"] == "[]"


def test_local_structure_review_summary_fails_threshold_excess() -> None:
    rows = [
        {
            "candidate_id": "candidate_a",
            "region_id": region_id,
            "status": "available",
            "local_ca_rmsd_angstrom": 1.0,
            "local_ca_rmsd_threshold_status": "passed",
        }
        for region_id in LOCAL_STRUCTURE_REGION_IDS
    ]
    failed_region = "thumb_contact_track_context"
    threshold = LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM[failed_region]
    failed_row = next(row for row in rows if row["region_id"] == failed_region)
    failed_row["local_ca_rmsd_angstrom"] = threshold + 0.2
    failed_row["local_ca_rmsd_threshold_status"] = "threshold_exceeded"
    failed_row["local_ca_rmsd_threshold_angstrom"] = threshold

    summary = build_local_structure_review_by_candidate(rows)["candidate_a"]

    assert summary["local_structure_gate_status"] == "threshold_exceeded"
    assert summary["local_structure_threshold_failed_region_count"] == 1
    assert f"{failed_region}:local_ca_rmsd" in summary["local_structure_gate_failure_reasons_json"]
