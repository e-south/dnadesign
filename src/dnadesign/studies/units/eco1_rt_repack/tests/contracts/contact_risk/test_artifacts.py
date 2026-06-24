"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/contact_risk/test_artifacts.py

Contact-risk artifact contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.contact_risk import (
    validate_contact_risk_profile_content,
)


def test_contact_risk_validator_rejects_rows_missing_risk_class(tmp_path: Path) -> None:
    path = tmp_path / "contact_risk_profile.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt_repack.contact_risk_profile",
                "schema_version": 1,
                "artifact_id": "eco1_rt_conservative_v1.contact_risk_profile",
                "status": "materialized",
                "created_by": "test",
                "created_at": "2026-06-22T00:00:00Z",
                "contact_risk_policy_id": "eco1_rt_contact_risk_audit_v1",
                "evidence_availability": {
                    "nearest_context_atom_distance": {"status": "materialized"},
                    "sidechain_context_distance": {"status": "not_materialized"},
                    "backbone_context_distance": {"status": "not_materialized"},
                    "contact_atom_density": {"status": "not_materialized"},
                    "retained_context_chain_count": {"status": "not_materialized"},
                },
                "sampling_decision": {"status": "not_sampling_authoritative"},
                "summary": {"total_positions": 1},
                "residues": [
                    {
                        "canonical_position": 1,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    issues = validate_contact_risk_profile_content(path)

    assert "eco1_rt.contact_risk.row_missing_required_field" in {issue.check_id for issue in issues}
