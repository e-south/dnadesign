"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/sampling/test_candidate_handoff.py

Candidate-handoff contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling import validate_candidate_handoff_content


def test_candidate_handoff_validator_rejects_stub_yaml(tmp_path: Path) -> None:
    handoff_path = tmp_path / "candidate_handoff.yaml"
    handoff_path.write_text("handoff_kind: rt_only_candidate_handoff\n", encoding="utf-8")

    issues = validate_candidate_handoff_content(handoff_path)

    check_ids = {issue.check_id for issue in issues}
    assert "eco1_rt.handoff.missing_required_field" in check_ids
    assert "eco1_rt.handoff.invalid_source_artifacts" in check_ids


def test_candidate_handoff_validator_rejects_construct_and_sae_gate_drift(tmp_path: Path) -> None:
    handoff_path = tmp_path / "candidate_handoff.yaml"
    handoff_path.write_text(yaml.safe_dump(_handoff_payload(), sort_keys=False), encoding="utf-8")

    issues = validate_candidate_handoff_content(handoff_path)

    check_ids = {issue.check_id for issue in issues}
    assert "eco1_rt.handoff.required_value_mismatch" in check_ids
    assert "eco1_rt.handoff.forbidden_field" in check_ids


def _handoff_payload() -> dict[str, object]:
    return {
        "schema_id": "thread_candidate_handoff_v1",
        "schema_version": 1,
        "handoff_id": "fixture",
        "handoff_kind": "rt_only_candidate_handoff",
        "study_id": "eco1_rt_repack",
        "subject_kind": "reverse_transcriptase_protein_only",
        "construct_subject_created": True,
        "construct_subject_id": "forbidden",
        "downstream_acceptance_required": True,
        "source_artifacts": {
            "candidate_table": "candidate_table.parquet",
            "foldcheck_report": "foldcheck_report.parquet",
            "foldcheck_review": "foldcheck_review/foldcheck_candidate_ranking.parquet",
            "feasibility_report": "selection/feasibility_report.parquet",
            "candidate_triage_table": "selection/candidate_triage_table.parquet",
            "candidate_selection_panel": "selection/candidate_selection_panel.parquet",
            "candidate_handoff_sequences": "selection/candidate_handoff_sequences.csv",
            "upstream_artifact_hashes": {"candidate_table": "sha256:" + "1" * 64},
        },
        "selection_policy": {"eligibility_rule": "fixture", "sae_acceptance_gate": True},
        "candidates": [
            {
                "candidate_id": "candidate_a",
                "sequence_hash": "sha256:" + "a" * 64,
                "candidate_handoff_sequence_csv_hash": "sha256:" + "b" * 64,
                "eligible_for_handoff": True,
                "foldcheck_status": "accepted",
                "feasibility_status": "feasible",
                "selection_slot": 1,
            }
        ],
    }
