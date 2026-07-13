"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_handoff_fixture.py

Candidate-handoff fixtures for Eco1 selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def candidate_handoff_payload() -> dict[str, object]:
    return {
        "schema_id": "thread_candidate_handoff_v1",
        "schema_version": 1,
        "handoff_id": "fixture_handoff",
        "handoff_kind": "rt_only_candidate_handoff",
        "study_id": "eco1_rt_repack",
        "subject_kind": "reverse_transcriptase_protein_only",
        "construct_subject_created": False,
        "downstream_acceptance_required": True,
        "source_artifacts": {
            "candidate_table": "candidate_table.parquet",
            "foldcheck_report": "foldcheck_report.parquet",
            "foldcheck_review": "generation_policies_v3/foldcheck_review/foldcheck_candidate_ranking.parquet",
            "candidate_triage_table": "generation_policies_v3/selection/candidate_triage_table.parquet",
            "candidate_selection_panel": "generation_policies_v3/selection/candidate_selection_panel.parquet",
            "candidate_handoff_sequences": "generation_policies_v3/selection/candidate_handoff_sequences.csv",
            "upstream_artifact_hashes": {"candidate_selection_panel": "sha256:" + "1" * 64},
        },
        "selection_policy": {"eligibility_rule": "fixture", "sae_acceptance_gate": False},
        "candidates": [
            {
                "candidate_id": "candidate_fixture",
                "sequence_hash": "sha256:" + "a" * 64,
                "candidate_handoff_sequence_csv_hash": "sha256:" + "b" * 64,
                "eligible_for_handoff": True,
                "foldcheck_status": "accepted",
                "selection_slot": 1,
            }
        ],
    }
