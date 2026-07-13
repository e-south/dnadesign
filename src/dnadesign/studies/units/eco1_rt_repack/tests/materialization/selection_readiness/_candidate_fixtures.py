"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_candidate_fixtures.py

Candidate-row fixtures for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    DISTAL_SCAFFOLD_POLICY_ID,
    GENERATION_POLICY_VERSION,
    PRIMARY_POLICY_IDS,
)


def candidate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    policy_ids = [policy_id for policy_id in PRIMARY_POLICY_IDS for _ in range(3)]
    for index, policy_id in enumerate(policy_ids, start=1):
        rows.append(
            {
                "candidate_id": f"candidate_{index}",
                "sequence_hash": f"sha256:{index:064d}",
                "sequence": sequence(index),
                "status": "accepted",
                "rank": index,
                "policy_id": policy_id,
                "primary_policy_id": policy_id,
                "policy_version": GENERATION_POLICY_VERSION,
                "source_policy_ids": [policy_id],
                "seed": 101 + index,
                "temperature": 0.1 if index % 2 else 0.3,
                "mutation_count": 20 + index,
                "mutable_mutation_count": 20 + index,
                "protected_mutation_count": 0,
                "outside_mutable_positions": [],
                "canonical_mutations": [f"A{index + 2}G", f"L{index + 20}V"],
            }
        )
    rows.extend([_low_confidence_candidate(), _mask_blocked_candidate()])
    return rows


def sequence(offset: int) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(alphabet[(offset + i) % len(alphabet)] for i in range(309))


def _low_confidence_candidate() -> dict[str, object]:
    return {
        "candidate_id": "candidate_low_conf",
        "sequence_hash": "sha256:" + "a" * 64,
        "sequence": sequence(21),
        "status": "accepted",
        "rank": 999,
        "policy_id": DISTAL_SCAFFOLD_POLICY_ID,
        "primary_policy_id": DISTAL_SCAFFOLD_POLICY_ID,
        "policy_version": GENERATION_POLICY_VERSION,
        "source_policy_ids": [DISTAL_SCAFFOLD_POLICY_ID],
        "seed": 303,
        "temperature": 0.3,
        "mutation_count": 25,
        "mutable_mutation_count": 25,
        "protected_mutation_count": 0,
        "outside_mutable_positions": [],
        "canonical_mutations": ["A7G"],
    }


def _mask_blocked_candidate() -> dict[str, object]:
    return {
        "candidate_id": "candidate_blocked_by_mask",
        "sequence_hash": "sha256:" + "b" * 64,
        "sequence": sequence(22),
        "status": "accepted",
        "rank": 1000,
        "policy_id": DISTAL_SCAFFOLD_POLICY_ID,
        "primary_policy_id": DISTAL_SCAFFOLD_POLICY_ID,
        "policy_version": GENERATION_POLICY_VERSION,
        "source_policy_ids": [DISTAL_SCAFFOLD_POLICY_ID],
        "seed": 303,
        "temperature": 0.3,
        "mutation_count": 26,
        "mutable_mutation_count": 25,
        "protected_mutation_count": 1,
        "outside_mutable_positions": [198],
        "canonical_mutations": ["Y198F"],
    }
