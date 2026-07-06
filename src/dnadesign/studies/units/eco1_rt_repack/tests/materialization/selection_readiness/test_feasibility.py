"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_feasibility.py

Feasibility materialization tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.feasibility import (
    build_feasibility_rows,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._fixtures import (
    sequence,
)


def test_feasibility_preserves_positions_from_serialized_mutation_lists() -> None:
    rows = build_feasibility_rows(
        candidate_rows=[
            {
                "candidate_id": "candidate_serialized",
                "sequence_hash": "sha256:" + "c" * 64,
                "sequence": sequence(4),
                "status": "accepted",
                "mutation_count": 2,
                "mutable_mutation_count": 2,
                "protected_mutation_count": 0,
                "outside_mutable_positions": [],
                "canonical_mutations": "['A7G', 'L21V']",
            }
        ],
        foldcheck_report_rows=[
            {"candidate_id": "wild_type", "input_sequence_hash": "sha256:" + "0" * 64, "status": "accepted"},
            {"candidate_id": "candidate_serialized", "status": "accepted"},
        ],
        input_candidate_pool_hash="sha256:pool",
        input_mask_policy_hash="sha256:mask",
        input_foldcheck_report_hash="sha256:fold",
        created_at="2026-07-02T00:00:00Z",
    )

    assert rows[0]["max_mutation_window_mutation_count"] == 1
    assert rows[0]["mutation_windows_json"] == (
        '[{"density": 1.0, "end": 7, "length": 1, "mutation_count": 1, "start": 7}, '
        '{"density": 1.0, "end": 21, "length": 1, "mutation_count": 1, "start": 21}]'
    )


def test_feasibility_requires_wild_type_parent_sequence_hash() -> None:
    try:
        build_feasibility_rows(
            candidate_rows=[
                {
                    "candidate_id": "candidate_without_parent",
                    "sequence_hash": "sha256:" + "c" * 64,
                    "sequence": sequence(4),
                    "status": "accepted",
                    "mutation_count": 1,
                    "mutable_mutation_count": 1,
                    "protected_mutation_count": 0,
                    "outside_mutable_positions": [],
                    "canonical_mutations": ["A7G"],
                }
            ],
            foldcheck_report_rows=[{"candidate_id": "candidate_without_parent", "status": "accepted"}],
            input_candidate_pool_hash="sha256:pool",
            input_mask_policy_hash="sha256:mask",
            input_foldcheck_report_hash="sha256:fold",
            created_at="2026-07-02T00:00:00Z",
        )
    except ValueError as exc:
        assert "wild_type" in str(exc)
        assert "parent sequence hash" in str(exc)
    else:  # pragma: no cover - pytest assertion path
        raise AssertionError("feasibility rows must require a wild_type parent sequence hash")
