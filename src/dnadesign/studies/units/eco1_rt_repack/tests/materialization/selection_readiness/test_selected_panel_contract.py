"""Selected-panel contract tests for Eco1 RT selection readiness."""

from __future__ import annotations

from collections import Counter

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    build_selected_panel_rows,
    validate_selected_panel,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTED_PANEL_SIZE,
    allocation_for_policy,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_trace import (
    build_selected_panel_trace_rows,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._panel_contract_fixtures import (
    candidate_row,
    comparison_candidates,
    comparison_input_hashes,
)


def test_selected_panel_has_one_eight_row_allocation() -> None:
    triage_rows, candidate_rows = comparison_candidates()

    panel = build_selected_panel_rows(
        triage_rows=triage_rows,
        candidate_rows=candidate_rows,
        input_hashes=comparison_input_hashes(),
    )

    assert len(panel) == SELECTED_PANEL_SIZE == 8
    assert Counter(str(row["policy_id"]) for row in panel) == {
        DISTAL_SCAFFOLD_POLICY_ID: 2,
        NEAR_DNA_RNA_ACID_FREE_POLICY_ID: 3,
        COMBINED_NEAR_PLUS_DISTAL_POLICY_ID: 3,
    }
    assert Counter(str(row["design_group_id"]) for row in panel) == {
        "distal_scaffold_repack": 2,
        "peripheral_shell_repack": 3,
        "combined_peripheral_and_distal_repack": 3,
    }
    assert [int(row["selection_rank"]) for row in panel] == list(range(1, 9))
    assert [int(row["within_group_rank"]) for row in panel] == [1, 2, 1, 2, 3, 1, 2, 3]
    assert not any("first_order_member" in row for row in panel)
    assert not any("comparison_panel_member" in row for row in panel)
    validate_selected_panel(panel)


def test_panel_allocation_contract_is_the_single_count_authority() -> None:
    assert allocation_for_policy(DISTAL_SCAFFOLD_POLICY_ID).selected_count == 2
    assert allocation_for_policy(NEAR_DNA_RNA_ACID_FREE_POLICY_ID).selected_count == 3
    assert allocation_for_policy(COMBINED_NEAR_PLUS_DISTAL_POLICY_ID).selected_count == 3


def test_within_group_pair_maximizes_position_distance_before_late_evidence() -> None:
    triage_rows, candidate_rows = comparison_candidates()
    distal_candidates = [row for row in candidate_rows if row["policy_id"] == DISTAL_SCAFFOLD_POLICY_ID]
    distal_candidates[0]["canonical_mutations"] = ["A10G", "A20G"]
    distal_candidates[1]["canonical_mutations"] = ["A10K", "A20K"]
    distal_candidates[2]["canonical_mutations"] = ["A30G", "A40G"]

    panel = build_selected_panel_rows(
        triage_rows=triage_rows,
        candidate_rows=candidate_rows,
        input_hashes=comparison_input_hashes(),
    )

    distal_ids = {str(row["candidate_id"]) for row in panel if row["policy_id"] == DISTAL_SCAFFOLD_POLICY_ID}
    assert distal_ids == {"distal_1", "distal_3"}


def test_alpha1_burden_does_not_override_mutation_set_distance() -> None:
    triage_rows, candidate_rows = comparison_candidates()
    distal_triage = [row for row in triage_rows if row["policy_id"] == DISTAL_SCAFFOLD_POLICY_ID]
    distal_candidates = [row for row in candidate_rows if row["policy_id"] == DISTAL_SCAFFOLD_POLICY_ID]
    distal_candidates[0]["canonical_mutations"] = ["A10G", "A20G"]
    distal_candidates[1]["canonical_mutations"] = ["A30G", "A40G"]
    distal_candidates[2]["canonical_mutations"] = ["A10K", "A20K"]
    distal_triage[0]["wang_alpha1_mutation_count"] = 3
    distal_triage[1]["wang_alpha1_mutation_count"] = 1
    distal_triage[2]["wang_alpha1_mutation_count"] = 4

    panel = build_selected_panel_rows(
        triage_rows=triage_rows,
        candidate_rows=candidate_rows,
        input_hashes=comparison_input_hashes(),
    )

    selected_distal = [row for row in panel if row["policy_id"] == DISTAL_SCAFFOLD_POLICY_ID]
    assert [row["candidate_id"] for row in selected_distal] == ["distal_1", "distal_2"]
    assert [row["wang_alpha1_mutation_count"] for row in selected_distal] == [3, 1]


def test_selection_flow_has_no_procurement_subset() -> None:
    triage_rows, candidate_rows = comparison_candidates()
    panel = build_selected_panel_rows(
        triage_rows=triage_rows,
        candidate_rows=candidate_rows,
        input_hashes=comparison_input_hashes(),
    )

    trace = build_selected_panel_trace_rows(triage_rows=triage_rows, panel_rows=panel)

    assert [row["stage_id"] for row in trace] == [
        "candidate_pool",
        "local_geometry_screen",
        "design_groups",
        "selected_panel",
    ]
    design_groups = trace[2]
    assert design_groups["selector_role"] == "experimental_design"
    assert design_groups["is_hard_gate"] is False
    assert design_groups["removed_count"] == 0
    assert design_groups["distal_pool_count"] == 3
    assert design_groups["peripheral_pool_count"] == 3
    assert design_groups["combined_pool_count"] == 3
    assert trace[-1]["selected_count"] == 8
    assert trace[-1]["remaining_count"] == 8


def test_selected_panel_rejects_missing_policy_pool() -> None:
    triage_rows, candidate_rows = comparison_candidates()
    triage_rows = [row for row in triage_rows if row["policy_id"] != DISTAL_SCAFFOLD_POLICY_ID]
    candidate_rows = [row for row in candidate_rows if row["policy_id"] != DISTAL_SCAFFOLD_POLICY_ID]

    with pytest.raises(ValueError, match="distal_scaffold_repack_v1"):
        build_selected_panel_rows(
            triage_rows=triage_rows,
            candidate_rows=candidate_rows,
            input_hashes=comparison_input_hashes(),
        )


def test_selected_panel_rejects_contract_row_missing_from_candidate_pool() -> None:
    triage_rows, candidate_rows = comparison_candidates()
    candidate_rows = [row for row in candidate_rows if row["candidate_id"] != "distal_1"]

    with pytest.raises(ValueError, match="distal_1"):
        build_selected_panel_rows(
            triage_rows=triage_rows,
            candidate_rows=candidate_rows,
            input_hashes=comparison_input_hashes(),
        )


def test_selected_panel_rejects_unknown_contract_policy() -> None:
    triage_rows, candidate_rows = comparison_candidates()
    triage_rows.append(candidate_row("unknown_1", policy_id="unknown_policy_v1", na_facing_mutation_count=1))
    candidate_rows.append(
        {
            "candidate_id": "unknown_1",
            "policy_id": "unknown_policy_v1",
            "sequence": "A" * 64,
            "canonical_mutations": ["A45G"],
        }
    )

    with pytest.raises(ValueError, match="unknown_policy_v1"):
        build_selected_panel_rows(
            triage_rows=triage_rows,
            candidate_rows=candidate_rows,
            input_hashes=comparison_input_hashes(),
        )
