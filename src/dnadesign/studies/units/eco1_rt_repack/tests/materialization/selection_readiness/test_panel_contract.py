"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_panel_contract.py

Panel coverage contract tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.handoff_readiness import (
    build_handoff_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    PRIMARY_PANEL_SIZE,
    build_primary_panel_selection_trace_rows,
    build_selection_panel_rows,
    validate_primary_panel,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._handoff_fixture import (
    candidate_handoff_payload,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._panel_contract_fixtures import (
    BOUNDARY_CLASS,
    PRIMARY_CLASS,
    candidate_row,
    panel_rows,
    primary_candidate_rows,
)


def test_primary_panel_allows_duplicate_design_classes() -> None:
    rows = panel_rows([PRIMARY_CLASS] * PRIMARY_PANEL_SIZE)

    validate_primary_panel(rows, required_panel_size=PRIMARY_PANEL_SIZE)


def test_primary_panel_rejects_wrong_panel_size() -> None:
    rows = panel_rows([PRIMARY_CLASS] * (PRIMARY_PANEL_SIZE - 1))

    with pytest.raises(ValueError, match="Selected rows: 5"):
        validate_primary_panel(rows, required_panel_size=PRIMARY_PANEL_SIZE)


def test_primary_panel_selection_ignores_non_primary_rows_even_with_better_static_values() -> None:
    rows = [
        candidate_row(
            "non_primary_row",
            design_class_id=BOUNDARY_CLASS,
            tier="not_panel_candidate",
            na_facing_mutation_count=1,
            chemistry_warning_count=0,
            mutation_count_total=1,
        ),
        *primary_candidate_rows(),
    ]

    panel = build_selection_panel_rows(
        triage_rows=rows,
        candidate_rows=[{"candidate_id": row["candidate_id"], "sequence": "A" * 12} for row in rows],
        input_hashes={
            "candidate_triage_table": "sha256:triage",
            "foldcheck_review": "sha256:fold",
            "feasibility_report": "sha256:feasibility",
            "sae_window_summary": None,
        },
    )

    assert len(panel) == PRIMARY_PANEL_SIZE
    assert "non_primary_row" not in {row["candidate_id"] for row in panel}
    assert {row["design_class_id"] for row in panel} == {PRIMARY_CLASS}
    assert {row["selection_candidate_tier"] for row in panel} == {"primary_panel_candidate"}


def test_primary_panel_trace_records_simplified_funnel_without_noop_msa_stage() -> None:
    rows = [
        candidate_row("primary_kept", na_facing_mutation_count=1),
        candidate_row(
            "acidic_gain_blocked",
            tier="not_panel_candidate",
            na_facing_mutation_count=1,
            acidic_gain_count=1,
        ),
        {
            **candidate_row("blocked", tier="not_panel_candidate", na_facing_mutation_count=1),
            "hard_gate_status": "ineligible",
        },
    ]

    trace = build_primary_panel_selection_trace_rows(
        triage_rows=rows,
        panel_rows=[{"candidate_id": "primary_kept"}],
    )

    stage_by_id = {row["stage_id"]: row for row in trace}
    assert stage_by_id["preservation_gate"]["remaining_count"] == 2
    assert "substrate-relevant local RMSD <= 3.0" not in stage_by_id["preservation_gate"]["filter_rule"]
    assert "C-terminal/thumb local RMSD within" not in stage_by_id["preservation_gate"]["filter_rule"]
    assert "passed declared local RMSD gate" in stage_by_id["preservation_gate"]["filter_rule"]
    assert stage_by_id["chemistry_support_gate"]["remaining_count"] == 1
    assert stage_by_id["global_conservative_diverse_selection"]["remaining_count"] == 1
    assert "primary_panel_candidate_pool" not in stage_by_id
    assert "primary_proximal_msa_support_gate" not in stage_by_id


def test_primary_panel_trace_records_chemistry_support_gate() -> None:
    rows = [
        candidate_row("primary_kept", na_facing_mutation_count=0),
        candidate_row(
            "acidic_gain_row",
            tier="not_panel_candidate",
            na_facing_mutation_count=2,
            acidic_gain_count=1,
        ),
        candidate_row(
            "proximal_unobserved_row",
            tier="not_panel_candidate",
            na_facing_mutation_count=2,
            proximal_unobserved_mutation_count=1,
        ),
    ]

    trace = build_primary_panel_selection_trace_rows(
        triage_rows=rows,
        panel_rows=[{"candidate_id": "primary_kept"}],
    )

    stage_by_id = {row["stage_id"]: row for row in trace}
    assert [row["stage_id"] for row in trace] == [
        "candidate_pool",
        "preservation_gate",
        "chemistry_support_gate",
        "global_conservative_diverse_selection",
    ]
    assert stage_by_id["preservation_gate"]["remaining_count"] == 3
    assert stage_by_id["chemistry_support_gate"]["remaining_count"] == 1
    assert "acidic gains" in stage_by_id["chemistry_support_gate"]["filter_rule"]
    assert "unobserved proximal substitutions" in stage_by_id["chemistry_support_gate"]["filter_rule"]


def test_handoff_readiness_uses_thread_root_candidate_handoff(tmp_path) -> None:
    selection_root = tmp_path / "outputs/thread/design_classes/selection"
    thread_handoff_path = tmp_path / "outputs/thread/candidate_handoff.yaml"
    selection_root.mkdir(parents=True)
    (selection_root / "candidate_handoff.yaml").write_text("handoff_kind: wrong_local_path\n", encoding="utf-8")

    readiness = build_handoff_readiness(
        selection_root=selection_root,
        panel_rows=panel_rows([PRIMARY_CLASS] * PRIMARY_PANEL_SIZE),
        candidate_handoff_path=thread_handoff_path,
    )

    assert readiness["candidate_handoff_path"] == "../../candidate_handoff.yaml"
    assert readiness["candidate_handoff_file_present"] is False
    assert readiness["candidate_handoff_materialized"] is False

    thread_handoff_path.write_text("handoff_kind: rt_only_candidate_handoff\n", encoding="utf-8")

    invalid_readiness = build_handoff_readiness(
        selection_root=selection_root,
        panel_rows=panel_rows([PRIMARY_CLASS] * PRIMARY_PANEL_SIZE),
        candidate_handoff_path=thread_handoff_path,
    )
    assert invalid_readiness["candidate_handoff_file_present"] is True
    assert invalid_readiness["candidate_handoff_materialized"] is False

    thread_handoff_path.write_text(yaml.safe_dump(candidate_handoff_payload(), sort_keys=False), encoding="utf-8")

    assert (
        build_handoff_readiness(
            selection_root=selection_root,
            panel_rows=panel_rows([PRIMARY_CLASS] * PRIMARY_PANEL_SIZE),
            candidate_handoff_path=thread_handoff_path,
        )["candidate_handoff_materialized"]
        is True
    )
