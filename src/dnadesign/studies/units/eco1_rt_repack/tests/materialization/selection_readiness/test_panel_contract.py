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
    _choose_primary_candidate,
    build_primary_panel_selection_trace_rows,
    build_selection_panel_rows,
    validate_primary_panel,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._handoff_fixture import (
    candidate_handoff_payload,
)

_PRIMARY_CLASS = "eco1_rt_clade9_plurality25_contact10a_v1"
_BOUNDARY_CLASS = "eco1_rt_clade9_plurality25_contact5a_v1"


def _panel_rows(classes: list[str]) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "design_class_id": design_class_id,
            "selection_candidate_tier": "primary_panel_candidate",
        }
        for index, design_class_id in enumerate(classes, start=1)
    ]


def _candidate_row(
    candidate_id: str,
    *,
    design_class_id: str = _PRIMARY_CLASS,
    tier: str = "primary_panel_candidate",
    na_facing_mutation_count: int,
    proximal_unobserved_mutation_count: int = 0,
    proximal_rare_or_unobserved_mutation_count: int = 0,
    acidic_gain_count: int = 0,
    basic_loss_count: int = 0,
    proline_glycine_gain_count: int = 0,
    substrate_relevant_local_rmsd: float = 1.0,
    msa_fraction: float = 1.0,
    msa_frequency: float = 0.5,
    chemistry_warning_count: int = 0,
    mutation_count_total: int = 100,
    c_terminal_rmsd: float = 1.0,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "sequence_hash": f"sha256:{candidate_id:0<64}"[:71],
        "design_class_id": design_class_id,
        "selection_candidate_tier": tier,
        "primary_panel_candidate": tier == "primary_panel_candidate",
        "primary_panel_failure_reasons_json": "[]"
        if tier == "primary_panel_candidate"
        else '["near_retained_dna_rna_acidic_gain"]',
        "fold_review_class": "strong_fold_preserved",
        "hard_gate_status": "eligible",
        "feasibility_status": "feasible",
        "proximal_review_unobserved_mutation_count": proximal_unobserved_mutation_count,
        "proximal_review_rare_or_unobserved_mutation_count": proximal_rare_or_unobserved_mutation_count,
        "selection_support_profile_id": "ec86_clade9_conservation_v1",
        "selection_support_alt_observed_fraction": msa_fraction,
        "selection_support_alt_frequency_mean": msa_frequency,
        "selection_support_unobserved_mutation_count": 0,
        "nucleic_acid_facing_mutation_count": na_facing_mutation_count,
        "nucleic_acid_facing_charge_delta": 1,
        "nucleic_acid_facing_acidic_gain_count": acidic_gain_count,
        "nucleic_acid_facing_basic_gain_count": 0,
        "nucleic_acid_facing_basic_loss_count": basic_loss_count,
        "nucleic_acid_facing_proline_glycine_gain_count": proline_glycine_gain_count,
        "nucleic_acid_facing_chemistry_warning_count": chemistry_warning_count,
        "nucleic_acid_facing_chemistry_compatible": True,
        "nucleic_acid_facing_chemistry_gate_status": "passed",
        "near_retained_dna_rna_acidic_gain_review_status": "passed",
        "primary_c_terminal_local_rmsd_gate_status": "passed",
        "proximal_msa_support_review_status": "passed",
        "catalytic_or_direct_contact_mutation_count": 0,
        "thumb_contact_track_mutation_count": 0,
        "c_terminal_primer_rna_recognition_mutation_count": 0,
        "distal_scaffold_mutation_count": 3,
        "local_structure_gate_status": "passed",
        "local_structure_threshold_policy_id": "fixture_threshold_policy",
        "local_structure_threshold_failed_region_count": 0,
        "local_structure_max_ca_rmsd_angstrom": max(substrate_relevant_local_rmsd, c_terminal_rmsd),
        "local_structure_substrate_relevant_max_ca_rmsd_angstrom": substrate_relevant_local_rmsd,
        "local_structure_substrate_relevant_max_gate_status": "passed",
        "local_structure_catalytic_initiation_context_ca_rmsd_angstrom": 1.0,
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom": 1.0,
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom": c_terminal_rmsd,
        "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom": 1.0,
        "mean_plddt": 90.0,
        "wt_runtime_ca_rmsd": 1.0,
        "cryoem_mapped_ca_rmsd": 2.0,
        "mutation_count_total": mutation_count_total,
        "sae_window_status": "wt_like_not_used_for_selection",
    }


def test_primary_panel_allows_duplicate_design_classes() -> None:
    rows = _panel_rows([_PRIMARY_CLASS] * PRIMARY_PANEL_SIZE)

    validate_primary_panel(rows, required_panel_size=PRIMARY_PANEL_SIZE)


def test_primary_panel_rejects_wrong_panel_size() -> None:
    rows = _panel_rows([_PRIMARY_CLASS] * (PRIMARY_PANEL_SIZE - 1))

    with pytest.raises(ValueError, match="Selected rows: 5"):
        validate_primary_panel(rows, required_panel_size=PRIMARY_PANEL_SIZE)


def test_primary_choice_prefers_lower_proximal_unsupported_count_before_msa() -> None:
    rows = [
        _candidate_row(
            "high_msa_with_proximal_unsupported_change",
            na_facing_mutation_count=1,
            proximal_unobserved_mutation_count=1,
            msa_fraction=1.0,
        ),
        _candidate_row(
            "lower_msa_no_proximal_unsupported_change",
            na_facing_mutation_count=1,
            proximal_unobserved_mutation_count=0,
            msa_fraction=0.5,
        ),
    ]

    chosen, _nearest_distance = _choose_primary_candidate(
        candidate_rows=rows,
        selected_rows=[],
        sequence_by_id={row["candidate_id"]: "A" * 12 for row in rows},
    )

    assert chosen["candidate_id"] == "lower_msa_no_proximal_unsupported_change"


def test_primary_choice_prefers_lower_chemistry_risk_before_mutation_burden() -> None:
    rows = [
        _candidate_row("no_acidic_gain_broad_near_region", na_facing_mutation_count=90, acidic_gain_count=0),
        _candidate_row("acidic_gain_moderate_near_region", na_facing_mutation_count=40, acidic_gain_count=1),
    ]

    chosen, _nearest_distance = _choose_primary_candidate(
        candidate_rows=rows,
        selected_rows=[],
        sequence_by_id={row["candidate_id"]: "A" * 12 for row in rows},
    )

    assert chosen["candidate_id"] == "no_acidic_gain_broad_near_region"


def test_primary_choice_prefers_lower_substrate_local_rmsd_after_chemistry_ties() -> None:
    rows = [
        _candidate_row("higher_local_rmsd", na_facing_mutation_count=40, substrate_relevant_local_rmsd=2.0),
        _candidate_row("lower_local_rmsd", na_facing_mutation_count=40, substrate_relevant_local_rmsd=1.2),
    ]

    chosen, _nearest_distance = _choose_primary_candidate(
        candidate_rows=rows,
        selected_rows=[],
        sequence_by_id={row["candidate_id"]: "A" * 12 for row in rows},
    )

    assert chosen["candidate_id"] == "lower_local_rmsd"


def test_primary_choice_prefers_mutation_position_dissimilarity_before_rmsd_micro_ties() -> None:
    selected = [_candidate_row("already_selected", na_facing_mutation_count=0)]
    rows = [
        _candidate_row(
            "lower_rmsd_overlapping_mutations",
            na_facing_mutation_count=0,
            substrate_relevant_local_rmsd=1.0,
            c_terminal_rmsd=1.0,
        ),
        _candidate_row(
            "higher_rmsd_distinct_mutations",
            na_facing_mutation_count=0,
            substrate_relevant_local_rmsd=1.6,
            c_terminal_rmsd=1.6,
        ),
    ]

    chosen, _nearest_distance = _choose_primary_candidate(
        candidate_rows=rows,
        selected_rows=selected,
        sequence_by_id={
            "already_selected": "A" * 12,
            "lower_rmsd_overlapping_mutations": "A" * 11 + "C",
            "higher_rmsd_distinct_mutations": "A" * 10 + "CC",
        },
        mutation_tokens_by_id={
            "already_selected": frozenset({"A10G", "L20V"}),
            "lower_rmsd_overlapping_mutations": frozenset({"A10G", "L20V"}),
            "higher_rmsd_distinct_mutations": frozenset({"A30G", "L40V"}),
        },
        mutation_positions_by_id={
            "already_selected": frozenset({10, 20}),
            "lower_rmsd_overlapping_mutations": frozenset({10, 20}),
            "higher_rmsd_distinct_mutations": frozenset({30, 40}),
        },
    )

    assert chosen["candidate_id"] == "higher_rmsd_distinct_mutations"


def test_primary_panel_selection_ignores_boundary_rows_even_with_better_static_values() -> None:
    rows = [
        _candidate_row(
            "boundary_row",
            design_class_id=_BOUNDARY_CLASS,
            tier="boundary_candidate",
            na_facing_mutation_count=1,
            chemistry_warning_count=0,
            mutation_count_total=1,
        ),
        *[
            _candidate_row(
                f"primary_{index}",
                design_class_id=_PRIMARY_CLASS,
                na_facing_mutation_count=10 + index,
                chemistry_warning_count=1,
                mutation_count_total=30 + index,
            )
            for index in range(PRIMARY_PANEL_SIZE)
        ],
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
    assert "boundary_row" not in {row["candidate_id"] for row in panel}
    assert {row["design_class_id"] for row in panel} == {_PRIMARY_CLASS}
    assert {row["selection_candidate_tier"] for row in panel} == {"primary_panel_candidate"}


def test_primary_panel_trace_records_simplified_funnel_without_noop_msa_stage() -> None:
    rows = [
        _candidate_row("primary_kept", na_facing_mutation_count=1),
        _candidate_row(
            "boundary_cterm",
            tier="boundary_candidate",
            na_facing_mutation_count=1,
            c_terminal_rmsd=2.8,
        ),
        {
            **_candidate_row("blocked", tier="not_panel_candidate", na_facing_mutation_count=1),
            "hard_gate_status": "ineligible",
        },
    ]

    trace = build_primary_panel_selection_trace_rows(
        triage_rows=rows,
        panel_rows=[{"candidate_id": "primary_kept"}],
    )

    stage_by_id = {row["stage_id"]: row for row in trace}
    assert stage_by_id["broad_contract_pool"]["remaining_count"] == 2
    assert stage_by_id["primary_panel_candidate_pool"]["remaining_count"] == 1
    assert stage_by_id["global_conservative_diverse_selection"]["remaining_count"] == 1
    assert "boundary_candidate_pool" not in stage_by_id
    assert "primary_proximal_msa_support_gate" not in stage_by_id


def test_handoff_readiness_uses_thread_root_candidate_handoff(tmp_path) -> None:
    selection_root = tmp_path / "outputs/thread/design_classes/selection"
    thread_handoff_path = tmp_path / "outputs/thread/candidate_handoff.yaml"
    selection_root.mkdir(parents=True)
    (selection_root / "candidate_handoff.yaml").write_text("handoff_kind: wrong_local_path\n", encoding="utf-8")

    readiness = build_handoff_readiness(
        selection_root=selection_root,
        panel_rows=_panel_rows([_PRIMARY_CLASS] * PRIMARY_PANEL_SIZE),
        candidate_handoff_path=thread_handoff_path,
    )

    assert readiness["candidate_handoff_path"] == "../../candidate_handoff.yaml"
    assert readiness["candidate_handoff_file_present"] is False
    assert readiness["candidate_handoff_materialized"] is False

    thread_handoff_path.write_text("handoff_kind: rt_only_candidate_handoff\n", encoding="utf-8")

    invalid_readiness = build_handoff_readiness(
        selection_root=selection_root,
        panel_rows=_panel_rows([_PRIMARY_CLASS] * PRIMARY_PANEL_SIZE),
        candidate_handoff_path=thread_handoff_path,
    )
    assert invalid_readiness["candidate_handoff_file_present"] is True
    assert invalid_readiness["candidate_handoff_materialized"] is False

    thread_handoff_path.write_text(yaml.safe_dump(candidate_handoff_payload(), sort_keys=False), encoding="utf-8")

    assert (
        build_handoff_readiness(
            selection_root=selection_root,
            panel_rows=_panel_rows([_PRIMARY_CLASS] * PRIMARY_PANEL_SIZE),
            candidate_handoff_path=thread_handoff_path,
        )["candidate_handoff_materialized"]
        is True
    )
