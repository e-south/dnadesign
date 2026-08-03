"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/contracts/test_protocol.py

Tests the declared metastudy protocol and dependency boundary.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
    validate_decision_payload,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts import (
    canonical_digest,
)

from .._builders import (
    HIGH_ANCHOR,
    KINETIC_IDS,
    LOW_ANCHOR,
)


def test_predeclared_protocol_is_exact_and_has_no_weighted_score() -> None:
    assert DEFAULT_PROTOCOL.protocol_id == "rt_lnrna_reporter_response_metastudy.v3"
    assert DEFAULT_PROTOCOL.primary_dose_uM == 500.0
    assert DEFAULT_PROTOCOL.sensitivity_doses_uM == (5.0, 50.0)
    assert DEFAULT_PROTOCOL.candidate_windows_h == (
        (4.0, 8.0),
        (6.0, 10.0),
        (8.0, 12.0),
        (10.0, 14.0),
        (12.0, 16.0),
    )
    assert DEFAULT_PROTOCOL.endpoint_sensitivity_h == (8.0, 10.0, 12.0, 14.0, 16.0)
    assert DEFAULT_PROTOCOL.centered_window_sensitivity_widths_h == (2.0, 6.0)
    assert DEFAULT_PROTOCOL.growth_phase_slope_window_h == 1.0
    assert DEFAULT_PROTOCOL.growth_phase_scale_quantile == 0.9
    assert DEFAULT_PROTOCOL.growth_phase_minimum_slope_points == 4
    assert DEFAULT_PROTOCOL.growth_phase_start_minimum == 0.5
    assert DEFAULT_PROTOCOL.growth_phase_end_minimum == 0.1
    assert DEFAULT_PROTOCOL.growth_phase_end_maximum == 0.6
    assert DEFAULT_PROTOCOL.minimum_kinetic_experiments == 7
    assert DEFAULT_PROTOCOL.planned_kinetic_experiments == 8
    assert DEFAULT_PROTOCOL.anchor_subject_order == (LOW_ANCHOR, HIGH_ANCHOR)
    assert DEFAULT_PROTOCOL.planned_anchor_experiment_ids == (
        KINETIC_IDS[0],
        KINETIC_IDS[1],
        KINETIC_IDS[3],
        KINETIC_IDS[4],
        KINETIC_IDS[5],
    )
    assert DEFAULT_PROTOCOL.reference_panel_target_ordered_acquisitions == 4
    assert DEFAULT_PROTOCOL.planned_anchor_acquisitions == 5
    assert DEFAULT_PROTOCOL.loo_same_or_adjacent_target_fraction == 0.75
    assert DEFAULT_PROTOCOL.selection_order[0] == "require_active_to_decelerating_growth_phase"
    assert "weight" not in repr(DEFAULT_PROTOCOL).lower()


def test_metastudy_has_no_reader_opal_or_historical_spop_import_dependency() -> None:
    study_unit = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "reporter_response" / "metastudy" / "__init__.py").is_file()
        and (parent / "tests" / "reporter_response").is_dir()
    )
    package = study_unit / "reporter_response" / "metastudy"
    forbidden_import_roots = {"reader", "reader_workbench", "opal"}
    paths = tuple(package.rglob("*.py"))
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "spop" not in text.lower()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not ({alias.name.split(".")[0] for alias in node.names} & forbidden_import_roots)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_import_roots


def test_checked_in_protocol_and_live_descriptive_selection_match_runtime_contracts() -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts import (
        protocol_digest,
    )

    study_root = next(
        parent / "docs/studies/rt_lnrna_sponging_construct_triage"
        for parent in Path(__file__).resolve().parents
        if (parent / "docs/studies/rt_lnrna_sponging_construct_triage").is_dir()
    )
    docs = study_root / "contexts/reporter-response-metastudy"
    protocol_payload = yaml.safe_load((docs / "protocol.yaml").read_text(encoding="utf-8"))
    expected_protocol = json.loads(json.dumps(asdict(DEFAULT_PROTOCOL)))
    assert protocol_payload == expected_protocol
    assert canonical_digest(protocol_payload) == protocol_digest()

    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
        state as operator_state,
    )

    state = yaml.safe_load((docs / "metastudy-state.yaml").read_text(encoding="utf-8"))
    operator_state.validate_state_payload(state)
    snapshot = state["readiness"]
    decision_payload = state["decision"]
    validate_decision_payload(decision_payload)
    assert decision_payload["readiness"] == {
        "selected_experiment_count": snapshot["selected_experiment_count"],
        "ready_experiment_count": snapshot["ready_experiment_count"],
        "ready_experiment_ids": snapshot["ready_experiment_ids"],
        "blocked_experiment_ids": snapshot["blocked_experiment_ids"],
        "receipt_digest": snapshot["source_identity"]["normalized_full_receipt_digest"],
    }
    assert decision_payload["status"] == "selected"
    assert decision_payload["selection_use"] == "descriptive_comparison"
    assert decision_payload["evidence_grade"] == "provisional_descriptive"
    assert decision_payload["selected_reduction"] == [6.0, 10.0]
    assert decision_payload["blockers"] == []
    assert len(decision_payload["evaluations"]) == len(DEFAULT_PROTOCOL.candidate_windows_h)
    assert {row["eligible_experiment_count"] for row in decision_payload["evaluations"]} == {8}
    eligible = [row for row in decision_payload["evaluations"] if row["eligible"]]
    assert [row["reduction"] for row in eligible] == [[6.0, 10.0]]
    assert state["objective_readiness"] == {
        "contract_id": "rt_lnrna_reporter_response_objective_readiness.v3",
        "status": "blocked",
        "objective_id": None,
        "blockers": [
            "constrained_objective_not_defined",
            "biological_replicate_uncertainty_not_estimable",
            "od_linearity_not_validated",
        ],
    }
    projection = state["acquisition_projection"]
    assert projection["selected_reduction"] == [6.0, 10.0]
    assert len(projection["coordinates"]) == 65
    assert {row["reduction_id"] for row in projection["coordinates"]} == {"window-6-10h"}

    route_text = (study_root / "routes/README.md").read_text(encoding="utf-8")
    assert "6-10 h reduction selected as `provisional_descriptive`" in route_text
    assert "objective readiness remains blocked" in route_text
    assert "0/8 selected kinetic Reader experiments" not in route_text
