"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_architecture.py

Architecture guards for the study-owned metric screen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy")


def _module_body_line_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0] == '"""':
        closing_line = next(index for index, line in enumerate(lines[1:], start=1) if line == '"""')
        return len(lines) - closing_line - 1
    return len(lines)


def test_evaluation_and_reporting_do_not_import_runtime() -> None:
    offenders: list[str] = []
    for layer in ("evaluation", "reporting"):
        for path in sorted((PACKAGE / layer).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and "runtime" in node.module.split("."):
                    offenders.append(f"{path}:{node.lineno}:{node.module}")
    assert offenders == []


def test_sfxi_and_response_label_contracts_stay_disjoint() -> None:
    prohibited = {
        "response_labels_as_sfxi_comparator",
        "sfxi_comparison_vec8",
        "build_sfxi_comparison_rows",
        "snapshot_y",
        "snapshot_vec8",
    }
    offenders: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        matches = sorted(token for token in prohibited if token in text)
        if matches:
            offenders[str(path.relative_to(PACKAGE))] = matches
    assert offenders == {}


def test_response_candidate_identity_does_not_consume_sfxi_label_sources() -> None:
    source = (PACKAGE / "runtime/candidate_identity.py").read_text(encoding="utf-8").lower()
    assert "label_source" not in source
    assert "sfxi" not in source
    audit_source = (PACKAGE / "runtime/audit.py").read_text(encoding="utf-8")
    assert "measurement_selection=measurement_selection.rows" in audit_source
    assert "label_sources=label_sources" not in audit_source
    assert "label_ids=response_ids" in audit_source


def test_response_model_modules_stay_semantically_bounded() -> None:
    limits = {
        "evaluation/model_screen.py": 360,
        "evaluation/grouped_models.py": 240,
        "evaluation/greedy_support.py": 160,
        "evaluation/model_representations.py": 180,
        "evaluation/multistate_behavior_cohort.py": 230,
        "evaluation/multistate_behavior_comparison.py": 250,
        "evaluation/multistate_behavior_allocation.py": 210,
        "evaluation/multistate_behavior_cardinality.py": 80,
        "evaluation/multistate_behavior_event.py": 150,
        "evaluation/multistate_behavior_face_validity.py": 110,
        "evaluation/multistate_behavior_gate_protocol.py": 230,
        "evaluation/multistate_behavior_grouped_validation.py": 430,
        "evaluation/multistate_behavior_normalization.py": 390,
        "evaluation/multistate_behavior_normalization_protocol.py": 130,
        "evaluation/multistate_behavior_normalization_sensitivity.py": 210,
        "evaluation/multistate_behavior_protocol.py": 350,
        "evaluation/multistate_behavior_protocol_fields.py": 120,
        "evaluation/multistate_behavior_rows.py": 110,
        "evaluation/multistate_behavior_rmf_replay.py": 140,
        "evaluation/multistate_behavior_shadow.py": 260,
        "evaluation/multistate_behavior_stability.py": 130,
        "runtime/audit.py": 360,
        "runtime/campaign_calibration.py": 120,
        "runtime/model_evidence_manifest.py": 200,
        "runtime/multistate_behavior_bundle_contract.py": 110,
        "runtime/multistate_behavior_audit_verification.py": 100,
        "runtime/multistate_behavior_bundle_verification.py": 250,
        "runtime/multistate_behavior_completion.py": 220,
        "runtime/multistate_behavior_completion_verification.py": 310,
        "runtime/multistate_behavior_censor.py": 100,
        "runtime/multistate_behavior_event_verification.py": 110,
        "runtime/multistate_behavior_grouped_verification.py": 330,
        "runtime/multistate_behavior_json.py": 60,
        "runtime/multistate_behavior_normalization_verification.py": 180,
        "runtime/multistate_behavior_allocation_verification.py": 130,
        "runtime/multistate_behavior_decision.py": 220,
        "runtime/multistate_behavior_decision_verification.py": 270,
        "runtime/multistate_behavior_prediction.py": 320,
        "runtime/multistate_behavior_publication.py": 230,
        "runtime/multistate_behavior_record_fields.py": 110,
        "runtime/multistate_behavior_reference.py": 110,
        "runtime/multistate_behavior_run_contract.py": 270,
        "runtime/multistate_behavior_semantic_verification.py": 200,
        "runtime/multistate_behavior_shadow.py": 210,
        "runtime/multistate_behavior_shadow_scoring.py": 130,
        "runtime/multistate_behavior_sensitivity_verification.py": 170,
        "runtime/multistate_behavior_source_equivalence.py": 200,
        "runtime/multistate_behavior_sources.py": 120,
        "runtime/multistate_behavior_source_receipt.py": 150,
        "runtime/multistate_behavior_table_coverage.py": 250,
        "runtime/multistate_behavior_table_derivations.py": 220,
        "runtime/multistate_behavior_table_provenance.py": 210,
        "runtime/multistate_behavior_table_schema.py": 270,
        "runtime/multistate_behavior_table_schema_completion.py": 190,
        "runtime/multistate_behavior_table_schema_evidence.py": 260,
        "runtime/publication.py": 230,
        "runtime/selected_reader_rows.py": 120,
        "runtime/response_screen.py": 280,
        "runtime/response_screen_publication.py": 220,
        "runtime/review_bundle.py": 200,
        "reporting/notebook.py": 240,
        "reporting/notebook_copy.py": 80,
        "reporting/notebook_summary.py": 120,
        "reporting/matrix_annotations.py": 120,
        "reporting/plot_style.py": 100,
        "reporting/plot_style_primitives.py": 100,
        "reporting/plot_vocabulary.py": 180,
        "reporting/response_assay_plots.py": 300,
        "reporting/response_model_plots.py": 240,
        "reporting/multistate_behavior_plots.py": 200,
        "reporting/multistate_behavior_plot_labels.py": 80,
        "reporting/multistate_behavior_plot_style.py": 70,
        "reporting/multistate_behavior_report.py": 280,
        "reporting/rmf_contract_plot.py": 100,
        "model_evidence/cli.py": 100,
        "model_evidence/contracts.py": 80,
        "model_evidence/evaluator_protocol.py": 100,
        "model_evidence/fields.py": 140,
        "model_evidence/json_io.py": 80,
        "model_evidence/projection.py": 240,
        "model_evidence/protocol_projection.py": 120,
        "model_evidence/source_evidence.py": 190,
        "model_evidence/storage.py": 180,
        "model_evidence/verification.py": 200,
    }
    observed = {relative: _module_body_line_count(PACKAGE / relative) for relative in limits}
    assert {path: lines for path, lines in observed.items() if lines > limits[path]} == {}
    assert not (PACKAGE / "reporting/response_metric_plots.py").exists()
