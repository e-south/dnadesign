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
        "evaluation/model_representations.py": 180,
        "runtime/audit.py": 360,
        "runtime/reader_display_contract.py": 80,
        "runtime/reader_response_bundle.py": 320,
        "runtime/response_screen.py": 280,
        "runtime/review_bundle.py": 200,
        "reporting/notebook.py": 240,
        "reporting/notebook_summary.py": 120,
        "reporting/matrix_annotations.py": 120,
        "reporting/plot_style.py": 100,
        "reporting/plot_vocabulary.py": 180,
        "reporting/response_assay_plots.py": 300,
        "reporting/response_model_plots.py": 240,
        "reporting/rmf_contract_plot.py": 100,
    }
    observed = {relative: _module_body_line_count(PACKAGE / relative) for relative in limits}
    assert {path: lines for path, lines in observed.items() if lines > limits[path]} == {}
    assert not (PACKAGE / "reporting/response_metric_plots.py").exists()
