"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_quality_score.py

Tests for CI-generated quality score inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.devtools.quality_score import build_quality_score_inputs, main


def test_build_quality_score_inputs_uses_coverage_summary_and_baseline() -> None:
    coverage_summary = {
        "overall_core": 65.0,
        "gate_passed_tools": 2,
        "gate_total_tools": 3,
        "tools": {
            "usr": {"actual": 57.0, "baseline": 57.0, "gate": "pass"},
            "notify": {"actual": 80.0, "baseline": 80.0, "gate": "pass"},
            "aligner": {"actual": 0.0, "baseline": 0.0, "gate": "pass"},
        },
    }
    baseline = {"usr": 57.0, "notify": 80.0, "aligner": 0.0}

    payload = build_quality_score_inputs(
        coverage_summary=coverage_summary,
        baseline=baseline,
        core_lane_result="success",
        external_integration_lane_result="skipped",
        publish_lane_result="success",
    )

    assert payload["lanes"]["core"] == "success"
    assert payload["lanes"]["external_integration"] == "skipped"
    assert payload["lanes"]["publish"] == "success"
    assert payload["coverage"]["overall_core"] == 65.0
    assert payload["coverage"]["gate_total_tools"] == 3
    assert payload["tools"][0]["tool"] == "aligner"
    assert payload["tools"][0]["actual"] == 0.0


def test_build_quality_score_inputs_fails_on_tool_inventory_mismatch() -> None:
    coverage_summary = {
        "overall_core": 65.0,
        "gate_passed_tools": 1,
        "gate_total_tools": 1,
        "tools": {"usr": {"actual": 57.0, "baseline": 57.0, "gate": "pass"}},
    }
    baseline = {"usr": 57.0, "notify": 80.0}

    try:
        build_quality_score_inputs(
            coverage_summary=coverage_summary,
            baseline=baseline,
            core_lane_result="success",
            external_integration_lane_result="success",
            publish_lane_result="success",
        )
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for tool mismatch")


def test_build_quality_score_inputs_fails_on_missing_actual_value() -> None:
    coverage_summary = {
        "overall_core": 65.0,
        "gate_passed_tools": 1,
        "gate_total_tools": 1,
        "tools": {
            "usr": {"baseline": 57.0, "gate": "pass"},
        },
    }
    baseline = {"usr": 57.0}

    try:
        build_quality_score_inputs(
            coverage_summary=coverage_summary,
            baseline=baseline,
            core_lane_result="success",
            external_integration_lane_result="success",
            publish_lane_result="success",
        )
    except ValueError as exc:
        assert "missing required field 'actual'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing actual field")


def test_main_writes_quality_score_inputs(tmp_path: Path) -> None:
    coverage_summary_path = tmp_path / "coverage-summary.json"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "quality-score-inputs.json"

    coverage_summary_path.write_text(
        json.dumps(
            {
                "overall_core": 70.0,
                "gate_passed_tools": 1,
                "gate_total_tools": 1,
                "tools": {
                    "usr": {"actual": 70.0, "baseline": 57.0, "gate": "pass"},
                },
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(json.dumps({"usr": 57.0}), encoding="utf-8")

    rc = main(
        [
            "--coverage-summary-json",
            str(coverage_summary_path),
            "--baseline-json",
            str(baseline_path),
            "--core-lane-result",
            "success",
            "--external-integration-lane-result",
            "success",
            "--publish-lane-result",
            "success",
            "--output-json",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["coverage"]["overall_core"] == 70.0
    assert payload["tools"][0]["tool"] == "usr"


def test_main_fails_for_invalid_lane_result(tmp_path: Path) -> None:
    coverage_summary_path = tmp_path / "coverage-summary.json"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "quality-score-inputs.json"

    coverage_summary_path.write_text(
        json.dumps(
            {
                "overall_core": 70.0,
                "gate_passed_tools": 1,
                "gate_total_tools": 1,
                "tools": {
                    "usr": {"actual": 70.0, "baseline": 57.0, "gate": "pass"},
                },
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(json.dumps({"usr": 57.0}), encoding="utf-8")

    rc = main(
        [
            "--coverage-summary-json",
            str(coverage_summary_path),
            "--baseline-json",
            str(baseline_path),
            "--core-lane-result",
            "maybe",
            "--external-integration-lane-result",
            "success",
            "--publish-lane-result",
            "success",
            "--output-json",
            str(output_path),
        ]
    )

    assert rc == 1
