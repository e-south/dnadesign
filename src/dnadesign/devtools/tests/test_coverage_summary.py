"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_coverage_summary.py

Tests for CI coverage summary payload generation used by quality score inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.devtools.coverage_summary import build_coverage_summary_payload, main


def test_build_coverage_summary_payload_computes_tool_rows_and_overall() -> None:
    coverage_json = {
        "files": {
            "src/dnadesign/usr/src/io.py": {"summary": {"covered_lines": 9, "num_statements": 10}},
            "src/dnadesign/densegen/src/main.py": {"summary": {"covered_lines": 3, "num_statements": 10}},
            "src/dnadesign/densegen/tests/test_main.py": {"summary": {"covered_lines": 1, "num_statements": 1}},
        }
    }
    baseline = {"densegen": 40.0, "usr": 80.0}

    payload = build_coverage_summary_payload(
        coverage_data=coverage_json, baseline=baseline, selected_tools=set(baseline)
    )

    assert payload["gate_total_tools"] == 2
    assert payload["gate_passed_tools"] == 1
    assert payload["overall_core"] == 60.0
    assert payload["tools"] == {
        "densegen": {"actual": 30.0, "baseline": 40.0, "gate": "fail"},
        "usr": {"actual": 90.0, "baseline": 80.0, "gate": "pass"},
    }


def test_main_scopes_summary_to_selected_tools(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    output_path = tmp_path / "coverage-summary.json"
    coverage_path.write_text(
        json.dumps(
            {
                "files": {
                    "src/dnadesign/usr/src/io.py": {"summary": {"covered_lines": 90, "num_statements": 100}},
                    "src/dnadesign/densegen/src/main.py": {"summary": {"covered_lines": 0, "num_statements": 100}},
                }
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text('{"usr": 80.0, "densegen": 70.0}', encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
            "--only-tools",
            "usr",
            "--output-json",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["gate_total_tools"] == 1
    assert payload["tools"] == {"usr": {"actual": 90.0, "baseline": 80.0, "gate": "pass"}}


def test_main_fails_when_only_tools_is_unknown(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    output_path = tmp_path / "coverage-summary.json"
    coverage_path.write_text(
        json.dumps({"files": {"src/dnadesign/usr/src/io.py": {"summary": {"covered_lines": 1, "num_statements": 1}}}}),
        encoding="utf-8",
    )
    baseline_path.write_text('{"usr": 0.0}', encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
            "--only-tools",
            "usr,ghost",
            "--output-json",
            str(output_path),
        ]
    )

    assert rc == 1
