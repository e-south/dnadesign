"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_tool_coverage.py

Tests for per-tool coverage aggregation and baseline regression checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.devtools.tool_coverage import (
    CoverageRegression,
    find_regressions,
    main,
    summarize_tool_coverage,
)


def test_summarize_tool_coverage_aggregates_by_tool() -> None:
    coverage_data = {
        "files": {
            "src/dnadesign/usr/src/a.py": {"summary": {"covered_lines": 90, "num_statements": 100}},
            "src/dnadesign/usr/src/b.py": {"summary": {"covered_lines": 10, "num_statements": 20}},
            "src/dnadesign/densegen/src/main.py": {"summary": {"covered_lines": 5, "num_statements": 10}},
            "src/dnadesign/densegen/tests/test_main.py": {"summary": {"covered_lines": 30, "num_statements": 30}},
            "src/other_pkg/ignored.py": {"summary": {"covered_lines": 100, "num_statements": 100}},
        }
    }

    by_tool = summarize_tool_coverage(
        coverage_data,
        tool_names={"densegen", "opal", "usr"},
    )

    assert by_tool["usr"] == 83.33
    assert by_tool["densegen"] == 50.0
    assert by_tool["opal"] == 0.0


def test_summarize_tool_coverage_handles_windows_style_paths() -> None:
    coverage_data = {
        "files": {
            r"src\dnadesign\usr\src\io.py": {"summary": {"covered_lines": 9, "num_statements": 10}},
        }
    }

    by_tool = summarize_tool_coverage(
        coverage_data,
        tool_names={"usr"},
    )

    assert by_tool["usr"] == 90.0


def test_find_regressions_reports_only_tools_below_baseline() -> None:
    regressions = find_regressions(
        baseline={"densegen": 60.0, "opal": 0.0, "usr": 80.0},
        actual={"densegen": 59.9, "opal": 0.0, "usr": 82.0},
    )

    assert regressions == [CoverageRegression(tool="densegen", baseline=60.0, actual=59.9)]


def test_main_exits_nonzero_on_regression(tmp_path: Path) -> None:
    coverage_json = {
        "files": {"src/dnadesign/usr/src/io.py": {"summary": {"covered_lines": 50, "num_statements": 100}}}
    }
    baseline_json = {"usr": 80.0}

    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    coverage_path.write_text(json.dumps(coverage_json), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline_json), encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
        ]
    )

    assert rc == 1


def test_main_exits_zero_when_meeting_baseline(tmp_path: Path) -> None:
    coverage_json = {
        "files": {"src/dnadesign/usr/src/io.py": {"summary": {"covered_lines": 90, "num_statements": 100}}}
    }
    baseline_json = {"usr": 80.0, "densegen": 0.0}

    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    coverage_path.write_text(json.dumps(coverage_json), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline_json), encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
        ]
    )

    assert rc == 0


def test_main_scopes_baselines_when_only_tools_is_set(tmp_path: Path) -> None:
    coverage_json = {
        "files": {"src/dnadesign/usr/src/io.py": {"summary": {"covered_lines": 90, "num_statements": 100}}}
    }
    baseline_json = {"usr": 80.0, "densegen": 70.0}

    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    coverage_path.write_text(json.dumps(coverage_json), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline_json), encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
            "--only-tools",
            "usr",
        ]
    )

    assert rc == 0


def test_main_fails_when_only_tools_has_unknown_tool(tmp_path: Path) -> None:
    coverage_json = {
        "files": {"src/dnadesign/usr/src/io.py": {"summary": {"covered_lines": 90, "num_statements": 100}}}
    }
    baseline_json = {"usr": 80.0}

    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    coverage_path.write_text(json.dumps(coverage_json), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline_json), encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
            "--only-tools",
            "usr,opal",
        ]
    )

    assert rc == 1


def test_main_fails_for_invalid_coverage_json(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    coverage_path.write_text("{", encoding="utf-8")
    baseline_path.write_text('{"usr": 0.0}', encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
        ]
    )

    assert rc == 1


def test_main_fails_for_out_of_range_baseline_value(tmp_path: Path) -> None:
    coverage_json = {
        "files": {"src/dnadesign/usr/src/io.py": {"summary": {"covered_lines": 90, "num_statements": 100}}}
    }
    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    coverage_path.write_text(json.dumps(coverage_json), encoding="utf-8")
    baseline_path.write_text('{"usr": 101.0}', encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
        ]
    )

    assert rc == 1


def test_main_fails_for_missing_files_payload(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    coverage_path.write_text('{"meta": {"format": 3}}', encoding="utf-8")
    baseline_path.write_text('{"usr": 0.0}', encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
        ]
    )

    assert rc == 1


def test_main_fails_for_invalid_file_summary_payload(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "tool-coverage-baseline.json"
    coverage_path.write_text(
        '{"files": {"src/dnadesign/usr/src/io.py": {"summary": {"covered_lines": "bad", "num_statements": 10}}}}',
        encoding="utf-8",
    )
    baseline_path.write_text('{"usr": 0.0}', encoding="utf-8")

    rc = main(
        [
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
        ]
    )

    assert rc == 1
