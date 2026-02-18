"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_pytest_gate.py

Tests for JUnit-based pytest execution guards used in CI lanes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.devtools.pytest_gate import evaluate_junit_report, main


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_evaluate_junit_report_passes_when_non_skipped_tests_exist(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    _write(
        report,
        ("<testsuite tests='5' failures='0' errors='0' skipped='2'><testcase classname='x' name='a'/></testsuite>"),
    )

    summary = evaluate_junit_report(report_path=report)
    assert summary.tests == 5
    assert summary.skipped == 2
    assert summary.non_skipped == 3


def test_evaluate_junit_report_fails_when_all_tests_skipped(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    _write(
        report,
        (
            "<testsuite tests='4' failures='0' errors='0' skipped='4'>"
            "<testcase classname='x' name='a'><skipped/></testcase>"
            "</testsuite>"
        ),
    )

    try:
        evaluate_junit_report(report_path=report)
    except ValueError as exc:
        assert "all collected tests were skipped" in str(exc)
    else:
        raise AssertionError("Expected ValueError when all tests are skipped.")


def test_evaluate_junit_report_fails_when_no_tests_collected(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    _write(report, "<testsuite tests='0' failures='0' errors='0' skipped='0'></testsuite>")

    try:
        evaluate_junit_report(report_path=report)
    except ValueError as exc:
        assert "no tests were collected" in str(exc)
    else:
        raise AssertionError("Expected ValueError when no tests are collected.")


def test_evaluate_junit_report_aggregates_testsuites_root(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    _write(
        report,
        (
            "<testsuites>"
            "<testsuite tests='2' failures='0' errors='0' skipped='1'></testsuite>"
            "<testsuite tests='3' failures='0' errors='0' skipped='1'></testsuite>"
            "</testsuites>"
        ),
    )

    summary = evaluate_junit_report(report_path=report)
    assert summary.tests == 5
    assert summary.skipped == 2
    assert summary.non_skipped == 3


def test_main_fails_when_report_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.xml"
    rc = main(["--junit-xml", str(missing), "--lane-name", "external-integration"])
    assert rc == 1


def test_main_passes_for_valid_report(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    _write(
        report,
        ("<testsuite tests='3' failures='0' errors='0' skipped='1'><testcase classname='x' name='a'/></testsuite>"),
    )
    rc = main(["--junit-xml", str(report), "--lane-name", "external-integration"])
    assert rc == 0


def test_evaluate_junit_report_fails_when_required_tool_has_no_executed_tests(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    _write(
        report,
        (
            "<testsuite tests='2' failures='0' errors='0' skipped='0'>"
            "<testcase classname='src.dnadesign.densegen.tests.test_one' name='test_a'/>"
            "<testcase classname='src.dnadesign.densegen.tests.test_two' name='test_b'/>"
            "</testsuite>"
        ),
    )

    try:
        evaluate_junit_report(report_path=report, required_tools={"densegen", "cruncher"})
    except ValueError as exc:
        assert "required tool(s) without executed non-skipped tests: cruncher" in str(exc)
    else:
        raise AssertionError("Expected ValueError when a required tool has no executed tests.")


def test_evaluate_junit_report_passes_when_all_required_tools_have_executed_tests(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    _write(
        report,
        (
            "<testsuite tests='2' failures='0' errors='0' skipped='0'>"
            "<testcase classname='src.dnadesign.densegen.tests.test_one' name='test_a'/>"
            "<testcase classname='src.dnadesign.cruncher.tests.test_two' name='test_b'/>"
            "</testsuite>"
        ),
    )

    summary = evaluate_junit_report(report_path=report, required_tools={"densegen", "cruncher"})
    assert summary.non_skipped == 2
