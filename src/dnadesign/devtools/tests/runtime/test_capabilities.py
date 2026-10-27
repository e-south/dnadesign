"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/runtime/test_capabilities.py

Capability evidence distinguishes successful execution from skipped coverage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("result", "expected"),
    [("", "passed"), ("<skipped/>", "not_run"), ("<failure/>", "failed"), ("<error/>", "failed")],
)
def test_video_capability_uses_test_outcome(tmp_path: Path, result: str, expected: str) -> None:
    from dnadesign.devtools.runtime.capabilities import collect_capabilities

    report = tmp_path / "core.xml"
    report.write_text(
        '<testsuites><testsuite><testcase classname="src.dnadesign.baserender.tests.test_outputs_explicit" '
        f'name="test_video_output_does_not_produce_images">{result}</testcase></testsuite></testsuites>'
    )
    evidence = collect_capabilities(report, revision="abc", run_id=7, run_attempt=2)
    assert evidence["capabilities"]["video_output"]["status"] == expected
    assert evidence["capabilities"]["gpu_inference"]["status"] == "not_run"
    assert (evidence["revision"], evidence["run_id"], evidence["run_attempt"]) == ("abc", 7, 2)


def test_absent_video_test_is_not_a_pass(tmp_path: Path) -> None:
    from dnadesign.devtools.runtime.capabilities import collect_capabilities

    report = tmp_path / "core.xml"
    report.write_text('<testsuites><testsuite><testcase name="unrelated"/></testsuite></testsuites>')
    assert (
        collect_capabilities(report, revision="abc", run_id=7, run_attempt=1)["capabilities"]["video_output"]["status"]
        == "not_run"
    )


def test_missing_report_does_not_claim_capability_execution(tmp_path: Path) -> None:
    from dnadesign.devtools.runtime.capabilities import collect_capabilities

    result = collect_capabilities(tmp_path / "missing.xml", revision="abc", run_id=7, run_attempt=1)
    assert result["capabilities"]["video_output"]["status"] == "not_run"
