"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_captured_source.py

Tests for bounded, descriptor-captured render sources.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pytest

from dnadesign.baserender.src.io.captured_source import CapturedSource
from dnadesign.baserender.src.reporting import RunReport


def test_verify_unchanged_reuses_capture_limit_after_content_release(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"data")
    captured = CapturedSource.capture(source, max_bytes=4).without_content()
    source.write_bytes(b"oversized")

    with pytest.raises(ValueError, match="Render source changed during execution") as error:
        captured.verify_unchanged()

    assert str(error.value.__cause__) == f"Render source exceeds the maximum of 4 bytes: {source}"


def test_run_report_applies_byte_limit_to_every_captured_source(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_bytes(b"data")
    selection_path = tmp_path / "selection.csv"
    selection_path.write_bytes(b"oversized")
    report = RunReport(
        job_name="bounded-sources",
        input_path=str(input_path),
        selection_path=str(selection_path),
    )

    with pytest.raises(ValueError, match="maximum of 4 bytes") as error:
        report.capture_source_evidence(max_bytes=4)

    assert str(selection_path) in str(error.value)
