"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_tool_agnostic_hardening.py

Static hardening tests that enforce tool-agnostic API and contract-driven parsing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text().lower()


def test_public_api_module_avoids_tool_specific_input_assumptions() -> None:
    root = Path(__file__).resolve().parents[1]
    api_text = _read(root / "src" / "api.py")
    assert "densegen" not in api_text
    assert "sigma70" not in api_text


def test_job_parser_avoids_adapter_kind_branching() -> None:
    root = Path(__file__).resolve().parents[1]
    parser_text = _read(root / "src" / "config" / "cruncher_showcase_job.py")
    assert 'if kind == "densegen_tfbs"' not in parser_text
    assert 'if kind == "cruncher_best_window"' not in parser_text
    assert "densegen_tfbs" not in parser_text
    assert "cruncher_best_window" not in parser_text
