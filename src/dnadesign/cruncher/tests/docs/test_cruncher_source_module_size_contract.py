"""
--------------------------------------------------------------------------------
cruncher project
src/dnadesign/cruncher/tests/docs/test_cruncher_source_module_size_contract.py

Contract checks that Cruncher source modules stay below the monolith threshold.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
MAX_SOURCE_LINES = 1200


def test_cruncher_source_modules_stay_below_monolith_threshold() -> None:
    offenders: list[tuple[str, int]] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        if line_count > MAX_SOURCE_LINES:
            offenders.append((str(path.relative_to(SOURCE_ROOT.parent)), line_count))
    assert not offenders, f"Cruncher modules exceed {MAX_SOURCE_LINES} lines: {offenders}"
