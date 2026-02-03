"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_progress.py

Stage-A progress line formatting tests.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ABOUTME: Tests Stage-A progress line formatting expectations.
# ABOUTME: Ensures progress output avoids non-monotonic percentage noise.
from __future__ import annotations

from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_progress import _format_pwm_progress_line


def test_progress_line_omits_percent_and_bar() -> None:
    line = _format_pwm_progress_line(
        motif_id="M1",
        backend="fimo",
        phase="mining",
        generated=100,
        target=1000,
        accepted=50,
        accepted_target=200,
        batch_index=2,
        batch_total=10,
        elapsed=1.0,
        target_fraction=0.001,
        tier_fractions=[0.001, 0.01, 0.09],
        tier_yield="1/2/3",
    )
    assert "%" not in line
    assert "[" not in line
    assert "]" not in line
    assert "gen 100/1000" in line
