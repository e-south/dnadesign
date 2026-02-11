"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_stage_a_progress.py

Stage-A progress line formatting tests.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ABOUTME: Tests Stage-A progress line formatting expectations.
# ABOUTME: Ensures progress output avoids non-monotonic percentage noise.
from __future__ import annotations

from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_progress import (
    StageAProgressState,
    _format_pwm_progress_line,
    _stage_a_live_render,
)


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


def _render_headers(state: dict[str, StageAProgressState]) -> list[str]:
    panel = _stage_a_live_render(state)
    table = panel.renderable
    return [str(col.header) for col in table.columns]


def test_stage_a_live_render_pwm_columns() -> None:
    state = {
        "pwm": StageAProgressState(
            motif_id="lexA",
            backend="fimo",
            phase="mining",
            generated=100,
            target=1000,
            accepted=50,
            accepted_target=200,
            target_fraction=None,
            tier_fractions=[0.001, 0.01, 0.09],
            elapsed=1.0,
            batch_index=1,
            batch_total=10,
            show_tier_yield=True,
            show_accept_rate=False,
            show_rejects=False,
        )
    }
    assert _render_headers(state) == [
        "motif",
        "phase",
        "generated/limit",
        "eligible_unique",
        "tier yield",
        "batch",
        "elapsed",
        "rate",
    ]


def test_stage_a_live_render_background_columns() -> None:
    state = {
        "background": StageAProgressState(
            motif_id="neutral_bg",
            backend="background",
            phase="background",
            generated=100,
            target=1000,
            accepted=20,
            accepted_target=200,
            target_fraction=None,
            tier_fractions=None,
            elapsed=1.0,
            batch_index=1,
            batch_total=10,
            show_tier_yield=False,
            show_accept_rate=True,
            show_rejects=True,
            reject_fimo=2,
            reject_kmer=3,
            reject_gc=4,
            reject_dup=5,
        )
    }
    assert _render_headers(state) == [
        "motif",
        "phase",
        "generated/limit",
        "eligible_unique",
        "accept %",
        "rejects fimo/kmer/gc/dup",
        "batch",
        "elapsed",
        "rate",
    ]
