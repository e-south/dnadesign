"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_screen_dashboard.py

Screen dashboard behavior for non-terminal output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from dnadesign.densegen.src.core.pipeline.progress import _build_screen_dashboard, _ScreenDashboard


def test_screen_dashboard_static_output_for_non_terminal() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    dashboard = _ScreenDashboard(console=console, refresh_seconds=1.0)
    panel = _build_screen_dashboard(
        source_label="demo",
        plan_name="plan",
        bar="[----]",
        generated=1,
        quota=2,
        pct=50.0,
        local_generated=1,
        local_target=2,
        library_index=1,
        cr_now=None,
        cr_avg=None,
        resamples=0,
        dup_out=0,
        dup_sol=0,
        fails=0,
        stalls=0,
        failure_totals=None,
        tf_usage={},
        tfbs_usage={},
        diversity_label="-",
        show_tfbs=False,
        show_solutions=False,
        sequence_preview=None,
    )

    dashboard.update(panel)
    dashboard.update(panel)
    assert buffer.getvalue() == ""

    dashboard.close()
    output = buffer.getvalue()
    assert output.count("DenseGen progress") == 1
