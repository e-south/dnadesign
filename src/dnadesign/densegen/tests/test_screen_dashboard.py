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
from types import SimpleNamespace

from rich.console import Console

from dnadesign.densegen.src.core.pipeline import orchestrator
from dnadesign.densegen.src.core.pipeline.progress import _build_screen_dashboard, _ScreenDashboard
from dnadesign.densegen.src.utils import logging_utils


def test_screen_dashboard_static_output_for_non_terminal() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    dashboard = _ScreenDashboard(console=console, refresh_seconds=1.0, append=True)
    panel = _build_screen_dashboard(
        source_label="demo",
        plan_name="plan",
        bar="[----]",
        generated=1,
        quota=2,
        pct=50.0,
        global_bar=None,
        global_generated=None,
        global_quota=None,
        global_pct=None,
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
        legend=None,
        show_tfbs=False,
        show_solutions=False,
        sequence_preview=None,
    )

    dashboard.update(panel)
    dashboard.update(panel)
    output = buffer.getvalue()
    assert output.count("DenseGen progress") == 2


def test_build_shared_dashboard_uses_logging_console(monkeypatch) -> None:
    console = Console()
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", console, raising=False)
    cfg = SimpleNamespace(progress_style="screen", progress_refresh_seconds=1.0)
    dashboard = orchestrator._build_shared_dashboard(cfg)
    assert dashboard is not None
    assert dashboard._console is console
