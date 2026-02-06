"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_screen_dashboard.py

Screen dashboard behavior for non-terminal output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from dnadesign.densegen.src.core.pipeline import orchestrator
from dnadesign.densegen.src.core.pipeline.progress import _build_screen_dashboard, _ScreenDashboard
from dnadesign.densegen.src.utils import logging_utils


def _demo_panel():
    return _build_screen_dashboard(
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


def test_screen_dashboard_static_output_for_non_terminal() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    dashboard = _ScreenDashboard(console=console, refresh_seconds=1.0, append=True)
    panel = _demo_panel()

    dashboard.update(panel)
    dashboard.update(panel)
    output = buffer.getvalue()
    assert output.count("DenseGen progress") == 2


def test_screen_dashboard_ignores_closed_console_stream() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    dashboard = _ScreenDashboard(console=console, refresh_seconds=1.0, append=True)
    panel = _demo_panel()

    buffer.close()
    dashboard.update(panel)
    dashboard.close()


def test_build_shared_dashboard_uses_logging_console(monkeypatch) -> None:
    console = Console(force_terminal=True, _environ={"TERM": "xterm-256color"})
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", console, raising=False)
    cfg = SimpleNamespace(progress_style="screen", progress_refresh_seconds=1.0)
    dashboard = orchestrator._build_shared_dashboard(cfg)
    assert dashboard is not None
    assert dashboard._console is console


def test_build_shared_dashboard_uses_live_on_terminal_console(monkeypatch) -> None:
    console = Console(force_terminal=True, _environ={"TERM": "xterm-256color"})
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", console, raising=False)
    cfg = SimpleNamespace(progress_style="screen", progress_refresh_seconds=1.0)
    dashboard = orchestrator._build_shared_dashboard(cfg)
    assert dashboard is not None
    assert dashboard._append is False
    assert dashboard._live is not None


def test_build_shared_dashboard_rejects_dumb_terminal_console(monkeypatch) -> None:
    console = Console(force_terminal=True, _environ={"TERM": "dumb"})
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", console, raising=False)
    cfg = SimpleNamespace(progress_style="screen", progress_refresh_seconds=1.0)
    with pytest.raises(RuntimeError, match="TERM"):
        orchestrator._build_shared_dashboard(cfg)


def test_build_shared_dashboard_requires_terminal_console(monkeypatch) -> None:
    console = Console(file=StringIO(), force_terminal=False, width=120)
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", console, raising=False)
    cfg = SimpleNamespace(progress_style="screen", progress_refresh_seconds=1.0)
    with pytest.raises(RuntimeError, match="interactive terminal"):
        orchestrator._build_shared_dashboard(cfg)


class _CloseTracker:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_close_plan_dashboard_skips_shared_dashboard() -> None:
    shared = _CloseTracker()
    orchestrator._close_plan_dashboard(dashboard=shared, shared_dashboard=shared)
    assert shared.closed == 0


def test_close_plan_dashboard_closes_non_shared_dashboard() -> None:
    shared = _CloseTracker()
    local = _CloseTracker()
    orchestrator._close_plan_dashboard(dashboard=local, shared_dashboard=shared)
    assert local.closed == 1


def test_screen_dashboard_live_uses_alt_screen(monkeypatch) -> None:
    from dnadesign.densegen.src.core.pipeline import progress as progress_module

    calls: list[dict] = []

    class _FakeLive:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))

        def start(self) -> None:
            return

        def update(self, *_args, **_kwargs) -> None:
            return

        def stop(self) -> None:
            return

    monkeypatch.setattr(progress_module, "Live", _FakeLive)
    console = Console(force_terminal=True, _environ={"TERM": "xterm-256color"})
    _ScreenDashboard(console=console, refresh_seconds=1.0, append=False)
    assert calls
    assert calls[0].get("screen") is True


def test_screen_dashboard_mutes_console_handlers_while_live(monkeypatch) -> None:
    from dnadesign.densegen.src.core.pipeline import progress as progress_module

    class _FakeLive:
        def __init__(self, **_kwargs):
            return

        def start(self) -> None:
            return

        def update(self, *_args, **_kwargs) -> None:
            return

        def stop(self) -> None:
            return

    monkeypatch.setattr(progress_module, "Live", _FakeLive)
    console = Console(force_terminal=True, _environ={"TERM": "xterm-256color"})
    dashboard = _ScreenDashboard(console=console, refresh_seconds=1.0, append=False)
    root = logging.getLogger()

    class _ConsoleHandler(logging.Handler):
        def __init__(self, rich_console, level: int) -> None:
            super().__init__(level=level)
            self.console = rich_console

    handler = _ConsoleHandler(console, logging.INFO)
    root.addHandler(handler)
    try:
        dashboard.update(_demo_panel())
        assert handler.level > logging.CRITICAL
        dashboard.close()
        assert handler.level == logging.INFO
    finally:
        root.removeHandler(handler)
