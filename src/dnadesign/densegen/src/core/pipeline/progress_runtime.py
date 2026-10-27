"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/progress_runtime.py

Progress-style resolution and screen dashboard bootstrap for Stage-B runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import sys

from rich.console import Console

from ...utils import logging_utils
from .progress import PlanProgressReporter, _ScreenDashboard
from .stage_b_runtime_types import ProgressSettings

log = logging.getLogger(__name__)


def _progress_terminal_probe(
    *,
    shared_dashboard: _ScreenDashboard | None = None,
) -> tuple[Console | None, bool | None, str | None]:
    console = getattr(shared_dashboard, "_console", None) if shared_dashboard is not None else None
    if console is None:
        console = logging_utils.get_logging_console()
    if console is None:
        return None, None, os.environ.get("TERM")
    is_tty = bool(getattr(console, "is_terminal", False))
    console_environ = getattr(console, "_environ", None)
    console_term = None
    if isinstance(console_environ, dict):
        console_term = console_environ.get("TERM")
    term_value = (
        "dumb" if bool(getattr(console, "is_dumb_terminal", False)) else (console_term or os.environ.get("TERM"))
    )
    return console, is_tty, term_value


def _dashboard_append_mode(console: Console) -> bool:
    if not bool(getattr(console, "is_terminal", False)):
        raise RuntimeError(
            "logging.progress_style=screen requires an interactive terminal. "
            "Use logging.progress_style=stream for non-interactive output."
        )
    if bool(getattr(console, "is_dumb_terminal", False)):
        raise RuntimeError(
            "logging.progress_style=screen requires TERM with cursor controls (TERM must not be 'dumb'). "
            "Set TERM=xterm-256color or switch logging.progress_style to 'stream'."
        )
    return False


def _build_shared_dashboard(log_cfg) -> _ScreenDashboard | None:
    requested_progress_style = str(getattr(log_cfg, "progress_style", "stream"))
    probed_console, probed_tty, probed_term = _progress_terminal_probe()
    progress_style, _progress_reason = logging_utils.resolve_progress_style(
        requested_progress_style,
        stdout=sys.stdout,
        term=probed_term,
        is_tty=probed_tty,
    )
    progress_refresh_seconds = float(getattr(log_cfg, "progress_refresh_seconds", 1.0))
    if progress_style != "screen":
        return None
    console = probed_console
    if console is None:
        tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if tty:
            console = Console()
        else:
            raise RuntimeError(
                "logging.progress_style=screen requires an interactive terminal. "
                "Use logging.progress_style=stream for non-interactive output."
            )
    return _ScreenDashboard(
        console=console,
        refresh_seconds=progress_refresh_seconds,
        append=_dashboard_append_mode(console),
    )


def _init_progress_settings(
    *,
    log_cfg,
    source_label: str,
    plan_name: str,
    quota: int,
    max_per_subsample: int,
    show_tfbs: bool,
    show_solutions: bool,
    extra_library_label: str | None,
    shared_dashboard: _ScreenDashboard | None = None,
) -> ProgressSettings:
    requested_progress_style = str(getattr(log_cfg, "progress_style", "stream"))
    probed_console, probed_tty, probed_term = _progress_terminal_probe(shared_dashboard=shared_dashboard)
    progress_style, _progress_reason = logging_utils.resolve_progress_style(
        requested_progress_style,
        stdout=sys.stdout,
        term=probed_term,
        is_tty=probed_tty,
    )
    progress_every = int(getattr(log_cfg, "progress_every", 1))
    progress_refresh_seconds = float(getattr(log_cfg, "progress_refresh_seconds", 1.0))
    logging_utils.set_progress_style(progress_style)
    logging_utils.set_progress_enabled(progress_style in {"stream", "screen"})
    screen_console = None
    if progress_style == "screen":
        if shared_dashboard is None:
            screen_console = probed_console
            if screen_console is None:
                tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
                if tty:
                    screen_console = Console()
                else:
                    raise RuntimeError(
                        "logging.progress_style=screen requires an interactive terminal. "
                        "Use logging.progress_style=stream for non-interactive output."
                    )
    show_tfbs = bool(show_tfbs or getattr(log_cfg, "show_tfbs", False))
    show_solutions = bool(show_solutions or getattr(log_cfg, "show_solutions", False))
    tf_colors = None
    if bool(getattr(log_cfg, "print_visual", False)):
        visuals = getattr(log_cfg, "visuals", None)
        tf_colors = getattr(visuals, "tf_colors", None) if visuals is not None else None
        if not tf_colors:
            raise ValueError("logging.visuals.tf_colors must be set when logging.print_visual is true")
    if progress_style == "screen":
        if shared_dashboard is not None:
            dashboard = shared_dashboard
        else:
            dashboard = (
                _ScreenDashboard(
                    console=screen_console,
                    refresh_seconds=progress_refresh_seconds,
                    append=_dashboard_append_mode(screen_console),
                )
                if screen_console is not None
                else None
            )
    else:
        dashboard = None
    progress_reporter = PlanProgressReporter(
        source_label=source_label,
        plan_name=plan_name,
        quota=int(quota),
        max_per_subsample=int(max_per_subsample),
        progress_style=progress_style,
        progress_every=progress_every,
        progress_refresh_seconds=progress_refresh_seconds,
        show_tfbs=show_tfbs,
        show_solutions=show_solutions,
        print_visual=bool(getattr(log_cfg, "print_visual", False)),
        tf_colors=tf_colors,
        extra_library_label=extra_library_label,
        dashboard=dashboard,
        logger=log,
    )
    return ProgressSettings(
        progress_style=progress_style,
        progress_every=progress_every,
        progress_refresh_seconds=progress_refresh_seconds,
        print_visual=bool(getattr(log_cfg, "print_visual", False)),
        tf_colors=tf_colors,
        show_tfbs=show_tfbs,
        show_solutions=show_solutions,
        dashboard=dashboard,
        reporter=progress_reporter,
    )
