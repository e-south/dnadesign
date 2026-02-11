"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_screen_progress.py

Tests for screen-mode progress reporting behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from rich.console import Console

from dnadesign.densegen.src.core.pipeline import orchestrator
from dnadesign.densegen.src.core.pipeline.progress import PlanProgressReporter
from dnadesign.densegen.src.utils import logging_utils


class _DummyDashboard:
    def __init__(self) -> None:
        self.update_count = 0
        self.last_renderable = None

    def update(self, _renderable) -> None:
        self.update_count += 1
        self.last_renderable = _renderable

    def terminal_height(self) -> int | None:
        return None


def _make_log_cfg(**overrides):
    defaults = {
        "progress_style": "screen",
        "progress_every": 1,
        "progress_refresh_seconds": 1.0,
        "print_visual": True,
        "show_tfbs": False,
        "show_solutions": False,
        "visuals": SimpleNamespace(tf_colors={"TF1": "#A6CEE3"}),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_screen_progress_uses_live_dashboard_when_tty(monkeypatch) -> None:
    monkeypatch.setenv("PIXI_IN_SHELL", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", None, raising=False)
    settings = orchestrator._init_progress_settings(
        log_cfg=_make_log_cfg(),
        source_label="demo",
        plan_name="demo",
        quota=1,
        max_per_subsample=1,
        show_tfbs=False,
        show_solutions=False,
        extra_library_label=None,
    )
    assert settings.progress_style == "screen"
    assert settings.dashboard is not None
    assert settings.dashboard._live is not None


def test_screen_progress_rejects_dumb_term_for_live_dashboard(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", None, raising=False)
    with pytest.raises(RuntimeError, match="TERM"):
        orchestrator._init_progress_settings(
            log_cfg=_make_log_cfg(),
            source_label="demo",
            plan_name="demo",
            quota=1,
            max_per_subsample=1,
            show_tfbs=False,
            show_solutions=False,
            extra_library_label=None,
        )


def test_screen_progress_requires_tty(monkeypatch) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", None, raising=False)
    with pytest.raises(RuntimeError, match="interactive terminal"):
        orchestrator._init_progress_settings(
            log_cfg=_make_log_cfg(),
            source_label="demo",
            plan_name="demo",
            quota=1,
            max_per_subsample=1,
            show_tfbs=False,
            show_solutions=False,
            extra_library_label=None,
        )


def test_auto_progress_uses_live_dashboard_when_interactive(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", None, raising=False)
    settings = orchestrator._init_progress_settings(
        log_cfg=_make_log_cfg(progress_style="auto"),
        source_label="demo",
        plan_name="demo",
        quota=1,
        max_per_subsample=1,
        show_tfbs=False,
        show_solutions=False,
        extra_library_label=None,
    )
    assert settings.progress_style == "screen"
    assert settings.dashboard is not None
    assert settings.dashboard._live is not None


def test_auto_progress_downgrades_to_summary_when_non_interactive(monkeypatch) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", None, raising=False)
    settings = orchestrator._init_progress_settings(
        log_cfg=_make_log_cfg(progress_style="auto", print_visual=False, visuals=None),
        source_label="demo",
        plan_name="demo",
        quota=1,
        max_per_subsample=1,
        show_tfbs=False,
        show_solutions=False,
        extra_library_label=None,
    )
    assert settings.progress_style == "summary"
    assert settings.dashboard is None


def test_auto_progress_downgrades_to_stream_for_dumb_term(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(logging_utils, "_LOGGING_CONSOLE", None, raising=False)
    settings = orchestrator._init_progress_settings(
        log_cfg=_make_log_cfg(progress_style="auto", print_visual=False, visuals=None),
        source_label="demo",
        plan_name="demo",
        quota=1,
        max_per_subsample=1,
        show_tfbs=False,
        show_solutions=False,
        extra_library_label=None,
    )
    assert settings.progress_style == "stream"
    assert settings.dashboard is None


def test_screen_progress_updates_per_solution_when_visual(monkeypatch) -> None:
    dashboard = _DummyDashboard()
    reporter = PlanProgressReporter(
        source_label="demo",
        plan_name="plan",
        quota=2,
        max_per_subsample=1,
        progress_style="screen",
        progress_every=1,
        progress_refresh_seconds=999.0,
        show_tfbs=False,
        show_solutions=False,
        print_visual=True,
        tf_colors={"TF1": "#A6CEE3"},
        dashboard=dashboard,
    )
    monkeypatch.setattr("dnadesign.densegen.src.core.pipeline.progress.time.monotonic", lambda: 0.0)

    class _VisualSol:
        compression_ratio = 1.0
        library = ["AAA"]
        offsets_fwd = [0]
        offsets_rev = [None]
        sequence_length = 3
        sequence = "AAA"

        def __str__(self) -> str:
            return "--> AAA\n--> AAA\n<-- TTT"

    sol = _VisualSol()
    counters = SimpleNamespace(total_resamples=0)
    for idx in range(2):
        reporter.record_solution(
            global_generated=idx + 1,
            local_generated=1,
            library_index=1,
            sol=sol,
            library_tfs=["TF1"],
            library_tfbs=["AAAA"],
            used_tfbs_detail=[],
            used_tf_list=["TF1"],
            final_seq="AAAA",
            counters=counters,
            duplicate_records=0,
            duplicate_solutions=0,
            failed_solutions=0,
            stall_events=0,
            usage_counts={},
            tf_usage_counts={},
            tf_usage_display={},
            tfbs_usage_display={},
        )
    assert dashboard.update_count == 2


def test_screen_progress_reuses_shared_dashboard(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    shared = orchestrator._ScreenDashboard(console=Console(), refresh_seconds=1.0)
    settings_a = orchestrator._init_progress_settings(
        log_cfg=_make_log_cfg(print_visual=False, visuals=None),
        source_label="demo",
        plan_name="plan-a",
        quota=1,
        max_per_subsample=1,
        show_tfbs=False,
        show_solutions=False,
        extra_library_label=None,
        shared_dashboard=shared,
    )
    settings_b = orchestrator._init_progress_settings(
        log_cfg=_make_log_cfg(print_visual=False, visuals=None),
        source_label="demo",
        plan_name="plan-b",
        quota=1,
        max_per_subsample=1,
        show_tfbs=False,
        show_solutions=False,
        extra_library_label=None,
        shared_dashboard=shared,
    )
    assert settings_a.dashboard is settings_b.dashboard


def test_screen_progress_shows_dense_array_visual(monkeypatch) -> None:
    dashboard = _DummyDashboard()
    reporter = PlanProgressReporter(
        source_label="demo",
        plan_name="plan",
        quota=1,
        max_per_subsample=1,
        progress_style="screen",
        progress_every=1,
        progress_refresh_seconds=999.0,
        show_tfbs=False,
        show_solutions=False,
        print_visual=True,
        tf_colors={"TF1": "#A6CEE3"},
        dashboard=dashboard,
    )
    monkeypatch.setattr("dnadesign.densegen.src.core.pipeline.progress.time.monotonic", lambda: 0.0)

    class _VisualSol:
        compression_ratio = 1.0
        library = ["AAA"]
        offsets_fwd = [0]
        offsets_rev = [None]
        sequence_length = 3
        sequence = "AAA"

        def __str__(self) -> str:
            return "--> AAA\n--> AAA\n<-- TTT"

    sol = _VisualSol()
    counters = SimpleNamespace(total_resamples=0)
    reporter.record_solution(
        global_generated=1,
        local_generated=1,
        library_index=1,
        sol=sol,
        library_tfs=["TF1"],
        library_tfbs=["AAAA"],
        used_tfbs_detail=[],
        used_tf_list=["TF1"],
        final_seq="AAAA",
        counters=counters,
        duplicate_records=0,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        usage_counts={},
        tf_usage_counts={},
        tf_usage_display={},
        tfbs_usage_display={},
    )
    assert dashboard.last_renderable is not None
    console = Console(width=120, record=True)
    console.print(dashboard.last_renderable)
    text = console.export_text(styles=False)
    assert "AAA" in text


def test_screen_progress_includes_visual_legend(monkeypatch) -> None:
    dashboard = _DummyDashboard()
    reporter = PlanProgressReporter(
        source_label="demo",
        plan_name="plan",
        quota=1,
        max_per_subsample=1,
        progress_style="screen",
        progress_every=1,
        progress_refresh_seconds=999.0,
        show_tfbs=False,
        show_solutions=False,
        print_visual=True,
        tf_colors={"TF1": "#A6CEE3"},
        dashboard=dashboard,
    )
    monkeypatch.setattr("dnadesign.densegen.src.core.pipeline.progress.time.monotonic", lambda: 0.0)

    class _VisualSol:
        compression_ratio = 1.0
        library = ["AAA"]
        offsets_fwd = [0]
        offsets_rev = [None]
        sequence_length = 3
        sequence = "AAA"

        def __str__(self) -> str:
            return "--> AAA\n--> AAA\n<-- TTT"

    sol = _VisualSol()
    counters = SimpleNamespace(total_resamples=0)
    reporter.record_solution(
        global_generated=1,
        local_generated=1,
        library_index=1,
        sol=sol,
        library_tfs=["TF1"],
        library_tfbs=["AAAA"],
        used_tfbs_detail=[],
        used_tf_list=["TF1"],
        final_seq="AAAA",
        counters=counters,
        duplicate_records=0,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        usage_counts={},
        tf_usage_counts={},
        tf_usage_display={},
        tfbs_usage_display={},
    )
    console = Console(width=120, record=True)
    console.print(dashboard.last_renderable)
    text = console.export_text(styles=False)
    assert "legend:" in text
    assert "TF1" in text
    assert "counts (resample/dup/fail/stall)" in text
    assert "TFBS usage (unique tf/tfbs used)" in text
    assert "diversity (tf_coverage/tfbs_coverage/tfbs_entropy)" in text


def test_screen_progress_includes_global_progress(monkeypatch) -> None:
    dashboard = _DummyDashboard()
    reporter = PlanProgressReporter(
        source_label="demo",
        plan_name="plan",
        quota=2,
        max_per_subsample=1,
        progress_style="screen",
        progress_every=1,
        progress_refresh_seconds=999.0,
        show_tfbs=False,
        show_solutions=True,
        print_visual=False,
        tf_colors=None,
        dashboard=dashboard,
    )
    monkeypatch.setattr("dnadesign.densegen.src.core.pipeline.progress.time.monotonic", lambda: 0.0)

    class _Sol:
        compression_ratio = 1.0
        library = ["AAA"]
        offsets_fwd = [0]
        offsets_rev = [None]
        sequence_length = 3
        sequence = "AAA"

        def __str__(self) -> str:
            return "--> AAA\n--> AAA\n<-- TTT"

    sol = _Sol()
    counters = SimpleNamespace(total_resamples=0)
    reporter.record_solution(
        global_generated=1,
        local_generated=1,
        library_index=1,
        sol=sol,
        library_tfs=["TF1"],
        library_tfbs=["AAAA"],
        used_tfbs_detail=[],
        used_tf_list=["TF1"],
        final_seq="AAAA",
        counters=counters,
        duplicate_records=0,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        usage_counts={},
        tf_usage_counts={},
        tf_usage_display={},
        tfbs_usage_display={},
        global_total_generated=2,
        global_total_quota=4,
    )
    console = Console(width=120, record=True)
    console.print(dashboard.last_renderable)
    text = console.export_text(styles=False)
    assert "global progress" in text


def test_dense_array_visual_coloring_requires_tf_colors() -> None:
    from dnadesign.densegen.src.core.pipeline import progress as progress_module

    class _Sol:
        library = ["AAA"]
        offsets_fwd = [0]
        offsets_rev = [None]
        sequence_length = 3
        sequence = "AAA"

        def __str__(self) -> str:
            return "--> AAA\n--> AAA\n<-- TTT"

    with pytest.raises(ValueError, match="logging.visuals.tf_colors"):
        progress_module._render_dense_array_visual(_Sol(), library_tfs=["TF1"], tf_colors={})


def test_dense_array_visual_colors_label_lines() -> None:
    from rich.console import Group
    from rich.text import Text

    from dnadesign.densegen.src.core.pipeline import progress as progress_module

    class _Sol:
        library = ["AAA"]
        offsets_fwd = [0]
        offsets_rev = [None]
        sequence_length = 3
        sequence = "AAA"

        def __str__(self) -> str:
            return "--> AAA\n--> AAA\n<-- TTT"

    renderable = progress_module._render_dense_array_visual(
        _Sol(),
        library_tfs=["TF1"],
        tf_colors={"TF1": "#A6CEE3"},
    )
    assert isinstance(renderable, Group)
    label_line = renderable.renderables[0]
    assert isinstance(label_line, Text)
    assert "AAA" in label_line.plain
    assert label_line.spans


def test_dense_array_visual_handles_extra_library_entries() -> None:
    from dnadesign.densegen.src.core.pipeline import progress as progress_module

    class _Sol:
        library = ["AAA", "TT"]
        offsets_fwd = [0, 3]
        offsets_rev = [None, None]
        sequence_length = 5
        sequence = "AAATT"

    renderable = progress_module._render_dense_array_visual(
        _Sol(),
        library_tfs=["TF1"],
        tf_colors={
            "TF1": "#A6CEE3",
            progress_module._EXTRA_LIBRARY_LABEL: "#8DD3C7",
        },
    )
    assert renderable is not None


def test_dense_array_visual_uses_custom_extra_label() -> None:
    from dnadesign.densegen.src.core.pipeline import progress as progress_module

    class _Sol:
        library = ["AAA", "TT"]
        offsets_fwd = [0, 3]
        offsets_rev = [None, None]
        sequence_length = 5
        sequence = "AAATT"

    renderable = progress_module._render_dense_array_visual(
        _Sol(),
        library_tfs=["TF1"],
        tf_colors={
            "TF1": "#A6CEE3",
            "sigma70_consensus": "#8DD3C7",
        },
        extra_label="sigma70_consensus",
    )
    assert renderable is not None


def test_dense_array_visual_requires_extra_label_color() -> None:
    from dnadesign.densegen.src.core.pipeline import progress as progress_module

    class _Sol:
        library = ["AAA", "TT"]
        offsets_fwd = [0, 3]
        offsets_rev = [None, None]
        sequence_length = 5
        sequence = "AAATT"

    with pytest.raises(ValueError, match=progress_module._EXTRA_LIBRARY_LABEL):
        progress_module._render_dense_array_visual(
            _Sol(),
            library_tfs=["TF1"],
            tf_colors={"TF1": "#A6CEE3"},
        )


def test_screen_progress_slides_visual_window_when_height_constrained(monkeypatch) -> None:
    class _ConstrainedDashboard(_DummyDashboard):
        def terminal_height(self) -> int | None:
            return 16

    dashboard = _ConstrainedDashboard()
    reporter = PlanProgressReporter(
        source_label="demo",
        plan_name="plan",
        quota=2,
        max_per_subsample=1,
        progress_style="screen",
        progress_every=1,
        progress_refresh_seconds=999.0,
        show_tfbs=False,
        show_solutions=False,
        print_visual=True,
        tf_colors={"TF1": "#A6CEE3"},
        dashboard=dashboard,
    )
    monkeypatch.setattr("dnadesign.densegen.src.core.pipeline.progress.time.monotonic", lambda: 0.0)

    class _VisualSol:
        compression_ratio = 1.0
        library = ["AAA", "CCC", "GGG", "TTT", "ATA", "CGC"]
        offsets_fwd = [0, 0, 0, 0, 0, 0]
        offsets_rev = [0, 0, 0, 0, 0, 0]
        sequence_length = 6
        sequence = "ACGTAC"

        def __str__(self) -> str:
            return "visual"

    sol = _VisualSol()
    counters = SimpleNamespace(total_resamples=0)

    reporter.record_solution(
        global_generated=1,
        local_generated=1,
        library_index=1,
        sol=sol,
        library_tfs=["TF1"] * len(sol.library),
        library_tfbs=["AAAA"] * len(sol.library),
        used_tfbs_detail=[],
        used_tf_list=["TF1"],
        final_seq="ACGTAC",
        counters=counters,
        duplicate_records=0,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        usage_counts={},
        tf_usage_counts={},
        tf_usage_display={},
        tfbs_usage_display={},
    )
    first = Console(width=120, record=True)
    first.print(dashboard.last_renderable)
    first_text = first.export_text(styles=False)
    assert "window" in first_text

    reporter.record_solution(
        global_generated=2,
        local_generated=1,
        library_index=1,
        sol=sol,
        library_tfs=["TF1"] * len(sol.library),
        library_tfbs=["AAAA"] * len(sol.library),
        used_tfbs_detail=[],
        used_tf_list=["TF1"],
        final_seq="ACGTAC",
        counters=counters,
        duplicate_records=0,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        usage_counts={},
        tf_usage_counts={},
        tf_usage_display={},
        tfbs_usage_display={},
    )
    second = Console(width=120, record=True)
    second.print(dashboard.last_renderable)
    second_text = second.export_text(styles=False)
    assert first_text != second_text


def test_screen_progress_inlines_solver_settings(monkeypatch) -> None:
    dashboard = _DummyDashboard()
    reporter = PlanProgressReporter(
        source_label="demo",
        plan_name="plan",
        quota=1,
        max_per_subsample=1,
        progress_style="screen",
        progress_every=1,
        progress_refresh_seconds=999.0,
        show_tfbs=False,
        show_solutions=True,
        print_visual=False,
        tf_colors=None,
        dashboard=dashboard,
    )
    reporter.solver_settings = "backend=CBC strategy=iterate strands=double"
    monkeypatch.setattr("dnadesign.densegen.src.core.pipeline.progress.time.monotonic", lambda: 0.0)

    class _Sol:
        compression_ratio = 1.0
        library = ["AAA"]
        offsets_fwd = [0]
        offsets_rev = [None]
        sequence_length = 3
        sequence = "AAA"

    counters = SimpleNamespace(total_resamples=0)
    reporter.record_solution(
        global_generated=1,
        local_generated=1,
        library_index=1,
        sol=_Sol(),
        library_tfs=["TF1"],
        library_tfbs=["AAAA"],
        used_tfbs_detail=[],
        used_tf_list=["TF1"],
        final_seq="AAAA",
        counters=counters,
        duplicate_records=0,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        usage_counts={},
        tf_usage_counts={},
        tf_usage_display={},
        tfbs_usage_display={},
        global_total_generated=1,
        global_total_quota=1,
    )
    console = Console(width=120, record=True)
    console.print(dashboard.last_renderable)
    text = console.export_text(styles=False)
    assert "solver" in text.lower()
    assert "cbc" in text.lower()


def test_screen_progress_respects_rendered_line_budget(monkeypatch) -> None:
    dashboard = _DummyDashboard()
    reporter = PlanProgressReporter(
        source_label="demo",
        plan_name="plan",
        quota=1,
        max_per_subsample=1,
        progress_style="screen",
        progress_every=1,
        progress_refresh_seconds=999.0,
        show_tfbs=False,
        show_solutions=False,
        print_visual=True,
        tf_colors={"TF1": "#A6CEE3"},
        dashboard=dashboard,
    )
    reporter.solver_settings = (
        "backend=CBC strategy=iterate strands=double time_limit=5.0s threads=1 seq_len=60 long_field=abcdef"
    )
    monkeypatch.setattr("dnadesign.densegen.src.core.pipeline.progress.time.monotonic", lambda: 0.0)

    class _VisualSol:
        compression_ratio = 1.0
        library = ["AAA", "CCC", "GGG", "TTT", "ATA", "CGC", "TGA", "ACT"]
        offsets_fwd = [0] * 8
        offsets_rev = [0] * 8
        sequence_length = 8
        sequence = "ACGTACGT"

    class _BudgetDashboard(_DummyDashboard):
        def terminal_height(self) -> int | None:
            return 14

        def terminal_width(self) -> int | None:
            return 90

    budget_dashboard = _BudgetDashboard()
    reporter.dashboard = budget_dashboard
    counters = SimpleNamespace(total_resamples=0)
    reporter.record_solution(
        global_generated=1,
        local_generated=1,
        library_index=1,
        sol=_VisualSol(),
        library_tfs=["TF1"] * 8,
        library_tfbs=["AAAA"] * 8,
        used_tfbs_detail=[],
        used_tf_list=["TF1"],
        final_seq="ACGTACGT",
        counters=counters,
        duplicate_records=0,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        usage_counts={},
        tf_usage_counts={},
        tf_usage_display={},
        tfbs_usage_display={},
        global_total_generated=1,
        global_total_quota=1,
    )
    console = Console(width=90, record=True)
    console.print(budget_dashboard.last_renderable)
    text = console.export_text(styles=False).rstrip("\n")
    assert len(text.splitlines()) <= 14
