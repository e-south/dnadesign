"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_sampling_diagnostics.py

Runtime logging behavior tests for Stage-B sampling diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.core.pipeline.sampling_diagnostics import SamplingDiagnostics


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args) -> None:
        if args:
            self.messages.append(message % args)
        else:
            self.messages.append(message)


def test_sampling_diagnostics_summary_mode_is_quiet() -> None:
    logger = _Recorder()
    diagnostics = SamplingDiagnostics(
        usage_counts={("lexA", "AAA"): 2},
        tf_usage_counts={"lexA": 2},
        failure_counts={},
        source_label="demo_input",
        plan_name="demo_plan",
        display_tf_label=lambda label: label,
        progress_style="summary",
        show_tfbs=False,
        track_failures=True,
        logger=logger,
        library_tfs=["lexA"],
        library_tfbs=["AAA"],
        library_site_ids=["s1"],
    )

    diagnostics.log_snapshot()

    assert logger.messages == []


def test_sampling_diagnostics_stream_mode_emits_leaderboard_lines() -> None:
    logger = _Recorder()
    diagnostics = SamplingDiagnostics(
        usage_counts={("lexA", "AAA"): 2},
        tf_usage_counts={"lexA": 2},
        failure_counts={},
        source_label="demo_input",
        plan_name="demo_plan",
        display_tf_label=lambda label: label,
        progress_style="stream",
        show_tfbs=False,
        track_failures=True,
        logger=logger,
        library_tfs=["lexA"],
        library_tfbs=["AAA"],
        library_site_ids=["s1"],
    )

    diagnostics.log_snapshot()

    assert any("Leaderboard (TF)" in message for message in logger.messages)
    assert any("TFBS usage" in message for message in logger.messages)
