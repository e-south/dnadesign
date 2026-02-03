"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_mining_progress.py

Stage-A mining progress rendering rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from rich.console import Console

from dnadesign.densegen.src.adapters.sources.stage_a import stage_a_progress
from dnadesign.densegen.src.utils import logging_utils


class _BufferStream:
    def __init__(self, *, isatty: bool) -> None:
        self._isatty = bool(isatty)
        self.data: List[str] = []

    def isatty(self) -> bool:
        return self._isatty

    def write(self, text: str) -> None:
        self.data.append(text)

    def flush(self) -> None:
        return None


def test_stage_a_screen_progress_stream_for_non_tty() -> None:
    prev_enabled = logging_utils.is_progress_enabled()
    prev_style = logging_utils.get_progress_style()
    logging_utils.set_progress_enabled(True)
    logging_utils.set_progress_style("screen")
    stream = _BufferStream(isatty=False)
    progress = stage_a_progress._PwmSamplingProgress(
        motif_id="demo_motif",
        backend="fimo",
        target=10,
        accepted_target=5,
        stream=stream,
        target_fraction=0.01,
    )
    try:
        progress.update(generated=10, accepted=5)
        progress.finish()
        output = "".join(stream.data)
        assert "PWM demo_motif" in output
        assert "\r" not in output
    finally:
        logging_utils.set_progress_enabled(prev_enabled)
        logging_utils.set_progress_style(prev_style)


def test_stage_a_screen_progress_stream_for_pixi_tty(monkeypatch) -> None:
    prev_enabled = logging_utils.is_progress_enabled()
    prev_style = logging_utils.get_progress_style()
    monkeypatch.setenv("PIXI_IN_SHELL", "1")
    logging_utils.set_progress_enabled(True)
    logging_utils.set_progress_style("screen")
    stream = _BufferStream(isatty=True)
    progress = stage_a_progress._PwmSamplingProgress(
        motif_id="demo_motif",
        backend="fimo",
        target=10,
        accepted_target=5,
        stream=stream,
        target_fraction=0.01,
    )
    try:
        assert progress._use_live is True
        assert progress._use_table is False
        progress.update(generated=10, accepted=5)
        progress.finish()
        assert progress._manager is not None
        assert progress._manager._live is None
    finally:
        logging_utils.set_progress_enabled(prev_enabled)
        logging_utils.set_progress_style(prev_style)


def test_stage_a_progress_target_updates_in_render() -> None:
    state = {
        "demo": stage_a_progress.StageAProgressState(
            motif_id="demo",
            backend="fimo",
            phase=None,
            generated=0,
            target=5,
            accepted=None,
            accepted_target=None,
            target_fraction=None,
            tier_fractions=[0.001, 0.01, 0.09],
            elapsed=0.0,
            batch_index=None,
            batch_total=None,
        )
    }
    state["demo"].generated = 10
    state["demo"].accepted = 2
    state["demo"].elapsed = 1.0
    state["demo"].target = 20
    renderable = stage_a_progress._stage_a_live_render(state)
    console = Console(width=120, record=True)
    console.print(renderable)
    text = console.export_text()
    assert "10/20" in text


def test_stage_a_progress_renders_accept_rate_and_rejects() -> None:
    state = {
        "bg": stage_a_progress.StageAProgressState(
            motif_id="neutral_bg",
            backend="background",
            phase="background",
            generated=10,
            target=100,
            accepted=5,
            accepted_target=20,
            target_fraction=None,
            tier_fractions=None,
            elapsed=1.0,
            batch_index=1,
            batch_total=10,
            show_tier_yield=False,
            show_accept_rate=True,
            show_rejects=True,
            reject_fimo=1,
            reject_kmer=2,
            reject_gc=3,
            reject_dup=4,
        )
    }
    renderable = stage_a_progress._stage_a_live_render(state)
    console = Console(width=140, record=True)
    console.print(renderable)
    text = console.export_text()
    assert "accept %" in text
    assert "rejects" in text
    assert "50%" in text
    assert "1/2/3/4" in text


def test_stage_a_milestone_message_format() -> None:
    message = stage_a_progress._format_stage_a_milestone(
        motif_id="lexA",
        phase="postprocess",
        detail="eligible_unique=10",
        elapsed=1.23,
    )
    assert "Stage-A postprocess" in message
    assert "lexA" in message
    assert "eligible_unique=10" in message
    assert "1.2s" in message


def test_stage_a_tier_yield_hidden_when_disabled() -> None:
    label = stage_a_progress._tier_yield_label(
        accepted=10,
        tier_fractions=[0.001, 0.01, 0.09],
        show_tier_yield=False,
    )
    assert label == "-"
    label = stage_a_progress._tier_yield_label(
        accepted=10,
        tier_fractions=[0.001, 0.01, 0.09],
        show_tier_yield=True,
    )
    assert label != "-"


def test_background_sampling_progress_uses_live_table(monkeypatch) -> None:
    prev_enabled = logging_utils.is_progress_enabled()
    prev_style = logging_utils.get_progress_style()
    monkeypatch.setenv("PIXI_IN_SHELL", "1")
    logging_utils.set_progress_enabled(True)
    logging_utils.set_progress_style("screen")
    stream = _BufferStream(isatty=True)
    progress = stage_a_progress._BackgroundSamplingProgress(
        input_name="neutral_bg",
        target=100,
        accepted_target=10,
        stream=stream,
    )
    try:
        assert progress._use_live is True
        assert progress._manager is not None
        state = list(progress._manager._state.values())
        assert state
        assert state[0].show_tier_yield is False
        assert state[0].show_accept_rate is True
        assert state[0].show_rejects is True
        progress.update(generated=10, accepted=2, batch_index=1, batch_total=10)
        progress.finish()
        assert progress._manager._live is None
    finally:
        logging_utils.set_progress_enabled(prev_enabled)
        logging_utils.set_progress_style(prev_style)
