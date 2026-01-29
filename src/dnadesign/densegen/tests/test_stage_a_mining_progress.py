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

from dnadesign.densegen.src.adapters.sources import stage_a_progress
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


def test_stage_a_screen_progress_static_table_for_non_tty() -> None:
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
        assert "Stage-A mining" in output
        assert "\r" not in output
    finally:
        logging_utils.set_progress_enabled(prev_enabled)
        logging_utils.set_progress_style(prev_style)


def test_stage_a_screen_progress_static_table_for_pixi_tty(monkeypatch) -> None:
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
        assert progress._use_live is False
        assert progress._use_table is True
        progress.update(generated=10, accepted=5)
        progress.finish()
        output = "".join(stream.data)
        assert "Stage-A mining" in output
        assert "\r" not in output
    finally:
        logging_utils.set_progress_enabled(prev_enabled)
        logging_utils.set_progress_style(prev_style)


def test_stage_a_progress_target_updates_in_render() -> None:
    prev_state = dict(stage_a_progress._STAGE_A_LIVE_STATE)
    stage_a_progress._STAGE_A_LIVE_STATE.clear()
    stage_a_progress._STAGE_A_LIVE_STATE["demo"] = {
        "motif": "demo",
        "generated": 0,
        "target": 5,
        "accepted": None,
        "accepted_target": None,
        "target_fraction": None,
        "elapsed": 0.0,
    }
    try:
        stage_a_progress._stage_a_live_update(
            key="demo",
            generated=10,
            accepted=2,
            elapsed=1.0,
            target=20,
        )
        renderable = stage_a_progress._stage_a_live_render(stage_a_progress._STAGE_A_LIVE_STATE)
        console = Console(width=120, record=True)
        console.print(renderable)
        text = console.export_text()
        assert "10/20" in text
    finally:
        stage_a_progress._STAGE_A_LIVE_STATE.clear()
        stage_a_progress._STAGE_A_LIVE_STATE.update(prev_state)


def test_stage_a_milestone_message_format() -> None:
    message = stage_a_progress._format_stage_a_milestone(
        motif_id="lexA",
        phase="postprocess",
        detail="eligible=10",
        elapsed=1.23,
    )
    assert "Stage-A postprocess" in message
    assert "lexA" in message
    assert "eligible=10" in message
    assert "1.2s" in message
