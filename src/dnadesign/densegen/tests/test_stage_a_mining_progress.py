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

from dnadesign.densegen.src.adapters.sources import pwm_sampling
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
    progress = pwm_sampling._PwmSamplingProgress(
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
