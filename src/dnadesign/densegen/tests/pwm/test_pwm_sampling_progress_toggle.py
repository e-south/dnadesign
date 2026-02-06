"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/pwm/test_pwm_sampling_progress_toggle.py

Unit tests for Stage-A PWM sampling progress output toggles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_progress import _PwmSamplingProgress
from dnadesign.densegen.src.utils import logging_utils


class _DummyStream:
    def __init__(self) -> None:
        self.contents = ""

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.contents += text
        return len(text)

    def flush(self) -> None:
        return None


def test_pwm_progress_respects_global_toggle() -> None:
    stream = _DummyStream()
    logging_utils.set_progress_enabled(False)
    progress = _PwmSamplingProgress(
        motif_id="motif",
        backend="fimo",
        target=10,
        accepted_target=None,
        stream=stream,
    )
    progress.update(generated=1, accepted=None, force=True)
    progress.finish()
    logging_utils.set_progress_enabled(True)
    assert stream.contents == ""
