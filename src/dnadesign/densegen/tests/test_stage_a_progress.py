"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_progress.py

Stage-A progress rendering behavior for interactive streams.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import io

from dnadesign.densegen.src.adapters.sources.stage_a_progress import _PwmSamplingProgress, _stage_a_live_start
from dnadesign.densegen.src.utils import logging_utils


class _DummyStream:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def write(self, _text: str) -> int:
        return 0

    def flush(self) -> None:
        return None


class _CaptureStream(io.StringIO):
    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_progress_allows_carriage_for_stream_tty() -> None:
    prev_style = logging_utils.get_progress_style()
    prev_enabled = logging_utils.is_progress_enabled()
    try:
        logging_utils.set_progress_style("stream")
        logging_utils.set_progress_enabled(True)
        progress = _PwmSamplingProgress(
            motif_id="demo",
            backend="fimo",
            target=10,
            accepted_target=5,
            stream=_DummyStream(tty=True),
            target_fraction=0.001,
        )
        try:
            assert progress._allow_carriage is True
            assert progress._use_live is False
        finally:
            progress.finish()
    finally:
        logging_utils.set_progress_style(prev_style)
        logging_utils.set_progress_enabled(prev_enabled)


def test_stream_phase_update_emits_line() -> None:
    prev_style = logging_utils.get_progress_style()
    prev_enabled = logging_utils.is_progress_enabled()
    try:
        logging_utils.set_progress_style("stream")
        logging_utils.set_progress_enabled(True)
        stream = _CaptureStream(tty=True)
        progress = _PwmSamplingProgress(
            motif_id="demo",
            backend="fimo",
            target=10,
            accepted_target=5,
            stream=stream,
            target_fraction=0.001,
        )
        try:
            progress.update(generated=2, accepted=1)
            progress.set_phase("postprocess")
        finally:
            progress.finish()
        output = stream.getvalue()
        assert "phase postprocess" in output
    finally:
        logging_utils.set_progress_style(prev_style)
        logging_utils.set_progress_enabled(prev_enabled)


def test_live_start_requires_tty() -> None:
    live, console = _stage_a_live_start(_DummyStream(tty=False))
    assert live is None
    assert console is None
