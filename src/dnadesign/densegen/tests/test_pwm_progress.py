"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_progress.py

Stage-A PWM progress formatting and stdout filtering.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import io
import logging

from dnadesign.densegen.src.adapters.sources import pwm_sampling
from dnadesign.densegen.src.utils import logging_utils


def test_pwm_progress_line_densegen_fields() -> None:
    line = pwm_sampling._format_pwm_progress_line(
        motif_id="M1",
        backend="densegen",
        generated=50,
        target=100,
        accepted=None,
        accepted_target=None,
        batch_index=None,
        batch_total=None,
        elapsed=1.2,
    )
    assert line.startswith("PWM M1 | densegen | gen 50% (50/100)")
    assert "acc" not in line
    assert "| 1.2s |" in line
    assert line.endswith("/s")


def test_pwm_progress_line_fimo_fields() -> None:
    line = pwm_sampling._format_pwm_progress_line(
        motif_id="M2",
        backend="fimo",
        generated=25,
        target=100,
        accepted=10,
        accepted_target=40,
        batch_index=3,
        batch_total=None,
        elapsed=2.5,
    )
    assert line.startswith("PWM M2 | fimo | gen 25% (25/100) | batch 3/-")
    assert "| 2.5s |" in line
    assert line.endswith("/s")


def test_fimo_mining_batch_filter_blocks_batch_lines() -> None:
    record = logging.LogRecord(
        name="dnadesign.densegen.src.adapters.sources.pwm_sampling",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="FIMO mining M1 batch 2/-: generated=10/20 accepted=5",
        args=(),
        exc_info=None,
    )
    filt = logging_utils.FimoMiningBatchLogFilter()
    assert filt.filter(record) is False


def test_fimo_mining_batch_filter_allows_non_batch_lines() -> None:
    record = logging.LogRecord(
        name="dnadesign.densegen.src.adapters.sources.pwm_sampling",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="FIMO debug TSV written: /tmp/fimo.tsv",
        args=(),
        exc_info=None,
    )
    filt = logging_utils.FimoMiningBatchLogFilter()
    assert filt.filter(record) is True


def test_fimo_mining_batch_filter_blocks_config_lines() -> None:
    record = logging.LogRecord(
        name="dnadesign.densegen.src.adapters.sources.pwm_sampling",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="FIMO mining config for M1: target=10 batch=2",
        args=(),
        exc_info=None,
    )
    filt = logging_utils.FimoMiningBatchLogFilter()
    assert filt.filter(record) is False


def test_fimo_mining_batch_filter_blocks_yield_lines() -> None:
    record = logging.LogRecord(
        name="dnadesign.densegen.src.adapters.sources.pwm_sampling",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="FIMO yield for motif M1: hits=10 accepted=9 selected=2",
        args=(),
        exc_info=None,
    )
    filt = logging_utils.FimoMiningBatchLogFilter()
    assert filt.filter(record) is False


def test_progress_handler_inserts_newline_when_progress_visible() -> None:
    stream = io.StringIO()
    handler = logging_utils.ProgressAwareStreamHandler(stream=stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("progress_handler_visible")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logging_utils.set_progress_active(True)
    logging_utils.mark_progress_line_visible()
    logger.info("hello")
    logging_utils.set_progress_active(False)
    output = stream.getvalue()
    assert output.startswith("\nhello\n")


def test_progress_handler_no_newline_when_progress_hidden() -> None:
    stream = io.StringIO()
    handler = logging_utils.ProgressAwareStreamHandler(stream=stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("progress_handler_hidden")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logging_utils.set_progress_active(False)
    logger.info("hello")
    output = stream.getvalue()
    assert output == "hello\n"


def test_progress_handler_suppresses_flagged_records() -> None:
    stream = io.StringIO()
    handler = logging_utils.ProgressAwareStreamHandler(stream=stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("progress_handler_suppress")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info("quiet", extra={"suppress_stdout": True})
    assert stream.getvalue() == ""


def test_progress_handler_closed_stream_avoids_logging_error(capsys) -> None:
    stream = io.StringIO()
    handler = logging_utils.ProgressAwareStreamHandler(stream=stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("progress_handler_closed")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream.close()
    raise_exceptions = logging.raiseExceptions
    logging.raiseExceptions = True
    try:
        logger.info("hello")
    finally:
        logging.raiseExceptions = raise_exceptions
        logger.handlers = []
    captured = capsys.readouterr()
    assert "Logging error" not in captured.err


class _TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_pwm_progress_dedupes_identical_updates() -> None:
    stream = _TtyBuffer()
    progress = pwm_sampling._PwmSamplingProgress(
        motif_id="M1",
        backend="densegen",
        target=10,
        accepted_target=None,
        stream=stream,
    )
    progress.update(generated=5, accepted=None, force=True)
    progress.update(generated=5, accepted=None, force=True)
    progress.finish()
    output = stream.getvalue()
    assert output.count("PWM M1") == 1
