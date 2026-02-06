"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/logging/test_logging_setup.py

Tests for logging setup behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys

from dnadesign.densegen.src.utils import logging_utils


def test_setup_logging_sets_file_handler_debug_level(tmp_path, monkeypatch) -> None:
    prev_handlers = list(logging.getLogger().handlers)
    prev_level = logging.getLogger().level
    monkeypatch.setattr(logging_utils, "install_native_stderr_filters", lambda **kwargs: None)

    log_path = tmp_path / "dense.log"
    logging_utils.setup_logging(level="INFO", logfile=str(log_path), suppress_solver_stderr=False)

    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers
    assert file_handlers[0].level == logging.DEBUG
    console_handlers = [h for h in root.handlers if isinstance(h, logging_utils.ProgressAwareRichHandler)]
    assert console_handlers
    handler = console_handlers[0]
    info_record = logging.LogRecord(
        name="fontTools.subset",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="noisy",
        args=(),
        exc_info=None,
    )
    warn_record = logging.LogRecord(
        name="fontTools.subset",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="warn",
        args=(),
        exc_info=None,
    )
    assert all(f.filter(info_record) for f in handler.filters) is False
    assert all(f.filter(warn_record) for f in handler.filters) is True

    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in prev_handlers:
        root.addHandler(handler)
    root.setLevel(prev_level)


def test_setup_logging_uses_stdout_for_screen_progress(monkeypatch) -> None:
    prev_handlers = list(logging.getLogger().handlers)
    prev_level = logging.getLogger().level
    monkeypatch.setattr(logging_utils, "install_native_stderr_filters", lambda **kwargs: None)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    logging_utils.set_progress_enabled(True)
    logging_utils.set_progress_style("screen")

    logging_utils.setup_logging(level="INFO", logfile=None, suppress_solver_stderr=False)

    root = logging.getLogger()
    console_handlers = [h for h in root.handlers if isinstance(h, logging_utils.ProgressAwareRichHandler)]
    assert console_handlers
    assert console_handlers[0].console.file is sys.stdout

    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in prev_handlers:
        root.addHandler(handler)
    root.setLevel(prev_level)
