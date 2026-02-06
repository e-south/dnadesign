"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/utils/logging_utils.py

Logging setup and stderr filtering utilities.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional, TextIO

from rich.console import Console
from rich.logging import RichHandler

_NATIVE_STDERR_PATTERNS: list[tuple[str, re.Pattern, str | None]] = []
_NATIVE_STDERR_LOCK = threading.Lock()
_FIMO_STDOUT_SUPPRESS_RE = re.compile(r"^\s*FIMO (mining|yield)\b")
_STAGE_A_VERBOSE_RE = re.compile(r"^\s*Stage-A (fimo batch|postprocess|diversity)\b")
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_ENABLED = True
_PROGRESS_ACTIVE = False
_PROGRESS_VISIBLE = False
_PROGRESS_STYLE = "stream"
_LOGGING_CONSOLE: Console | None = None


def set_progress_enabled(enabled: bool) -> None:
    global _PROGRESS_ENABLED, _PROGRESS_ACTIVE, _PROGRESS_VISIBLE
    with _PROGRESS_LOCK:
        _PROGRESS_ENABLED = bool(enabled)
        if not _PROGRESS_ENABLED:
            _PROGRESS_ACTIVE = False
            _PROGRESS_VISIBLE = False


def set_progress_style(style: str) -> None:
    global _PROGRESS_STYLE
    with _PROGRESS_LOCK:
        _PROGRESS_STYLE = str(style)


def get_progress_style() -> str:
    with _PROGRESS_LOCK:
        return str(_PROGRESS_STYLE)


def set_logging_console(console: Console | None) -> None:
    global _LOGGING_CONSOLE
    _LOGGING_CONSOLE = console


def get_logging_console() -> Console | None:
    return _LOGGING_CONSOLE


def is_progress_enabled() -> bool:
    with _PROGRESS_LOCK:
        return bool(_PROGRESS_ENABLED)


def set_progress_active(active: bool) -> None:
    global _PROGRESS_ACTIVE, _PROGRESS_VISIBLE
    with _PROGRESS_LOCK:
        _PROGRESS_ACTIVE = bool(active)
        if not _PROGRESS_ACTIVE:
            _PROGRESS_VISIBLE = False


def mark_progress_line_visible() -> None:
    global _PROGRESS_VISIBLE
    with _PROGRESS_LOCK:
        if _PROGRESS_ACTIVE:
            _PROGRESS_VISIBLE = True


def is_progress_line_visible() -> bool:
    with _PROGRESS_LOCK:
        return bool(_PROGRESS_VISIBLE)


def _maybe_clear_progress_line(stream: TextIO) -> None:
    global _PROGRESS_VISIBLE
    with _PROGRESS_LOCK:
        if not (_PROGRESS_ACTIVE and _PROGRESS_VISIBLE):
            return
        _PROGRESS_VISIBLE = False
    stream.write("\n")
    stream.flush()


class FimoMiningBatchLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _FIMO_STDOUT_SUPPRESS_RE.search(message):
            return False
        if _STAGE_A_VERBOSE_RE.search(message):
            return False
        return True


class FontToolsConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("fontTools") and record.levelno < logging.WARNING:
            return False
        return True


def _configure_fonttools_logger() -> None:
    logger = logging.getLogger("fontTools")
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True


class ProgressAwareStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self.stream, "closed", False):
            return
        if getattr(record, "suppress_stdout", False):
            return
        if _FIMO_STDOUT_SUPPRESS_RE.search(record.getMessage()):
            return
        if _STAGE_A_VERBOSE_RE.search(record.getMessage()):
            return
        _maybe_clear_progress_line(self.stream)
        try:
            super().emit(record)
        except ValueError:
            if getattr(self.stream, "closed", False):
                return
            raise


class ProgressAwareRichHandler(RichHandler):
    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "suppress_stdout", False):
            return
        message = record.getMessage()
        if _FIMO_STDOUT_SUPPRESS_RE.search(message):
            return
        if _STAGE_A_VERBOSE_RE.search(message):
            return
        stream = self.console.file if self.console is not None else sys.stdout
        if getattr(stream, "closed", False):
            return
        _maybe_clear_progress_line(stream)
        super().emit(record)


def _register_native_stderr_patterns(patterns: Iterable[tuple[str, str | None]]) -> None:
    with _NATIVE_STDERR_LOCK:
        existing = {pat for pat, _compiled, _msg in _NATIVE_STDERR_PATTERNS}
        for pat, msg in patterns:
            if pat in existing:
                continue
            _NATIVE_STDERR_PATTERNS.append((pat, re.compile(pat), msg))
            existing.add(pat)


def _install_native_stderr_deduper(patterns: Iterable[tuple[str, str | None]]) -> None:
    """
    Redirect the *process*'s stderr (FD=2) through a pipe and suppress repeated
    lines matching any of `patterns` (regex). The first time a pattern is seen,
    emit a single Python WARNING; subsequent native lines are dropped.

    This catches warnings printed by OR-Tools/absl from C++ (which bypass Python
    logging). Safe to call once; subsequent calls are no-ops.
    """
    _register_native_stderr_patterns(patterns)

    # If we've already redirected, don't do it again.
    if getattr(_install_native_stderr_deduper, "_installed", False):
        return

    log = logging.getLogger("densegen.stderr")

    # Duplicate current stderr FD (so we can still forward non-matching lines)
    try:
        orig_fd = os.dup(2)
    except Exception:
        # If we can't dup (unlikely), bail out silently.
        return

    r_fd, w_fd = os.pipe()
    try:
        os.dup2(w_fd, 2)  # redirect process-level stderr to our pipe writer
    finally:
        try:
            os.close(w_fd)
        except Exception:
            pass

    seen: set[str] = set()

    def reader() -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(r_fd, 4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace")
                    suppressed = False
                    with _NATIVE_STDERR_LOCK:
                        pats = list(_NATIVE_STDERR_PATTERNS)
                    for pat_str, pat, msg in pats:
                        if pat.search(text):
                            key = pat_str
                            if key not in seen:
                                seen.add(key)
                                if msg and log.hasHandlers():
                                    log.warning(msg)
                            suppressed = True
                            break
                    if not suppressed:
                        try:
                            os.write(orig_fd, line + b"\n")
                        except Exception:
                            pass
            except Exception:
                break

        try:
            os.close(r_fd)
        except Exception:
            pass
        try:
            os.close(orig_fd)
        except Exception:
            pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    _install_native_stderr_deduper._installed = True  # type: ignore[attr-defined]


_PYARROW_STDERR_PATTERNS = [
    (r"arrow/cpp/src/arrow/util/cpu_info\.cc", None),
    (r"sysctlbyname failed for 'hw\.", None),
]
_SOLVER_STDERR_PATTERNS = [
    (
        r"SetSolverSpecificParametersAsString\(\) not supported by Cbc",
        "CBC backend does not support SetSolverSpecificParametersAsString; "
        "continuing without solver-specific parameter strings.",
    ),
]


def install_native_stderr_filters(*, suppress_solver_messages: bool = True) -> None:
    patterns = list(_PYARROW_STDERR_PATTERNS)
    if suppress_solver_messages:
        patterns.extend(_SOLVER_STDERR_PATTERNS)
    _install_native_stderr_deduper(patterns=patterns)


def setup_logging(
    level: str = "INFO",
    logfile: Optional[str] = None,
    *,
    suppress_solver_stderr: bool = True,
) -> None:
    """
    Configure root logging for console + optional file, and optionally install a
    native stderr deduper to suppress noisy OR-Tools/absl warnings.
    When progress rendering is enabled, console logs are sent to stderr so
    screen/stream progress updates can own stdout.

    Parameters
    ----------
    level : str
        One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    logfile : Optional[str]
        Path to a log file. Created if missing.
    suppress_solver_stderr : bool
        If True, hide repeated native CBC messages like
        "SetSolverSpecificParametersAsString() not supported by Cbc ..."
    """
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)

    # Clear old handlers (in case of re-entry)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    progress_enabled = is_progress_enabled()
    progress_style = get_progress_style()
    if progress_enabled and progress_style == "screen":
        console_stream = sys.stdout
    elif progress_enabled and progress_style == "stream":
        if bool(getattr(sys.stderr, "isatty", lambda: False)()):
            console_stream = sys.stderr
        elif bool(getattr(sys.stdout, "isatty", lambda: False)()):
            console_stream = sys.stdout
        else:
            console_stream = sys.stdout
    else:
        console_stream = sys.stdout
    is_tty = bool(getattr(console_stream, "isatty", lambda: False)())
    console = Console(
        file=console_stream,
        force_terminal=is_tty,
        color_system="truecolor" if is_tty else None,
    )
    set_logging_console(console)
    sh = ProgressAwareRichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        log_time_format="%Y-%m-%d %H:%M:%S",
    )
    sh.setLevel(lvl)
    sh.addFilter(FimoMiningBatchLogFilter())
    sh.addFilter(FontToolsConsoleFilter())
    root.addHandler(sh)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _configure_fonttools_logger()

    root.setLevel(lvl)
    logging.getLogger(__name__).info("Logging initialized (level=%s)", level)

    install_native_stderr_filters(suppress_solver_messages=suppress_solver_stderr)
