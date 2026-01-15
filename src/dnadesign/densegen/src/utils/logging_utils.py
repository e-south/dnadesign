"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/utils/logging_utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional


def _install_native_stderr_deduper(patterns: Iterable[str]) -> None:
    """
    Redirect the *process*'s stderr (FD=2) through a pipe and suppress repeated
    lines matching any of `patterns` (regex). The first time a pattern is seen,
    emit a single Python WARNING; subsequent native lines are dropped.

    This catches warnings printed by OR-Tools/absl from C++ (which bypass Python
    logging). Safe to call once; subsequent calls are no-ops.
    """
    # If we've already redirected, don't do it again.
    if getattr(_install_native_stderr_deduper, "_installed", False):
        return

    pats = [re.compile(p) for p in patterns]
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
                    for pat in pats:
                        if pat.search(text):
                            key = pat.pattern
                            if key not in seen:
                                seen.add(key)
                                # Emit one clean Python warning instead of the raw native spam
                                log.warning(
                                    "CBC backend does not support SetSolverSpecificParametersAsString; "
                                    "continuing without solver-specific parameter strings."
                                )
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


def setup_logging(
    level: str = "INFO",
    logfile: Optional[str] = None,
    *,
    suppress_solver_stderr: bool = True,
) -> None:
    """
    Configure root logging for console + optional file, and optionally install a
    native stderr deduper to suppress noisy OR-Tools/absl warnings.

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

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    root.setLevel(lvl)
    logging.getLogger(__name__).info("Logging initialized (level=%s)", level)

    if suppress_solver_stderr:
        _install_native_stderr_deduper(
            patterns=[
                r"SetSolverSpecificParametersAsString\(\) not supported by Cbc",
            ]
        )
