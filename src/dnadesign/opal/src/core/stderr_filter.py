"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/core/stderr_filter.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Iterable

_PYARROW_SYSCTL_MATCH = ("arrow/util/cpu_info.cc", "sysctlbyname failed for")


def _line_matches(line: str, needles: Iterable[str]) -> bool:
    for n in needles:
        if n not in line:
            return False
    return True


def _install_stderr_filter(needles: Iterable[str]) -> None:
    if getattr(sys, "_opal_stderr_filter_installed", False):
        return

    # Duplicate the current stderr FD (respects OS-level redirection).
    try:
        orig_fd = os.dup(2)
    except Exception:
        return

    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    needles = tuple(needles)

    def _reader() -> None:
        buf = b""
        try:
            with os.fdopen(r_fd, "rb", closefd=True) as r:
                while True:
                    chunk = r.read(1024)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = line.decode(errors="replace")
                        if _line_matches(text, needles):
                            continue
                        os.write(orig_fd, line + b"\n")
                if buf:
                    text = buf.decode(errors="replace")
                    if not _line_matches(text, needles):
                        os.write(orig_fd, buf)
        except Exception:
            # Fail open: restore direct writes if filtering fails.
            try:
                os.dup2(orig_fd, 2)
            except Exception:
                pass

    t = threading.Thread(target=_reader, daemon=True, name="opal-stderr-filter")
    t.start()

    def _cleanup() -> None:
        if getattr(sys, "_opal_stderr_filter_cleaned", False):
            return
        setattr(sys, "_opal_stderr_filter_cleaned", True)
        # Restore stderr to the original FD, which also closes the pipe writer.
        try:
            os.dup2(orig_fd, 2)
        except Exception:
            pass
        # Give the reader a brief chance to flush remaining buffered lines.
        try:
            t.join(timeout=0.2)
        except Exception:
            pass
        try:
            os.close(orig_fd)
        except Exception:
            pass

    atexit.register(_cleanup)
    setattr(sys, "_opal_stderr_filter_cleanup", _cleanup)
    setattr(sys, "_opal_stderr_filter_installed", True)


def maybe_install_pyarrow_sysctl_filter() -> None:
    """
    Suppress noisy PyArrow sysctlbyname warnings on macOS.
    Default behavior: suppress in both TTY and non-TTY contexts.
    Override with OPAL_SUPPRESS_PYARROW_SYSCTL=0 to disable suppression (show raw warnings).
    """
    if sys.platform != "darwin":
        return
    flag_raw = os.getenv("OPAL_SUPPRESS_PYARROW_SYSCTL", "")
    flag = flag_raw.strip().lower()
    if flag in {"0", "false", "no", "off", "raw", "show"}:
        return
    if flag not in {"", "1", "true", "yes", "on"}:
        # Invalid override → be explicit but keep default (suppress).
        print(
            "Invalid OPAL_SUPPRESS_PYARROW_SYSCTL value. Expected 0/1/true/false/on/off/show/raw.",
            file=sys.stderr,
        )
    _install_stderr_filter(_PYARROW_SYSCTL_MATCH)
