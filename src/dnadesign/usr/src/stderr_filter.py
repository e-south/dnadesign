"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/src/stderr_filter.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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
    if getattr(sys, "_usr_stderr_filter_installed", False):
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

    t = threading.Thread(target=_reader, daemon=True, name="usr-stderr-filter")
    t.start()
    setattr(sys, "_usr_stderr_filter_installed", True)


def maybe_install_pyarrow_sysctl_filter() -> None:
    """
    Suppress noisy PyArrow sysctlbyname warnings on macOS.
    Default behavior: only when stderr is a TTY.
    Override with USR_SUPPRESS_PYARROW_SYSCTL=1 to force suppression in non-TTY contexts,
    or USR_SUPPRESS_PYARROW_SYSCTL=0 to disable suppression entirely.
    """
    if sys.platform != "darwin":
        return
    flag_raw = os.getenv("USR_SUPPRESS_PYARROW_SYSCTL")
    flag = flag_raw.strip().lower() if flag_raw is not None else ""
    if flag_raw is None or flag == "":
        try:
            if not sys.stderr.isatty():
                return
        except Exception:
            return
    elif flag in {"0", "false", "no"}:
        return
    _install_stderr_filter(_PYARROW_SYSCTL_MATCH)
