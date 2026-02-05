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
    except OSError as e:
        raise RuntimeError("Failed to duplicate stderr for filtering.") from e

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
        except OSError as e:
            os.dup2(orig_fd, 2)
            raise RuntimeError("stderr filter failed while reading.") from e

    t = threading.Thread(target=_reader, daemon=True, name="usr-stderr-filter")
    t.start()
    setattr(sys, "_usr_stderr_filter_installed", True)


def _parse_env_flag(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    flag = raw.strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be 0/1/true/false/yes/no.")


def should_filter_pyarrow_sysctl() -> bool:
    if sys.platform != "darwin":
        return False
    suppress = _parse_env_flag("USR_SUPPRESS_PYARROW_SYSCTL")
    if suppress is not None:
        return suppress
    show = _parse_env_flag("USR_SHOW_PYARROW_SYSCTL")
    if show is not None:
        return not show
    return True


def maybe_install_pyarrow_sysctl_filter() -> None:
    """
    Suppress noisy PyArrow sysctlbyname warnings on macOS.
    Default suppress on macOS, opt-out with USR_SHOW_PYARROW_SYSCTL=1.
    Back-compat: USR_SUPPRESS_PYARROW_SYSCTL=1/0 takes precedence when set.
    """
    if should_filter_pyarrow_sysctl():
        _install_stderr_filter(_PYARROW_SYSCTL_MATCH)
