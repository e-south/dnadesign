"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/delivery/secrets/shell_backend.py

External command probing and execution for notify secret backends.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess

from ...errors import NotifyConfigError


def probe_command(*, args: list[str], ok_codes: set[int], timeout_seconds: float = 2.0) -> bool:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return int(result.returncode) in ok_codes


def run_command(args: list[str], *, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            args,
            input=input_text,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise NotifyConfigError(f"failed to run secret backend command '{args[0]}'") from exc
    if result.returncode != 0:
        message = str(result.stderr or result.stdout or "").strip() or "unknown command failure"
        raise NotifyConfigError(f"secret backend command failed: {message}")
    return str(result.stdout or "").strip()
