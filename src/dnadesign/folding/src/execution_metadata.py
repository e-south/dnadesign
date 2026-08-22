"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/execution_metadata.py

Deterministic execution metadata shared by folding writers and replay.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prediction_command(
    *,
    interface: str,
    python_module: str | None,
    resolved_executable: str | Path | None,
    parameters: dict[str, Any],
) -> list[str]:
    """Return the exact command record for one supported backend interface."""
    if interface == "python_api":
        if python_module is None:
            raise ValueError("Python folding execution requires a module name.")
        return [f"{python_module}.fold_compound", "mfe"]
    if resolved_executable is None:
        raise ValueError("CLI folding execution requires a resolved executable.")
    command = [Path(resolved_executable).as_posix(), "--noPS"]
    temperature_c = parameters.get("temperature_c")
    if temperature_c is not None:
        command.extend(["--temp", f"{float(temperature_c):g}"])
    return command


def prediction_log_paths(*, interface: str) -> tuple[str, str]:
    """Return the distinct log names owned by one backend interface."""
    if interface == "python_api":
        return "ViennaRNA.python_api.stdout.txt", "ViennaRNA.python_api.stderr.txt"
    return "RNAfold.stdout.txt", "RNAfold.stderr.txt"


def exception_evidence_text(*, exception_type: str, message: str) -> str:
    """Return canonical backend-exception evidence for one generated stderr log."""
    return (
        json.dumps(
            {
                "contract": "folding_backend_exception_v1",
                "exception_type": exception_type,
                "message": message,
            },
            sort_keys=True,
        )
        + "\n"
    )


__all__ = ["exception_evidence_text", "prediction_command", "prediction_log_paths"]
