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


def cli_failure_evidence_text(*, returncode: int, backend_stderr: str) -> str:
    """Return canonical evidence for one nonzero CLI process outcome."""
    if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode == 0:
        raise ValueError("CLI failure evidence requires a nonzero integer return code.")
    return (
        json.dumps(
            {
                "backend_stderr": backend_stderr,
                "contract": "folding_cli_failure_v1",
                "returncode": returncode,
            },
            sort_keys=True,
        )
        + "\n"
    )


def parse_cli_failure_evidence(content: str) -> tuple[int, str]:
    """Parse and require the canonical nonzero CLI process evidence form."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("CLI failure evidence is not valid JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"backend_stderr", "contract", "returncode"}:
        raise ValueError("CLI failure evidence has an unsupported shape.")
    returncode = payload["returncode"]
    backend_stderr = payload["backend_stderr"]
    if payload["contract"] != "folding_cli_failure_v1" or not isinstance(backend_stderr, str):
        raise ValueError("CLI failure evidence has invalid fields.")
    expected = cli_failure_evidence_text(returncode=returncode, backend_stderr=backend_stderr)
    if content != expected:
        raise ValueError("CLI failure evidence is not canonical.")
    return returncode, backend_stderr


__all__ = [
    "cli_failure_evidence_text",
    "exception_evidence_text",
    "parse_cli_failure_evidence",
    "prediction_command",
    "prediction_log_paths",
]
