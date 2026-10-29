"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/backends/mafft.py

MAFFT backend wrapper for generic MSA runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from dnadesign.aligner.msa.backends.execution import preflight_backend_executable, run_staged_backend_alignment
from dnadesign.aligner.msa.contracts import MsaBackendSpec, MsaRequest, MsaRunResult


class MafftUnavailableError(RuntimeError):
    """Raised when the declared MAFFT executable is unavailable."""


def preflight_mafft(spec: MsaBackendSpec | None = None) -> tuple[str, str]:
    """Return executable path and version for a declared MAFFT backend."""

    backend = spec or MsaBackendSpec(backend_id="mafft")
    return preflight_backend_executable(
        backend,
        display_name="MAFFT",
        parse_version=_parse_mafft_version,
        unavailable_error=MafftUnavailableError,
    )


def run_mafft(request: MsaRequest) -> MsaRunResult:
    """Run MAFFT and write an aligned FASTA bundle manifest."""

    executable_path, version = preflight_mafft(request.backend)
    return run_staged_backend_alignment(
        request,
        display_name="MAFFT",
        executable_path=executable_path,
        backend_version=version,
        build_command=lambda _temporary_output: (executable_path, *request.command_args, str(request.input_fasta)),
        stdout_target="temporary_output",
    )


def _parse_mafft_version(version_text: str) -> str:
    match = re.search(r"v?(\d+(?:\.\d+)+)", version_text)
    if match:
        return match.group(1)
    return version_text.splitlines()[0] if version_text else "unknown"
