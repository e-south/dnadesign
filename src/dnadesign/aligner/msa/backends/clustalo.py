"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/backends/clustalo.py

Clustal Omega backend wrapper for generic MSA runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from dnadesign.aligner.msa.backends.execution import preflight_backend_executable, run_staged_backend_alignment
from dnadesign.aligner.msa.contracts import MsaBackendSpec, MsaRequest, MsaRunResult


class ClustalOmegaUnavailableError(RuntimeError):
    """Raised when the declared Clustal Omega executable is unavailable."""


def preflight_clustalo(spec: MsaBackendSpec | None = None) -> tuple[str, str]:
    """Return executable path and version for a declared Clustal Omega backend."""

    backend = spec or MsaBackendSpec(backend_id="clustalo")
    return preflight_backend_executable(
        backend,
        display_name="Clustal Omega",
        parse_version=_parse_clustalo_version,
        unavailable_error=ClustalOmegaUnavailableError,
    )


def run_clustalo(request: MsaRequest) -> MsaRunResult:
    """Run Clustal Omega and write an aligned FASTA bundle manifest."""

    _reject_io_args(request.command_args)
    executable_path, version = preflight_clustalo(request.backend)
    return run_staged_backend_alignment(
        request,
        display_name="Clustal Omega",
        executable_path=executable_path,
        backend_version=version,
        build_command=lambda temporary_output: (
            executable_path,
            *request.command_args,
            "-i",
            str(request.input_fasta),
            "-o",
            str(temporary_output),
        ),
        stdout_target="stderr_log",
    )


def _reject_io_args(command_args: tuple[str, ...]) -> None:
    forbidden = {"-i", "--infile", "--in", "-o", "--outfile", "--out"}
    present = sorted(arg for arg in command_args if arg in forbidden)
    if present:
        raise ValueError(
            f"Clustal Omega command_args must not include input/output file flags; got {', '.join(present)}"
        )


def _parse_clustalo_version(version_text: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)+)", version_text)
    if match:
        return match.group(1)
    return version_text.splitlines()[0] if version_text else "unknown"
