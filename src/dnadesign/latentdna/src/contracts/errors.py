"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/contracts/errors.py

Error taxonomy and exit codes for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class LatentDNAError(RuntimeError):
    exit_code = 19


class WorkspaceValidationError(LatentDNAError):
    exit_code = 10


class SourceResolutionError(LatentDNAError):
    exit_code = 11


class AlignmentError(LatentDNAError):
    exit_code = 12


class MissingArtifactError(LatentDNAError):
    exit_code = 13


class CoordinateSpaceError(LatentDNAError):
    exit_code = 14


class ArtifactConflictError(LatentDNAError):
    exit_code = 15


class IOErrorError(LatentDNAError):
    exit_code = 16


class BackendUnavailableError(LatentDNAError):
    exit_code = 17


class ContractViolationError(LatentDNAError):
    exit_code = 18


class MemoryPreflightError(LatentDNAError):
    exit_code = 20


class OperationLockError(LatentDNAError):
    exit_code = 21


def exit_code_for_error(exc: Exception) -> int:
    if isinstance(exc, LatentDNAError):
        return exc.exit_code
    return 19
