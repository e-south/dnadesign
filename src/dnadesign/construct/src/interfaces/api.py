"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/interfaces/api.py

Public construct API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from dnadesign.usr import SequencesError as USRSequencesError

from ..composition.review import publish_composition_review_svg
from ..composition.runtime import (
    LinearSsdnaCompositionResult,
    LinearSsdnaCompositionSummary,
    load_linear_ssdna_composition_config,
    run_linear_ssdna_composition,
    summarize_linear_ssdna_composition,
)
from ..contracts.config import JobConfig
from ..contracts.config import load_job_config as _load_job_config
from ..contracts.errors import ExecutionError, ValidationError
from ..orchestration.runtime import (
    PreflightResult,
    RunResult,
    _dry_run_result,
    _persist_construct_run,
    _planned_run_from_config,
)
from ..orchestration.runtime import (
    preflight_from_config as _runtime_preflight_from_config,
)


def _wrap_usr_error(
    exc: USRSequencesError,
    *,
    message: str,
    error_type: type[ValidationError] | type[ExecutionError],
) -> NoReturn:
    # Public construct API should not leak sibling-tool exception types.
    raise error_type(f"{message}: {exc}") from exc


def load_job_config(path: str | Path) -> tuple[JobConfig, Path]:
    return _load_job_config(path)


def preflight_from_config(path: str | Path) -> PreflightResult:
    try:
        return _runtime_preflight_from_config(path)
    except USRSequencesError as exc:
        _wrap_usr_error(
            exc,
            message="construct preflight failed while reading USR inputs",
            error_type=ValidationError,
        )


def run_from_config(path: str | Path, *, dry_run: bool = False) -> RunResult:
    try:
        planned = _planned_run_from_config(path)
    except USRSequencesError as exc:
        _wrap_usr_error(
            exc,
            message="construct run planning failed while reading USR inputs",
            error_type=ValidationError,
        )
    if dry_run:
        return _dry_run_result(planned)
    try:
        return _persist_construct_run(planned)
    except USRSequencesError as exc:
        _wrap_usr_error(
            exc,
            message="construct run failed while writing USR outputs",
            error_type=ExecutionError,
        )


__all__ = [
    "JobConfig",
    "LinearSsdnaCompositionResult",
    "LinearSsdnaCompositionSummary",
    "PreflightResult",
    "RunResult",
    "load_job_config",
    "load_linear_ssdna_composition_config",
    "preflight_from_config",
    "publish_composition_review_svg",
    "run_from_config",
    "run_linear_ssdna_composition",
    "summarize_linear_ssdna_composition",
    "Path",
]
