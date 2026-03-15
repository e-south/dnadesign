"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/__init__.py

Public API:
  - run_extract
  - run_generate
  - run_job (YAML-driven)
  - validate_runbook_gpu_resources

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_extract(*args: Any, **kwargs: Any):
    from .src.api import run_extract as _run_extract

    return _run_extract(*args, **kwargs)


def run_generate(*args: Any, **kwargs: Any):
    from .src.api import run_generate as _run_generate

    return _run_generate(*args, **kwargs)


def run_job(*args: Any, **kwargs: Any):
    from .src.api import run_job as _run_job

    return _run_job(*args, **kwargs)


def validate_runbook_gpu_resources(
    *,
    config_path: Path,
    declared_gpus: int,
    gpu_capability: str | None,
    gpu_memory_gib: float | None,
) -> None:
    from .src.resource_contracts import validate_runbook_gpu_resources as _validate_runbook_gpu_resources

    _validate_runbook_gpu_resources(
        config_path=config_path,
        declared_gpus=declared_gpus,
        gpu_capability=gpu_capability,
        gpu_memory_gib=gpu_memory_gib,
    )


__all__ = (
    "run_extract",
    "run_generate",
    "run_job",
    "validate_runbook_gpu_resources",
)
