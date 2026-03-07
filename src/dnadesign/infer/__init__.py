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

from .src.api import run_extract, run_generate, run_job
from .src.resource_contracts import validate_runbook_gpu_resources


__all__ = ["run_extract", "run_generate", "run_job", "validate_runbook_gpu_resources"]
