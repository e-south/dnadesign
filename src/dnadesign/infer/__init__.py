"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/__init__.py

Public API:
  - run_extract
  - run_generate
  - run_job (YAML-driven)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# Side-effect import: registers models & functions (evo2.*, etc.)
from . import adapters  # noqa: F401
from .api import run_extract, run_generate, run_job

__all__ = ["run_extract", "run_generate", "run_job"]
