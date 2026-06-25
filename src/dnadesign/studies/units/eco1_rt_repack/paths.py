"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/paths.py

Study-local path contracts for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

STUDY_PACKAGE_ROOT = Path("src/dnadesign/studies/units/eco1_rt_repack")
DEFAULT_WORKSPACE_ID = "eco1_rt_conservative_v1"
DEFAULT_WORKSPACE_ROOT = STUDY_PACKAGE_ROOT / "workspaces" / DEFAULT_WORKSPACE_ID
DEFAULT_THREAD_OUTPUT_ROOT = DEFAULT_WORKSPACE_ROOT / "outputs" / "thread"
