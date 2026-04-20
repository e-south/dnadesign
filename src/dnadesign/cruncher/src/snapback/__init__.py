"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/__init__.py

Public snapback workflow contracts for Cruncher.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from dnadesign.cruncher.snapback.errors import SnapbackError, SnapbackPlanningError, SnapbackSpecError
from dnadesign.cruncher.snapback.load import (
    load_snapback_solve_spec,
    load_snapback_spec,
    resolve_workspace_root_for_snapback_solve_spec,
    resolve_workspace_root_for_snapback_spec,
)
from dnadesign.cruncher.snapback.models import SingleNickSnapbackSpec, SnapbackEvaluationReport
from dnadesign.cruncher.snapback.planner import build_snapback_report, render_markdown_report
from dnadesign.cruncher.snapback.solve_models import SingleNickSnapbackSolveSpec, SnapbackSolveReport
from dnadesign.cruncher.snapback.solver import render_solve_markdown_report, solve_snapback_search

__all__ = [
    "SnapbackError",
    "SnapbackEvaluationReport",
    "SnapbackPlanningError",
    "SnapbackSolveReport",
    "SnapbackSpecError",
    "SingleNickSnapbackSolveSpec",
    "SingleNickSnapbackSpec",
    "build_snapback_report",
    "load_snapback_solve_spec",
    "load_snapback_spec",
    "render_markdown_report",
    "render_solve_markdown_report",
    "resolve_workspace_root_for_snapback_solve_spec",
    "resolve_workspace_root_for_snapback_spec",
    "solve_snapback_search",
]
