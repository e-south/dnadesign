"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/__init__.py

Public snapback workflow contracts for Cruncher.

Module Author(s): Eric J. South
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
from dnadesign.cruncher.snapback.primitive_exports import (
    SnapbackCapPrimitive,
    SnapbackPrimitiveExportError,
    load_released_solve_cap_primitives,
)
from dnadesign.cruncher.snapback.solve_models import SingleNickSnapbackSolveSpec, SnapbackSolveReport
from dnadesign.cruncher.snapback.solver import render_solve_markdown_report, solve_snapback_search
from dnadesign.cruncher.snapback.target_models import SnapbackTargetSearchReport
from dnadesign.cruncher.snapback.target_search import render_target_search_markdown_report, search_snapback_target_hits

__all__ = [
    "SnapbackCapPrimitive",
    "SnapbackError",
    "SnapbackEvaluationReport",
    "SnapbackPlanningError",
    "SnapbackPrimitiveExportError",
    "SnapbackSolveReport",
    "SnapbackTargetSearchReport",
    "SnapbackSpecError",
    "SingleNickSnapbackSolveSpec",
    "SingleNickSnapbackSpec",
    "build_snapback_report",
    "load_snapback_solve_spec",
    "load_snapback_spec",
    "load_released_solve_cap_primitives",
    "render_markdown_report",
    "render_solve_markdown_report",
    "render_target_search_markdown_report",
    "resolve_workspace_root_for_snapback_solve_spec",
    "resolve_workspace_root_for_snapback_spec",
    "search_snapback_target_hits",
    "solve_snapback_search",
]
