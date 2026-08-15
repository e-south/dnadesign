"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/selection_view_performance.py

Public OPAL analysis for observed performance across selection views and rounds.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..src.analysis.selection_views.performance import (
    SelectionViewPerformance,
    selection_view_performance,
)
from ..src.plots.selection_view_performance import render_selection_view_performance

SELECTION_VIEW_PERFORMANCE_API_VERSION = "1"

__all__ = [
    "SELECTION_VIEW_PERFORMANCE_API_VERSION",
    "SelectionViewPerformance",
    "render_selection_view_performance",
    "selection_view_performance",
]
