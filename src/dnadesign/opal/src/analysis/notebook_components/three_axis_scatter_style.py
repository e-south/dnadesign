"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/three_axis_scatter_style.py

Define the publication design tokens for interactive three-axis scatters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

THREE_AXIS_TITLE_FONTSIZE = 23
THREE_AXIS_AXIS_TITLE_FONTSIZE = 17
THREE_AXIS_TICK_FONTSIZE = 14
THREE_AXIS_LEGEND_FONTSIZE = 14
THREE_AXIS_BASE_FONTSIZE = 15

OBSERVED_COLORS = (
    "#7C3AED",
    "#059669",
    "#DC2626",
    "#0891B2",
    "#A16207",
    "#DB2777",
    "#4F46E5",
    "#0F766E",
)
SELECTION_COLORS = ("#F59E0B", "#DB2777", "#7C3AED", "#0891B2", "#059669", "#DC2626")
SELECTION_SYMBOLS = ("diamond", "square", "x", "cross", "triangle-up", "triangle-down")


__all__ = [
    "OBSERVED_COLORS",
    "SELECTION_COLORS",
    "SELECTION_SYMBOLS",
    "THREE_AXIS_AXIS_TITLE_FONTSIZE",
    "THREE_AXIS_BASE_FONTSIZE",
    "THREE_AXIS_LEGEND_FONTSIZE",
    "THREE_AXIS_TICK_FONTSIZE",
    "THREE_AXIS_TITLE_FONTSIZE",
]
