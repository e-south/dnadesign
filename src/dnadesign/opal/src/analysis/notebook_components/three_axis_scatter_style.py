"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/three_axis_scatter_style.py

Define the publication design tokens for interactive three-axis scatters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

THREE_AXIS_TITLE_FONTSIZE = 23
THREE_AXIS_SUBTITLE_FONTSIZE = 16
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


def apply_three_axis_layout(
    figure: Any,
    *,
    title: str,
    subtitle: str,
    xaxis_title: str,
    yaxis_title: str,
    zaxis_title: str,
    camera_revision: str,
    complete_row_count: int,
    displayed_row_count: int,
    background_sample_limit: int,
) -> Any:
    """Apply the stable publication layout to an interactive three-axis figure."""

    axis_style = {
        "showbackground": True,
        "backgroundcolor": "#FAFAFA",
        "gridcolor": "#D1D5DB",
        "gridwidth": 1.0,
        "zeroline": True,
        "zerolinecolor": "#6B7280",
        "zerolinewidth": 1.5,
        "showspikes": False,
        "tickfont": {"size": THREE_AXIS_TICK_FONTSIZE, "color": "#252525"},
        "title": {"font": {"size": THREE_AXIS_AXIS_TITLE_FONTSIZE, "color": "#111827"}},
    }
    figure.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.96,
            "yanchor": "top",
            "font": {"size": THREE_AXIS_TITLE_FONTSIZE, "color": "#111827"},
        },
        annotations=(
            [
                {
                    "text": subtitle,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 1.035,
                    "xanchor": "center",
                    "yanchor": "top",
                    "showarrow": False,
                    "font": {"size": THREE_AXIS_SUBTITLE_FONTSIZE, "color": "#252525"},
                }
            ]
            if subtitle
            else []
        ),
        scene={
            "uirevision": camera_revision,
            "xaxis": {**axis_style, "title": {**axis_style["title"], "text": xaxis_title}},
            "yaxis": {**axis_style, "title": {**axis_style["title"], "text": yaxis_title}},
            "zaxis": {**axis_style, "title": {**axis_style["title"], "text": zaxis_title}},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.2}},
            "bgcolor": "white",
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.025,
            "yanchor": "top",
            "entrywidth": 0.32,
            "entrywidthmode": "fraction",
            "font": {"size": THREE_AXIS_LEGEND_FONTSIZE},
            "bgcolor": "rgba(255,255,255,0.88)",
        },
        font={"family": "Arial, Helvetica, sans-serif", "size": THREE_AXIS_BASE_FONTSIZE, "color": "#252525"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=800,
        margin={"l": 8, "r": 8, "t": 82, "b": 84, "autoexpand": False},
        hovermode="closest",
        uirevision=camera_revision,
        meta={
            "complete_row_count": int(complete_row_count),
            "displayed_row_count": int(displayed_row_count),
            "background_sample_limit": int(background_sample_limit),
        },
    )
    return figure


__all__ = [
    "OBSERVED_COLORS",
    "SELECTION_COLORS",
    "SELECTION_SYMBOLS",
    "apply_three_axis_layout",
    "THREE_AXIS_AXIS_TITLE_FONTSIZE",
    "THREE_AXIS_BASE_FONTSIZE",
    "THREE_AXIS_LEGEND_FONTSIZE",
    "THREE_AXIS_SUBTITLE_FONTSIZE",
    "THREE_AXIS_TICK_FONTSIZE",
    "THREE_AXIS_TITLE_FONTSIZE",
]
