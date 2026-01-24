"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/theme.py

Shared notebook helpers for OPAL marimo dashboards. Defines theme setup and
chart title utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import altair as alt


def setup_altair_theme() -> None:
    alt.data_transformers.disable_max_rows()

    @alt.theme.register("dnad_white", enable=True)
    def _dnad_white_theme():
        return alt.theme.ThemeConfig(
            {
                "config": {
                    "background": "white",
                    "legend": {
                        "labelColor": "#111111",
                        "titleColor": "#111111",
                        "labelFontSize": 11,
                        "titleFontSize": 12,
                        "fillColor": "transparent",
                        "fillOpacity": 0.0,
                        "strokeColor": None,
                    },
                    "title": {
                        "color": "#111111",
                        "subtitleColor": "#333333",
                        "fontSize": 14,
                        "subtitleFontSize": 11,
                    },
                    "axis": {
                        "domain": True,
                        "domainColor": "#111111",
                        "domainWidth": 1,
                        "grid": True,
                        "gridColor": "#e6e6e6",
                        "gridOpacity": 0.35,
                        "ticks": True,
                        "tickColor": "#111111",
                        "tickSize": 5,
                        "labels": True,
                        "labelFontSize": 11,
                        "labelColor": "#111111",
                        "titleFontSize": 12,
                        "titleColor": "#111111",
                        "labelPadding": 2,
                        "titlePadding": 4,
                    },
                    "axisX": {"domain": True},
                    "axisY": {"domain": True},
                    "view": {"stroke": None},
                }
            }
        )

    try:
        alt.theme.enable("dnad_white")
    except Exception:
        alt.themes.enable("dnad_white")


def _chart_title(text: str, subtitle: str | None = None) -> alt.TitleParams:
    if subtitle:
        return alt.TitleParams(text=text, subtitle=subtitle)
    return alt.TitleParams(text=text)


def with_title(chart, title: str, subtitle: str | None = None):
    return chart.properties(title=_chart_title(title, subtitle))
