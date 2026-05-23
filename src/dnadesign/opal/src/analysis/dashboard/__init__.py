"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/__init__.py

Dashboard and diagnostic support helpers.

This package is not the canonical generated-notebook source surface. Current
generated OPAL notebooks are rendered from `analysis.notebook_template` and
`analysis.notebook_set_template`, with public helper imports routed through
`dnadesign.opal.notebooks.api.generated`. Some plot plugins still depend on
dashboard chart/math helpers, so this package remains live support code rather
than an archive target.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from . import (
    api,
    artifacts,
    charts,
    datasets,
    diagnostics,
    filters,
    hues,
    labels,
    models,
    scores,
    selection,
    transient,
    ui,
    util,
    views,
    y_ops,
)

__all__ = [
    "api",
    "artifacts",
    "charts",
    "datasets",
    "diagnostics",
    "filters",
    "hues",
    "labels",
    "models",
    "selection",
    "scores",
    "transient",
    "ui",
    "util",
    "views",
    "y_ops",
]
