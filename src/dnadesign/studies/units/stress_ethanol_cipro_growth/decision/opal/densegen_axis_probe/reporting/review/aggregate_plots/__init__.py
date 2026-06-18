"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/review/aggregate_plots/__init__.py

Registry-backed aggregate plots for DenseGen axis probe reviews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .contracts import ProbeAggregatePlotSpec, build_probe_aggregate_plot_registry
from .writer import write_probe_aggregate_plots

__all__ = [
    "ProbeAggregatePlotSpec",
    "build_probe_aggregate_plot_registry",
    "write_probe_aggregate_plots",
]
