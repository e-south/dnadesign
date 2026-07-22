"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/multistate_behavior_plot_labels.py

Stable view and coordinate labels for MSRB shadow figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

VIEW_ORDER = ("ethanol", "ciprofloxacin", "and")
VIEW_COLORS = {"ethanol": "#C97A20", "ciprofloxacin": "#3B78B5", "and": "#6C5AA7"}


def scenario_order(frame: pd.DataFrame) -> list[str]:
    quantiles = sorted(frame.loc[frame["scenario_kind"].eq("scale_quantile"), "scenario_id"].astype(str).unique())
    holdouts = sorted(
        frame.loc[frame["scenario_kind"].eq("leave_one_source_experiment_out"), "scenario_id"].astype(str).unique()
    )
    return [*quantiles, *holdouts]


def view_label(value: str) -> str:
    return "AND" if value == "and" else value.capitalize()


def objective_label(value: str) -> str:
    return "Behavior" if value == "multistate_response_behavior_v1" else "RMF"


def coordinate_label(value: str) -> str:
    family, state = value.split(":", maxsplit=1)
    state = state.translate(str.maketrans("01>", "₀₁›"))
    if family == "response":
        return f"Δr {state}"
    return f"b {state}" if family == "on_signal" else f"−b {state}"


__all__ = [
    "VIEW_COLORS",
    "VIEW_ORDER",
    "coordinate_label",
    "objective_label",
    "scenario_order",
    "view_label",
]
