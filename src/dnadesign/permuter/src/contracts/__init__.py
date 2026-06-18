"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/contracts/__init__.py

Metric-contract facade for canonical Permuter column helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.permuter.src.contracts.metrics import (
    expected_metric_column,
    interaction_metric_column,
    metric_id_from_observed_column,
    observed_metric_column,
    observed_metric_ids,
    observed_metric_subcolumn,
    reject_legacy_metric_columns,
)

__all__ = [
    "expected_metric_column",
    "interaction_metric_column",
    "metric_id_from_observed_column",
    "observed_metric_column",
    "observed_metric_ids",
    "observed_metric_subcolumn",
    "reject_legacy_metric_columns",
]
