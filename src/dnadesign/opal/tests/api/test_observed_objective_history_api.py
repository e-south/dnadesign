"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/api/test_observed_objective_history_api.py

Public API checks for observed objective history contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.opal.api import (
    OBSERVED_OBJECTIVE_HISTORY_API_VERSION,
    RUN_SERIES_SCHEMA_VERSION,
    observed_objective_run_contract_sha256,
)


def test_observed_objective_history_public_contract_is_versioned() -> None:
    assert OBSERVED_OBJECTIVE_HISTORY_API_VERSION == "v1"
    assert RUN_SERIES_SCHEMA_VERSION == "opal.observed_objective_run_series.v1"
    assert callable(observed_objective_run_contract_sha256)
