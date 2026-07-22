"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/observed_objective_history.py

Public contract helpers for digest-bound observed objective history.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..src.analysis.observed_objective_history import (
    RUN_SERIES_SCHEMA_VERSION,
    observed_objective_run_contract_sha256,
)

OBSERVED_OBJECTIVE_HISTORY_API_VERSION = "v1"

__all__ = [
    "OBSERVED_OBJECTIVE_HISTORY_API_VERSION",
    "RUN_SERIES_SCHEMA_VERSION",
    "observed_objective_run_contract_sha256",
]
