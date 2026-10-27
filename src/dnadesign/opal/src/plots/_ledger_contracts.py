"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/_ledger_contracts.py

Metric-neutral validation for decision-plot ledger fields.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.utils import ExitCodes, OpalError


def validated_competition_ranks(values: pd.Series, *, objective_label: str) -> pd.Series:
    """Return positive integer ranks without accepting booleans or truncating fractions."""

    if values.isna().any() or values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise OpalError(
            f"{objective_label} selection ranks must be present positive integers.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    try:
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise OpalError(
            f"{objective_label} selection ranks must be positive integers.",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all() or np.any(numeric < 1.0):
        raise OpalError(
            f"{objective_label} selection ranks must be positive integers.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return pd.Series(numeric.astype(int), index=values.index, name=values.name)


def validated_selected_flags(values: pd.Series, *, objective_label: str) -> pd.Series:
    """Return exact Boolean selection flags without truthy string coercion."""

    if values.isna().any() or not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise OpalError(
            f"{objective_label} selected flags must be exact booleans.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return values.astype(bool)


__all__ = ["validated_competition_ranks", "validated_selected_flags"]
