"""Cross-profile comparability checks."""

from __future__ import annotations

from collections.abc import Iterable

from ._contract_values import ReporterResponseContractError
from .profile.normalized import ReporterResponseProfile


def require_comparable_profiles(profiles: Iterable[ReporterResponseProfile]) -> str:
    """Return the shared comparability key or reject aggregation."""

    rows = tuple(profiles)
    if len(rows) < 2:
        raise ReporterResponseContractError("cross-profile aggregation requires at least two profiles")
    if not all(isinstance(row, ReporterResponseProfile) for row in rows):
        raise ReporterResponseContractError("aggregation inputs must be ReporterResponseProfile values")
    expected = rows[0].comparability_key
    mismatches = [row.profile_id for row in rows[1:] if row.comparability_key != expected]
    if mismatches:
        raise ReporterResponseContractError(
            "cross-profile aggregation requires exactly matching comparability keys; "
            f"mismatched profiles: {', '.join(mismatches)}"
        )
    return expected


__all__ = ["require_comparable_profiles"]
