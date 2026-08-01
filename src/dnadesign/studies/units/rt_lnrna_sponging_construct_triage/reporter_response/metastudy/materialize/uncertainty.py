"""Descriptive uncertainty construction for reporter-response profiles."""

from __future__ import annotations

from ...profile.uncertainty import DoseUncertainty, NotEstimableMetricUncertainty
from .temporal import _reduce


def _descriptive_uncertainty(
    *,
    dose: float,
    values: list[tuple[float, float]],
    statistic: str,
    minimum_replicates: int,
    identity_complete: bool,
) -> DoseUncertainty:
    """Describe one dose without claiming unsupported inferential uncertainty."""

    count = len(values)
    if count == 0:
        raise ValueError("dose uncertainty requires acquisition observations")
    if not identity_complete:
        reason = "biological_replicate_identity_unknown"
        biological_replicate_count = 0
    elif count < minimum_replicates:
        reason = "below_minimum_biological_replicates"
        biological_replicate_count = count
    else:
        # Materialization remains descriptive until a biological-replicate
        # resampling method is explicitly selected and validated.
        reason = "insufficient_valid_resamples"
        biological_replicate_count = count
    return DoseUncertainty(
        dose_uM=dose,
        biological_replicate_count=biological_replicate_count,
        normalized_reporter_response=NotEstimableMetricUncertainty(
            estimate=_reduce((row[0] for row in values), statistic),
            reason=reason,
        ),
        relative_od=NotEstimableMetricUncertainty(
            estimate=_reduce((row[1] for row in values), statistic),
            reason=reason,
        ),
    )
