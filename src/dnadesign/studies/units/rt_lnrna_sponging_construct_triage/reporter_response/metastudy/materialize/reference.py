"""Reference-normalization availability from explicitly declared controls."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ... import ConditionMeasurement, ReferenceNormalizationUnavailable
from ..condition_ontology import ConditionDefinition


@dataclass(frozen=True, slots=True)
class ReferenceBasis:
    """Resolved baseline values and optional positive-control separation."""

    baseline_ratio: float
    baseline_od600: float
    separation: float | None
    unavailable: ReferenceNormalizationUnavailable | None


def resolve_reference_basis(
    *,
    baselines: list[ConditionMeasurement],
    positive: ConditionDefinition | None,
    positives: list[ConditionMeasurement],
) -> ReferenceBasis:
    """Resolve an optional stricter projection without inferring controls."""

    baseline_ratio = statistics.median(row.rfp_over_od600 for row in baselines)
    baseline_od600 = statistics.median(row.od600 for row in baselines)
    positive_ratio = statistics.median(row.rfp_over_od600 for row in positives) if positives else None
    separation = positive_ratio - baseline_ratio if positive_ratio is not None else None
    if separation is not None and separation > 0.0:
        return ReferenceBasis(baseline_ratio, baseline_od600, separation, None)
    reason = "positive_control_not_declared" if positive is None else "positive_control_separation_not_positive"
    return ReferenceBasis(
        baseline_ratio,
        baseline_od600,
        None,
        ReferenceNormalizationUnavailable(
            reason=reason,
            positive_control_condition_id=positive.condition_id if positive is not None else None,
        ),
    )


__all__ = ["ReferenceBasis", "resolve_reference_basis"]
