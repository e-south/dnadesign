"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/rt_intervals.py

RT interval contract helpers for Eco1 RT repack manual mask authority.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue

EXPECTED_RT_INTERVAL_FEATURE_IDS = {
    "rt1_interval",
    "rt2_interval",
    "rt3_interval",
    "rt4_interval",
    "rt5_interval",
    "rt6_interval",
    "rt7_interval",
}


@dataclass(frozen=True)
class RTIntervalFeature:
    """One source-declared Eco1 RT core interval in canonical position space."""

    feature_id: str
    canonical_start: int
    canonical_end: int
    policy: str

    @property
    def canonical_positions(self) -> set[int]:
        """Return every canonical Eco1 position covered by the interval."""

        return set(range(self.canonical_start, self.canonical_end + 1))


def rt_interval_features_from_source(authority_source: Mapping[str, Any]) -> tuple[RTIntervalFeature, ...]:
    """Return RT1-RT7 interval review-label features from the ontology source."""

    authority_sets = authority_source.get("authority_sets")
    if not isinstance(authority_sets, list):
        return ()
    features: list[RTIntervalFeature] = []
    for authority_set in authority_sets:
        if not isinstance(authority_set, Mapping):
            continue
        if authority_set.get("authority_type") != "rt_core_interval":
            continue
        if authority_set.get("policy") != "review_label":
            continue
        for feature in _as_list(authority_set.get("features")):
            feature_id = feature.get("id")
            start = feature.get("start")
            end = feature.get("end")
            if isinstance(feature_id, str) and isinstance(start, int) and isinstance(end, int) and start <= end:
                features.append(
                    RTIntervalFeature(
                        feature_id=feature_id,
                        canonical_start=start,
                        canonical_end=end,
                        policy="review_label",
                    )
                )
    return tuple(sorted(features, key=lambda item: item.feature_id))


def rt_interval_feature_ids_from_source(authority_source: Mapping[str, Any]) -> set[str]:
    """Return first-class RT-region interval feature ids from the ontology source."""

    return {feature.feature_id for feature in rt_interval_features_from_source(authority_source)}


def validate_rt_interval_authority(
    manual_mask_authority: Mapping[str, Any],
    *,
    authority_source: Mapping[str, Any],
    path: Path,
) -> list[ContractIssue]:
    """Validate generated RT1-RT7 review-label rows against the checked-in ontology."""

    expected_features = rt_interval_features_from_source(authority_source)
    expected_ids = {feature.feature_id for feature in expected_features}
    expected_positions_by_id = {feature.feature_id: feature.canonical_positions for feature in expected_features}
    observed_positions_by_id = _generated_rt_interval_positions_by_id(manual_mask_authority)
    deferred = manual_mask_authority.get("deferred_authority")
    has_deferred_pending = bool(deferred) if isinstance(deferred, list) else True
    manual_rt_positions = _generated_manual_rt_interval_positions(manual_mask_authority)

    if (
        expected_ids != EXPECTED_RT_INTERVAL_FEATURE_IDS
        or set(observed_positions_by_id) != expected_ids
        or observed_positions_by_id != expected_positions_by_id
        or has_deferred_pending
        or manual_rt_positions
    ):
        return [
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_missing_rt_intervals",
                message=(
                    "manual_mask_authority.yaml must retain audited RT1-RT7 review-label records "
                    "without promoting them to fixed manual-mask rows"
                ),
                path=str(path),
            )
        ]
    return []


def _generated_rt_interval_positions_by_id(manual_mask_authority: Mapping[str, Any]) -> dict[str, set[int]]:
    features = manual_mask_authority.get("features")
    positions_by_id: dict[str, set[int]] = {}
    if not isinstance(features, list):
        return positions_by_id
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        if feature.get("authority_type") != "rt_core_interval" or feature.get("policy") != "review_label":
            continue
        feature_id = feature.get("feature_id")
        positions = feature.get("canonical_positions")
        if isinstance(feature_id, str) and isinstance(positions, list):
            positions_by_id[feature_id] = {int(position) for position in positions if isinstance(position, int)}
    return positions_by_id


def _generated_manual_rt_interval_positions(manual_mask_authority: Mapping[str, Any]) -> set[int]:
    rows = manual_mask_authority.get("residues")
    if not isinstance(rows, list):
        return set()
    return {
        int(row["canonical_position"])
        for row in rows
        if (
            isinstance(row, Mapping)
            and isinstance(row.get("canonical_position"), int)
            and row.get("manual_mask") is True
            and _row_has_rt_interval_reason(row)
        )
    }


def _row_has_rt_interval_reason(row: Mapping[str, Any]) -> bool:
    return any(
        str(reason).startswith("rt") and "core_interval" in str(reason)
        for reason in str(row.get("manual_mask_reason", "")).split(";")
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
