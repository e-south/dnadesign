"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/feature_retention.py

Feature-retention classification for Construct-derived windows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from .annotations import AnnotationFeature, feature_intervals_as_dicts


@dataclass(frozen=True)
class FeatureRetentionSummary:
    retained: list[dict[str, object]]
    clipped: list[dict[str, object]]
    lost: list[dict[str, object]]


def classify_feature_retention(
    *,
    features: list[AnnotationFeature],
    source_start_0: int,
    source_end_0: int,
    derived_start_offset_0: int = 0,
) -> FeatureRetentionSummary:
    retained: list[dict[str, object]] = []
    clipped: list[dict[str, object]] = []
    lost: list[dict[str, object]] = []
    for feature in features:
        status, payload = _classify_one_feature(
            feature=feature,
            source_start_0=source_start_0,
            source_end_0=source_end_0,
            derived_start_offset_0=derived_start_offset_0,
        )
        if status == "retained":
            retained.append(payload)
        elif status == "clipped":
            clipped.append(payload)
        else:
            lost.append(payload)
    return FeatureRetentionSummary(retained=retained, clipped=clipped, lost=lost)


def _classify_one_feature(
    *,
    feature: AnnotationFeature,
    source_start_0: int,
    source_end_0: int,
    derived_start_offset_0: int,
) -> tuple[str, dict[str, object]]:
    original_intervals = list(feature_intervals_as_dicts(feature))
    retained_intervals: list[dict[str, object]] = []
    clipped_intervals: list[dict[str, object]] = []
    clipped_bp_total = 0

    for interval in feature.intervals_0:
        overlap_start = max(interval.start_0, source_start_0)
        overlap_end = min(interval.end_0, source_end_0)
        if overlap_end <= overlap_start:
            clipped_bp_total += interval.end_0 - interval.start_0
            continue
        translated_interval = {
            "start_0": derived_start_offset_0 + overlap_start - source_start_0,
            "end_0": derived_start_offset_0 + overlap_end - source_start_0,
            "strand": interval.strand,
            "partial": interval.partial,
        }
        if interval.start_0 >= source_start_0 and interval.end_0 <= source_end_0:
            retained_intervals.append(translated_interval)
        else:
            clipped_intervals.append(translated_interval)
            clipped_bp_total += (overlap_start - interval.start_0) + (interval.end_0 - overlap_end)

    if feature.intervals_0 and len(retained_intervals) == len(feature.intervals_0) and not clipped_intervals:
        return (
            "retained",
            _retention_payload(
                feature=feature,
                status="retained",
                original_intervals=original_intervals,
                derived_intervals=retained_intervals,
                clipped_bp=0,
                reason=None,
            ),
        )
    if retained_intervals or clipped_intervals:
        return (
            "clipped",
            _retention_payload(
                feature=feature,
                status="clipped",
                original_intervals=original_intervals,
                derived_intervals=[*retained_intervals, *clipped_intervals],
                clipped_bp=clipped_bp_total or None,
                reason="partially_retained",
            ),
        )
    return (
        "lost",
        _retention_payload(
            feature=feature,
            status="lost",
            original_intervals=original_intervals,
            derived_intervals=None,
            clipped_bp=None,
            reason="no_interval_overlap",
        ),
    )


def _retention_payload(
    *,
    feature: AnnotationFeature,
    status: str,
    original_intervals: list[dict[str, object]],
    derived_intervals: list[dict[str, object]] | None,
    clipped_bp: int | None,
    reason: str | None,
) -> dict[str, object]:
    return {
        "feature_id": feature.feature_id,
        "label": feature.label,
        "role_hint": feature.role_hint,
        "feature_type": feature.feature_type,
        "status": status,
        "original_intervals_0": original_intervals,
        "derived_intervals_0": derived_intervals,
        "clipped_bp": clipped_bp,
        "reason": reason,
    }
