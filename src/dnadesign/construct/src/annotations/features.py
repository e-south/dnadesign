"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/annotations/features.py

Helpers for reading and matching USR-backed sequence annotations inside.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..contracts.errors import ValidationError


@dataclass(frozen=True)
class AnnotationInterval:
    start_0: int
    end_0: int
    strand: int | None
    partial: bool


@dataclass(frozen=True)
class AnnotationFeature:
    feature_id: str
    feature_order: int
    feature_type: str
    label: str | None
    role_hint: str | None
    start_0: int | None
    end_0: int | None
    intervals_0: tuple[AnnotationInterval, ...]
    confidence: str | None

    @property
    def has_precise_bounds(self) -> bool:
        return self.start_0 is not None and self.end_0 is not None and self.end_0 >= self.start_0

    @property
    def center_0(self) -> float:
        if not self.has_precise_bounds:
            raise ValueError(f"feature {self.feature_id!r} does not have precise bounds")
        if self.start_0 is None or self.end_0 is None:
            raise ValueError(f"feature {self.feature_id!r} does not have precise bounds")
        return (float(self.start_0) + float(self.end_0)) / 2.0


def load_annotation_features(
    row: Mapping[str, object],
    *,
    namespace: str = "seq_annot",
) -> list[AnnotationFeature]:
    raw_features = row.get(f"{namespace}__features")
    if raw_features is None:
        return []
    if not isinstance(raw_features, list):
        raise ValidationError(f"{namespace}__features must be a list when provided.")
    features: list[AnnotationFeature] = []
    for feature_index, raw_feature in enumerate(raw_features):
        if not isinstance(raw_feature, Mapping):
            raise ValidationError(f"{namespace}__features[{feature_index}] must be a mapping.")
        intervals: list[AnnotationInterval] = []
        raw_intervals = raw_feature.get("intervals_0")
        if raw_intervals is not None and not isinstance(raw_intervals, list):
            raise ValidationError(f"{namespace}__features[{feature_index}].intervals_0 must be a list.")
        for interval_index, raw_interval in enumerate(raw_intervals or []):
            if not isinstance(raw_interval, Mapping):
                raise ValidationError(
                    f"{namespace}__features[{feature_index}].intervals_0[{interval_index}] must be a mapping."
                )
            start = _required_int(
                raw_interval.get("start_0"),
                label=f"{namespace}__features[{feature_index}].intervals_0[{interval_index}].start_0",
            )
            end = _required_int(
                raw_interval.get("end_0"),
                label=f"{namespace}__features[{feature_index}].intervals_0[{interval_index}].end_0",
            )
            _validate_bounds(
                start,
                end,
                label=f"{namespace}__features[{feature_index}].intervals_0[{interval_index}]",
            )
            intervals.append(
                AnnotationInterval(
                    start_0=start,
                    end_0=end,
                    strand=_optional_int(
                        raw_interval.get("strand"),
                        label=f"{namespace}__features[{feature_index}].intervals_0[{interval_index}].strand",
                    ),
                    partial=bool(raw_interval.get("partial")),
                )
            )
        start_0 = _optional_int(raw_feature.get("start_0"), label=f"{namespace}__features[{feature_index}].start_0")
        end_0 = _optional_int(raw_feature.get("end_0"), label=f"{namespace}__features[{feature_index}].end_0")
        if start_0 is not None or end_0 is not None:
            if start_0 is None or end_0 is None:
                raise ValidationError(
                    f"{namespace}__features[{feature_index}] must define both start_0 and end_0 or neither."
                )
            _validate_bounds(start_0, end_0, label=f"{namespace}__features[{feature_index}]")
        feature = AnnotationFeature(
            feature_id=_required_text(raw_feature.get("feature_id"), fallback="feature"),
            feature_order=(
                _optional_int(
                    raw_feature.get("feature_order"),
                    label=f"{namespace}__features[{feature_index}].feature_order",
                )
                or 0
            ),
            feature_type=_required_text(raw_feature.get("feature_type"), fallback="feature"),
            label=_optional_text(raw_feature.get("label")),
            role_hint=_optional_text(raw_feature.get("role_hint")),
            start_0=start_0,
            end_0=end_0,
            intervals_0=tuple(intervals),
            confidence=_optional_text(raw_feature.get("confidence")),
        )
        features.append(feature)
    return features


def match_annotation_features(
    features: Iterable[AnnotationFeature],
    *,
    role_hint: str | None = None,
    labels: Iterable[str] = (),
) -> list[AnnotationFeature]:
    normalized_role = _optional_text(role_hint)
    normalized_labels = {text.casefold() for text in (_optional_text(label) for label in labels) if text is not None}
    matched: list[AnnotationFeature] = []
    for feature in features:
        label_matches = feature.label is not None and feature.label.casefold() in normalized_labels
        role_matches = normalized_role is not None and feature.role_hint == normalized_role
        if normalized_role is None and not normalized_labels:
            continue
        if role_matches or label_matches:
            matched.append(feature)
    return matched


def feature_intersects_interval(feature: AnnotationFeature, *, start_0: int, end_0: int) -> bool:
    return any(interval.end_0 > start_0 and interval.start_0 < end_0 for interval in feature.intervals_0)


def feature_intervals_as_dicts(feature: AnnotationFeature) -> list[dict[str, object]]:
    return [
        {
            "start_0": interval.start_0,
            "end_0": interval.end_0,
            "strand": interval.strand,
            "partial": interval.partial,
        }
        for interval in feature.intervals_0
    ]


def _required_text(value: object, *, fallback: str) -> str:
    text = _optional_text(value)
    return text or fallback


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _required_int(value: object, *, label: str) -> int:
    if value is None or value == "":
        raise ValidationError(f"{label} is required.")
    parsed = _optional_int(value, label=label)
    if parsed is None:
        raise ValidationError(f"{label} is required.")
    return parsed


def _optional_int(value: object, *, label: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{label} must be an integer.") from None


def _validate_bounds(start_0: int, end_0: int, *, label: str) -> None:
    if start_0 < 0 or end_0 < 0:
        raise ValidationError(f"{label} start_0/end_0 must be >= 0.")
    if start_0 > end_0:
        raise ValidationError(f"{label} start_0 must be <= end_0.")
