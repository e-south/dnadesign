"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/annotations.py

Helpers for reading and matching USR-backed sequence annotations inside
Construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


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
        assert self.start_0 is not None
        assert self.end_0 is not None
        return (float(self.start_0) + float(self.end_0)) / 2.0


def load_annotation_features(
    row: Mapping[str, object],
    *,
    namespace: str = "seq_annot",
) -> list[AnnotationFeature]:
    raw_features = row.get(f"{namespace}__features")
    if not isinstance(raw_features, list):
        return []
    features: list[AnnotationFeature] = []
    for raw_feature in raw_features:
        if not isinstance(raw_feature, Mapping):
            continue
        intervals: list[AnnotationInterval] = []
        raw_intervals = raw_feature.get("intervals_0")
        if isinstance(raw_intervals, list):
            for raw_interval in raw_intervals:
                if not isinstance(raw_interval, Mapping):
                    continue
                start = raw_interval.get("start_0")
                end = raw_interval.get("end_0")
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                intervals.append(
                    AnnotationInterval(
                        start_0=int(start),
                        end_0=int(end),
                        strand=_optional_int(raw_interval.get("strand")),
                        partial=bool(raw_interval.get("partial")),
                    )
                )
        features.append(
            AnnotationFeature(
                feature_id=_required_text(raw_feature.get("feature_id"), fallback="feature"),
                feature_order=int(raw_feature.get("feature_order") or 0),
                feature_type=_required_text(raw_feature.get("feature_type"), fallback="feature"),
                label=_optional_text(raw_feature.get("label")),
                role_hint=_optional_text(raw_feature.get("role_hint")),
                start_0=_optional_int(raw_feature.get("start_0")),
                end_0=_optional_int(raw_feature.get("end_0")),
                intervals_0=tuple(intervals),
                confidence=_optional_text(raw_feature.get("confidence")),
            )
        )
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


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
