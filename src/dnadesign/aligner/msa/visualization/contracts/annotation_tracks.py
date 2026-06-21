"""Annotation-track contract parsing for generic MSA visualizations."""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.aligner.msa.visualization.contracts.models import (
    AnnotationFeature,
    AnnotationTrack,
)
from dnadesign.aligner.msa.visualization.renderers.feature_labels import validate_label_position


def load_annotation_tracks(path: Path | None) -> tuple[AnnotationTrack, ...]:
    """Load display-only target-position annotation tracks."""

    if path is None:
        return ()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("annotation tracks YAML must be a mapping")
    if payload.get("coordinate_space") != "target_ungapped_position":
        raise ValueError("annotation tracks coordinate_space must be target_ungapped_position")
    raw_tracks = payload.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise ValueError("annotation tracks YAML must define a non-empty tracks list")
    tracks: list[AnnotationTrack] = []
    for raw_track in raw_tracks:
        if not isinstance(raw_track, dict):
            raise ValueError("annotation track entries must be mappings")
        track_id = _required_string(raw_track, "id")
        label = _required_string(raw_track, "label")
        color = raw_track.get("color", "#666666")
        if not isinstance(color, str) or not color:
            raise ValueError(f"annotation track {track_id} color must be a non-empty string")
        raw_features = raw_track.get("features")
        if not isinstance(raw_features, list) or not raw_features:
            raise ValueError(f"annotation track {track_id} must define a non-empty features list")
        features = tuple(_parse_feature(raw_feature, track_id=track_id, color=color) for raw_feature in raw_features)
        tracks.append(AnnotationTrack(id=track_id, label=label, color=color, features=features))
    return tuple(tracks)


def validate_annotation_track_ranges(
    *,
    profile_id: str,
    tracks: tuple[AnnotationTrack, ...],
    canonical_position_count: int,
) -> None:
    """Validate annotations against a rendered target coordinate space."""

    for track in tracks:
        for feature in track.features:
            if feature.end > canonical_position_count:
                raise ValueError(
                    f"{profile_id} annotation feature {feature.id} range "
                    f"{feature.start}-{feature.end} is outside target position range "
                    f"1-{canonical_position_count}"
                )


def _parse_feature(raw_feature: object, *, track_id: str, color: str) -> AnnotationFeature:
    if not isinstance(raw_feature, dict):
        raise ValueError(f"annotation track {track_id} feature entries must be mappings")
    feature_id = _required_string(raw_feature, "id")
    feature_label = _required_string(raw_feature, "label")
    start = _required_positive_int(raw_feature, "start", feature_id)
    end = _required_positive_int(raw_feature, "end", feature_id)
    if start > end:
        raise ValueError(f"annotation feature {feature_id} start must be <= end")
    feature_color = raw_feature.get("color", color)
    if not isinstance(feature_color, str) or not feature_color:
        raise ValueError(f"annotation feature {feature_id} color must be a non-empty string")
    fill_opacity = _optional_float(
        raw_feature,
        "fill_opacity",
        feature_id=feature_id,
        default=0.84,
        minimum=0.0,
        maximum=1.0,
    )
    stroke_color = raw_feature.get("stroke_color", feature_color)
    if not isinstance(stroke_color, str) or not stroke_color:
        raise ValueError(f"annotation feature {feature_id} stroke_color must be a non-empty string")
    stroke_width = _optional_float(
        raw_feature,
        "stroke_width",
        feature_id=feature_id,
        default=0.0,
        minimum=0.0,
        maximum=10.0,
    )
    text_color = raw_feature.get("text_color")
    if text_color is not None and (not isinstance(text_color, str) or not text_color):
        raise ValueError(f"annotation feature {feature_id} text_color must be a non-empty string")
    return AnnotationFeature(
        id=feature_id,
        label=feature_label,
        start=start,
        end=end,
        color=feature_color,
        fill_opacity=fill_opacity,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        text_color=text_color,
        label_position=validate_label_position(raw_feature.get("label_position"), feature_id=feature_id),
    )


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"annotation {key} must be a non-empty string")
    return value


def _required_positive_int(payload: dict[str, object], key: str, feature_id: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"annotation feature {feature_id} {key} must be a positive integer")
    return value


def _optional_float(
    payload: dict[str, object],
    key: str,
    *,
    feature_id: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"annotation feature {feature_id} {key} must be a number")
    result = float(value)
    if result < minimum or result > maximum:
        raise ValueError(f"annotation feature {feature_id} {key} must be between {minimum:g} and {maximum:g}")
    return result
