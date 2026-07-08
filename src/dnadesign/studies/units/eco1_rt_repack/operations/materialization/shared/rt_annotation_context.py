"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/shared/rt_annotation_context.py

RT annotation context for Eco1 materialization visuals.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RT_ANNOTATION_TRACKS_SOURCE_LABEL = "docs/studies/eco1_rt_repack/workbench/ontology/rt-annotation-tracks.yaml"
MANUAL_MASK_AUTHORITY_SOURCE_LABEL = "docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml"


@dataclass(frozen=True)
class RTAnnotationFeature:
    """One display-only RT annotation feature in canonical Eco1 position space."""

    feature_id: str
    track_id: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class RTAnnotationContext:
    """Validated display-only RT annotation tracks for review figures."""

    target_sequence_hash: str
    annotation_tracks_path: Path
    manual_mask_authority_source_path: Path
    features: tuple[RTAnnotationFeature, ...]

    @property
    def source_paths(self) -> dict[str, Path]:
        """Return source paths for manifest hashing."""

        return {
            "rt_annotation_tracks": self.annotation_tracks_path,
            "manual_mask_authority_source": self.manual_mask_authority_source_path,
        }

    @property
    def source_table_labels(self) -> list[str]:
        """Return logical source labels for review-deliverable manifests."""

        return [RT_ANNOTATION_TRACKS_SOURCE_LABEL, MANUAL_MASK_AUTHORITY_SOURCE_LABEL]

    def features_for_track(self, track_id: str) -> tuple[RTAnnotationFeature, ...]:
        """Return features from one display track."""

        return tuple(feature for feature in self.features if feature.track_id == track_id)


def load_rt_annotation_context(
    *,
    annotation_tracks_path: Path,
    manual_mask_authority_source_path: Path,
) -> RTAnnotationContext:
    """Load and validate display-only RT annotation tracks against manual authority."""

    annotation_tracks = _load_yaml(annotation_tracks_path)
    manual_authority = _load_yaml(manual_mask_authority_source_path)
    annotation_hash = _require_text(annotation_tracks, "target_sequence_hash", path=annotation_tracks_path)
    manual_hash = _require_text(manual_authority, "target_sequence_hash", path=manual_mask_authority_source_path)
    if annotation_hash != manual_hash:
        raise ValueError(
            "RT annotation tracks target_sequence_hash does not match manual mask authority source: "
            f"{annotation_tracks_path} != {manual_mask_authority_source_path}"
        )

    manual_features = _manual_features_by_id(manual_authority, path=manual_mask_authority_source_path)
    features = tuple(
        _annotation_features(
            annotation_tracks,
            manual_features=manual_features,
            path=annotation_tracks_path,
        )
    )
    if not features:
        raise ValueError(f"RT annotation tracks must declare at least one display feature: {annotation_tracks_path}")
    return RTAnnotationContext(
        target_sequence_hash=annotation_hash,
        annotation_tracks_path=annotation_tracks_path,
        manual_mask_authority_source_path=manual_mask_authority_source_path,
        features=features,
    )


def _annotation_features(
    annotation_tracks: Mapping[str, Any],
    *,
    manual_features: Mapping[str, RTAnnotationFeature],
    path: Path,
) -> Iterable[RTAnnotationFeature]:
    tracks = _require_list(annotation_tracks, "tracks", path=path)
    for track in tracks:
        if not isinstance(track, Mapping):
            raise ValueError(f"RT annotation tracks entries must be mappings: {path}")
        track_id = _require_text(track, "id", path=path)
        for feature in _require_list(track, "features", path=path):
            if not isinstance(feature, Mapping):
                raise ValueError(f"RT annotation feature entries must be mappings: {path}")
            annotation_feature = RTAnnotationFeature(
                feature_id=_require_text(feature, "id", path=path),
                track_id=track_id,
                label=_require_text(feature, "label", path=path),
                start=_require_int(feature, "start", path=path),
                end=_require_int(feature, "end", path=path),
            )
            _validate_feature_span(annotation_feature, path=path)
            manual_feature = manual_features.get(annotation_feature.feature_id)
            if manual_feature is None:
                raise ValueError(
                    f"RT annotation feature {annotation_feature.feature_id} is missing from manual authority"
                )
            if (
                annotation_feature.start != manual_feature.start
                or annotation_feature.end != manual_feature.end
                or annotation_feature.label != manual_feature.label
            ):
                raise ValueError(
                    f"RT annotation feature {annotation_feature.feature_id} does not match manual authority"
                )
            yield annotation_feature


def _manual_features_by_id(manual_authority: Mapping[str, Any], *, path: Path) -> dict[str, RTAnnotationFeature]:
    features: dict[str, RTAnnotationFeature] = {}
    for authority_set in _require_list(manual_authority, "authority_sets", path=path):
        if not isinstance(authority_set, Mapping):
            raise ValueError(f"manual authority_sets entries must be mappings: {path}")
        authority_type = _require_text(authority_set, "authority_type", path=path)
        for feature in _require_list(authority_set, "features", path=path):
            if not isinstance(feature, Mapping):
                raise ValueError(f"manual authority feature entries must be mappings: {path}")
            _add_manual_feature(
                features,
                RTAnnotationFeature(
                    feature_id=_require_text(feature, "id", path=path),
                    track_id=authority_type,
                    label=_require_text(feature, "label", path=path),
                    start=_require_int(feature, "start", path=path),
                    end=_require_int(feature, "end", path=path),
                ),
                path=path,
            )
    for feature in manual_authority.get("context_only_spans", []):
        if not isinstance(feature, Mapping):
            raise ValueError(f"manual context_only_spans entries must be mappings: {path}")
        _add_manual_feature(
            features,
            RTAnnotationFeature(
                feature_id=_require_text(feature, "id", path=path),
                track_id="context_only_spans",
                label=_require_text(feature, "label", path=path),
                start=_require_int(feature, "start", path=path),
                end=_require_int(feature, "end", path=path),
            ),
            path=path,
        )
    return features


def _add_manual_feature(features: dict[str, RTAnnotationFeature], feature: RTAnnotationFeature, *, path: Path) -> None:
    _validate_feature_span(feature, path=path)
    existing = features.setdefault(feature.feature_id, feature)
    if existing != feature:
        raise ValueError(f"manual authority declares conflicting RT feature {feature.feature_id}: {path}")


def _validate_feature_span(feature: RTAnnotationFeature, *, path: Path) -> None:
    if feature.start > feature.end:
        raise ValueError(f"RT annotation feature {feature.feature_id} has start after end: {path}")
    if feature.start < 1:
        raise ValueError(f"RT annotation feature {feature.feature_id} must use 1-based positions: {path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _require_text(payload: Mapping[str, Any], field: str, *, path: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string in {path}")
    return value.strip()


def _require_int(payload: Mapping[str, Any], field: str, *, path: Path) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer in {path}")
    return value


def _require_list(payload: Mapping[str, Any], field: str, *, path: Path) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list in {path}")
    return value
