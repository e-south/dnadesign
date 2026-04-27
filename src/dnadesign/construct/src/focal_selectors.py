"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/focal_selectors.py

Annotation-aware focal point selection for Construct normalization jobs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from .annotations import AnnotationFeature, match_annotation_features
from .config import (
    AnnotationFeatureCenterSelectorConfig,
    AnnotationPairMidpointSelectorConfig,
    NormalizeAnchorSelectorConfig,
    SelectorChainConfig,
    SequenceMidpointSelectorConfig,
)


@dataclass(frozen=True)
class FocalSelection:
    focal_point_0: float
    focal_rule: str
    focal_features: tuple[str, ...]
    focal_confidence: str
    warnings: tuple[str, ...] = ()


def resolve_focal_selection(
    *,
    sequence_length: int,
    features: list[AnnotationFeature],
    selector_chain: SelectorChainConfig,
    allow_low_confidence: bool,
) -> FocalSelection:
    failures: list[str] = []
    for selector in selector_chain.selectors:
        try:
            selection = _resolve_single_selector(
                sequence_length=sequence_length,
                features=features,
                selector=selector,
            )
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if selection.focal_confidence == "low" and not allow_low_confidence:
            failures.append(
                f"{selection.focal_rule} resolved only a low-confidence focal point and fallback is disabled."
            )
            continue
        return selection
    joined = "; ".join(failures) if failures else "no selector resolved a focal point"
    raise ValueError(joined)


def _resolve_single_selector(
    *,
    sequence_length: int,
    features: list[AnnotationFeature],
    selector: NormalizeAnchorSelectorConfig,
) -> FocalSelection:
    if isinstance(selector, AnnotationPairMidpointSelectorConfig):
        first_matches = _require_unique_feature(
            features=features,
            role_hint=selector.first.role_hint,
            labels=selector.first.labels,
            selector_label="annotation_pair_midpoint.first",
        )
        second_matches = _require_unique_feature(
            features=features,
            role_hint=selector.second.role_hint,
            labels=selector.second.labels,
            selector_label="annotation_pair_midpoint.second",
        )
        first_feature = first_matches[0]
        second_feature = second_matches[0]
        return FocalSelection(
            focal_point_0=(first_feature.center_0 + second_feature.center_0) / 2.0,
            focal_rule="annotation_pair_midpoint",
            focal_features=(first_feature.feature_id, second_feature.feature_id),
            focal_confidence=selector.confidence,
        )

    if isinstance(selector, AnnotationFeatureCenterSelectorConfig):
        matched = _require_unique_feature(
            features=features,
            role_hint=selector.role_hint,
            labels=selector.labels,
            selector_label="annotation_feature_center",
        )
        feature = matched[0]
        return FocalSelection(
            focal_point_0=feature.center_0,
            focal_rule="annotation_feature_center",
            focal_features=(feature.feature_id,),
            focal_confidence=selector.confidence,
        )

    if isinstance(selector, SequenceMidpointSelectorConfig):
        if not selector.allowed:
            raise ValueError("sequence_midpoint fallback is configured but not allowed.")
        return FocalSelection(
            focal_point_0=float(sequence_length) / 2.0,
            focal_rule="sequence_midpoint",
            focal_features=(),
            focal_confidence=selector.confidence,
        )

    raise ValueError(f"Unsupported focal selector kind: {type(selector).__name__}")


def _require_unique_feature(
    *,
    features: list[AnnotationFeature],
    role_hint: str | None,
    labels: list[str],
    selector_label: str,
) -> list[AnnotationFeature]:
    matched = [feature for feature in match_annotation_features(features, role_hint=role_hint, labels=labels)]
    if not matched:
        raise ValueError(f"{selector_label} matched zero features.")
    if len(matched) > 1:
        raise ValueError(f"{selector_label} matched {len(matched)} features.")
    if not matched[0].has_precise_bounds:
        raise ValueError(f"{selector_label} matched feature {matched[0].feature_id!r} without precise bounds.")
    return matched
