"""Feature-label placement helpers for generic MSA SVG renderers."""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.aligner.msa.visualization.contracts.models import AnnotationFeature

ALLOWED_LABEL_POSITIONS = frozenset({"auto", "inside", "above", "below", "hidden"})


@dataclass(frozen=True)
class FeatureLabelPlacement:
    """Resolved SVG label placement for an annotation feature."""

    visible: bool
    x: float
    y: float
    anchor: str
    color: str
    position: str


def validate_label_position(value: object, *, feature_id: str) -> str:
    """Return a valid label position or raise a contract error."""

    if value is None:
        return "auto"
    if not isinstance(value, str) or value not in ALLOWED_LABEL_POSITIONS:
        allowed = ", ".join(sorted(ALLOWED_LABEL_POSITIONS))
        raise ValueError(f"annotation feature {feature_id} label_position must be one of: {allowed}")
    return value


def resolve_label_placement(
    *,
    feature: AnnotationFeature,
    x: float,
    width: float,
    y: float,
    inside_y: float,
    above_y: float,
    below_y: float,
    min_inside_width: float,
) -> FeatureLabelPlacement:
    """Resolve display-only feature-label placement for SVG output."""

    requested = feature.label_position
    if requested == "hidden":
        return FeatureLabelPlacement(
            visible=False,
            x=x + width / 2,
            y=inside_y,
            anchor="middle",
            color=feature.text_color or feature.stroke_color,
            position=requested,
        )

    resolved = requested
    if requested == "auto":
        resolved = "inside" if width >= min_inside_width else "above"

    if resolved == "inside":
        return FeatureLabelPlacement(
            visible=True,
            x=x + width / 2,
            y=inside_y,
            anchor="middle",
            color=feature.text_color or "#ffffff",
            position=requested,
        )
    if resolved == "above":
        return FeatureLabelPlacement(
            visible=True,
            x=x + width / 2,
            y=above_y,
            anchor="middle",
            color=feature.text_color or feature.stroke_color,
            position=requested,
        )
    return FeatureLabelPlacement(
        visible=True,
        x=x + width / 2,
        y=below_y,
        anchor="middle",
        color=feature.text_color or feature.stroke_color,
        position=requested,
    )
