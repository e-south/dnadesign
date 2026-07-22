"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/primitive_visual_roles.py

Primitive visual role contract for retron MSD composition and structure figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PrimitiveVisualRole:
    role_id: str
    display_label: str
    palette_token: str
    stroke_color: str
    fill_color: str
    priority: int
    applies_to: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["applies_to"] = list(self.applies_to)
        return payload


PRIMITIVE_VISUAL_ROLES: tuple[PrimitiveVisualRole, ...] = (
    PrimitiveVisualRole(
        role_id="flank_5p",
        display_label="5' flank",
        palette_token="flank.5p",
        stroke_color="#4B5563",
        fill_color="#D1D5DB",
        priority=10,
        applies_to=("backbone", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="flank_3p",
        display_label="3' flank",
        palette_token="flank.3p",
        stroke_color="#111827",
        fill_color="#E5E7EB",
        priority=10,
        applies_to=("backbone", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="stem_base_left",
        display_label="Left stem base",
        palette_token="stem_base.left",
        stroke_color="#0072B2",
        fill_color="#BFDBFE",
        priority=70,
        applies_to=("backbone", "basepair", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="stem_base_right",
        display_label="Right stem base",
        palette_token="stem_base.right",
        stroke_color="#882255",
        fill_color="#F5C2E7",
        priority=70,
        applies_to=("backbone", "basepair", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="stem_extension_left",
        display_label="Left stem extension",
        palette_token="stem_extension.left",
        stroke_color="#332288",
        fill_color="#D8D5F2",
        priority=50,
        applies_to=("backbone", "basepair", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="stem_extension_right",
        display_label="Right stem extension",
        palette_token="stem_extension.right",
        stroke_color="#4477AA",
        fill_color="#C7DAEF",
        priority=50,
        applies_to=("backbone", "basepair", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="payload_primary",
        display_label="Payload primary",
        palette_token="payload.primary",
        stroke_color="#D55E00",
        fill_color="#F8C491",
        priority=40,
        applies_to=("backbone", "basepair", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="payload_complement",
        display_label="Payload complement",
        palette_token="payload.complement",
        stroke_color="#AA4499",
        fill_color="#E9B9E3",
        priority=40,
        applies_to=("backbone", "basepair", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="snapback_foldback_geometry",
        display_label="Foldback geometry",
        palette_token="foldback.geometry",
        stroke_color="#009E73",
        fill_color="#BFE9DE",
        priority=35,
        applies_to=("backbone", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="snapback_retained_stem",
        display_label="Foldback stem",
        palette_token="foldback.stem",
        stroke_color="#117733",
        fill_color="#BFE3C8",
        priority=60,
        applies_to=("backbone", "basepair", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="snapback_cap",
        display_label="Cap",
        palette_token="cap.loop",
        stroke_color="#CC79A7",
        fill_color="#EBC7DC",
        priority=80,
        applies_to=("backbone", "nucleotide_text", "section_label"),
    ),
    PrimitiveVisualRole(
        role_id="snapback_foldback_return",
        display_label="Foldback return",
        palette_token="foldback.return",
        stroke_color="#44AA99",
        fill_color="#C9ECE7",
        priority=60,
        applies_to=("backbone", "basepair", "nucleotide_text", "section_label"),
    ),
)


MIN_STROKE_CONTRAST_RATIO = 2.0


def primitive_visual_roles_payload() -> dict[str, dict[str, object]]:
    validate_primitive_visual_roles()
    return {role.role_id: role.to_dict() for role in PRIMITIVE_VISUAL_ROLES}


def primitive_component_hues() -> dict[str, str]:
    validate_primitive_visual_roles()
    return {role.role_id: role.stroke_color for role in PRIMITIVE_VISUAL_ROLES}


def primitive_component_styles() -> dict[str, dict[str, object]]:
    validate_primitive_visual_roles()
    return {
        role.role_id: {"fill": role.fill_color, "alpha": 0.72, "edge_color": role.stroke_color}
        for role in PRIMITIVE_VISUAL_ROLES
    }


def validate_primitive_visual_roles(roles: tuple[PrimitiveVisualRole, ...] = PRIMITIVE_VISUAL_ROLES) -> None:
    role_ids = [role.role_id for role in roles]
    if len(set(role_ids)) != len(role_ids):
        raise ValueError("Primitive visual role IDs must be unique.")
    stroke_colors = [role.stroke_color.upper() for role in roles]
    if len(set(stroke_colors)) != len(stroke_colors):
        raise ValueError("Primitive visual role stroke colors must be unique.")
    low_contrast = [
        role.role_id for role in roles if _contrast_ratio(role.stroke_color, "#FFFFFF") < MIN_STROKE_CONTRAST_RATIO
    ]
    if low_contrast:
        raise ValueError(
            "Primitive visual role stroke colors must have contrast ratio "
            f">= {MIN_STROKE_CONTRAST_RATIO} against white: {', '.join(low_contrast)}"
        )


def _contrast_ratio(color_a: str, color_b: str) -> float:
    lum_a = _relative_luminance(color_a)
    lum_b = _relative_luminance(color_b)
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def _relative_luminance(hex_color: str) -> float:
    text = hex_color.strip().removeprefix("#")
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {hex_color!r}.")
    channels = []
    for offset in (0, 2, 4):
        channel = int(text[offset : offset + 2], 16) / 255.0
        channels.append(channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


__all__ = [
    "MIN_STROKE_CONTRAST_RATIO",
    "PRIMITIVE_VISUAL_ROLES",
    "PrimitiveVisualRole",
    "primitive_component_hues",
    "primitive_component_styles",
    "primitive_visual_roles_payload",
    "validate_primitive_visual_roles",
]
