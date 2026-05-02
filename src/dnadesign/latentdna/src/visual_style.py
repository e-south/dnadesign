"""
Shared visual style primitives for latentdna plots and notebooks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from textwrap import wrap

from .labels import humanize_label, sigma35_variant_display

TEXT_COLOR = "#16202A"
GRID_COLOR = "#D5DCE4"
SPINE_COLOR = "#5C6874"
ZERO_LINE_COLOR = "#94A3B8"
PANEL_BACKGROUND_COLOR = "#FCFDFE"

PLOT_FONT_FAMILY = "DejaVu Sans"
NOTEBOOK_FONT_STACK = '"Avenir Next", Avenir, "Segoe UI", "Helvetica Neue", sans-serif'
NOTEBOOK_MONO_STACK = '"SFMono-Regular", Menlo, Consolas, "DejaVu Sans Mono", monospace'

DEFAULT_PLOT_PNG_DPI = 400
DEFAULT_NOTEBOOK_FIG_DPI = 320
ANNOTATION_LABEL_BOX_ALPHA = 0.82

PLOT_SUPTITLE_FONT_SIZE = 15.5
PLOT_TITLE_FONT_SIZE = 13.75
PLOT_LABEL_FONT_SIZE = 12.5
PLOT_TICK_FONT_SIZE = 11.5
PLOT_LEGEND_FONT_SIZE = 11.5
PLOT_LEGEND_TITLE_SIZE = 11.5

PUBLICATION_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#111111",
]

_SEMANTIC_CATEGORY_COLORS = {
    "background": "#56B4E9",
    "background_only": "#56B4E9",
    "ethanol": "#E69F00",
    "ethanol_responsive": "#E69F00",
    "cipro": "#009E73",
    "cipro_responsive": "#009E73",
    "ciprofloxacin": "#009E73",
    "dual": "#CC79A7",
    "dual_and_responsive": "#CC79A7",
    "ethanol_ciprofloxacin": "#CC79A7",
    "control": "#111111",
    "selected": "#009E73",
    "baseline": "#0072B2",
    "challenger": "#E69F00",
    "orientation": "#7F8894",
    "context_anchor_mean": "#009E73",
    "whole_sequence_context": "#7F8894",
}

_SEMANTIC_CATEGORY_PRIORITY = {
    "control": 0,
    "background": 1,
    "background_only": 1,
    "ethanol": 2,
    "ethanol_responsive": 2,
    "cipro": 3,
    "cipro_responsive": 3,
    "ciprofloxacin": 3,
    "dual": 4,
    "dual_and_responsive": 4,
    "ethanol_ciprofloxacin": 4,
    "selected": 5,
    "baseline": 6,
    "challenger": 7,
    "orientation": 8,
    "context_anchor_mean": 9,
    "whole_sequence_context": 10,
}

_SIG35_VARIANT_STRENGTH_ORDER = ["f", "e", "d", "c", "b", "a"]
_SIG35_VARIANT_PRIORITY = {variant: index for index, variant in enumerate(_SIG35_VARIANT_STRENGTH_ORDER)}
_SIG35_LEGEND_CATEGORY_KEYS = frozenset({"f", "e", "d", "c", "b", "control"})
NONCANONICAL_SIG35_CATEGORY = "__latentdna_noncanonical_sig35__"
_SIG35_VARIANT_COLORS = {
    "f": "#B2182B",
    "e": "#D6604D",
    "d": "#F4A582",
    "c": "#92C5DE",
    "b": "#2166AC",
    "a": "#053061",
}
_SIG35_VARIANT_NEUTRAL_COLOR = "#7F8894"
_SPACER_LENGTH_COLOR_STOPS = ("#2C7BB6", "#ABD9E9", "#FEE090", "#F46D43", "#D73027")
_SINGLE_ROW_LEGEND_PLOT_IDS = frozenset(
    {
        "balanced_design_family_margin_gallery",
        "design_centroid_margin_gallery",
        "reference_alignment_summary",
        "representation_scree_diagnostic",
        "sigma35_stress_margin_gallery",
        "appendix_geometry_review",
        "appendix_umap_gallery",
    }
)
_LOWERED_LEGEND_PLOT_IDS = frozenset(
    {
        "balanced_design_family_margin_gallery",
        "design_centroid_margin_gallery",
        "reference_alignment_summary",
        "representation_scree_diagnostic",
        "sigma35_stress_margin_gallery",
        "appendix_geometry_review",
        "appendix_umap_gallery",
    }
)

_REGULATOR_COMPOSITION_DISPLAY = {
    "control": "Ctrl",
    "background": "Bg",
    "background_only": "Bg",
    "baer": "BaeR",
    "baer_only": "BaeR",
    "cpxr": "CpxR",
    "cpxr_only": "CpxR",
    "lexa": "LexA",
    "lexa_only": "LexA",
    "baer+lexa": "BaeR+LexA",
    "cpxr+lexa": "CpxR+LexA",
}


@dataclass(frozen=True)
class ScatterStyle:
    point_size: float
    alpha: float
    edgecolors: str
    linewidths: float
    rasterized: bool


@dataclass(frozen=True)
class LegendLayout:
    columns: int
    anchor_y: float
    bottom_margin: float


def normalize_category_key(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def is_sig35_legend_category(value: object) -> bool:
    return normalize_category_key(value) in _SIG35_LEGEND_CATEGORY_KEYS


def normalize_sig35_hue_category(value: object) -> str:
    normalized = normalize_category_key(value)
    if normalized in _SIG35_LEGEND_CATEGORY_KEYS:
        return normalized
    if normalized in {"", "na", "n/a", "none", "unknown"}:
        return NONCANONICAL_SIG35_CATEGORY
    return NONCANONICAL_SIG35_CATEGORY


def is_densegen_sig35_row(row: object) -> bool:
    if not isinstance(row, dict):
        return True
    if "source_class" in row:
        return normalize_category_key(row.get("source_class")) == "densegen"
    if "source_family" in row:
        return normalize_category_key(row.get("source_family")) == "densegen_generated"
    return True


def normalize_sig35_hue_category_for_row(row: object, value: object) -> str:
    if not is_densegen_sig35_row(row):
        return NONCANONICAL_SIG35_CATEGORY
    return normalize_sig35_hue_category(value)


def reference_annotation_label(value: object) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    text = re.sub(r"_context1kb_rc$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_core60$", "", text, flags=re.IGNORECASE)
    normalized = normalize_category_key(text)
    return {
        "spyp": "spyP",
        "sulap": "sulAp",
        "j23105": "J23105",
    }.get(normalized, text)


def _sig35_variant_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    normalized = normalize_category_key(text)
    if normalized == NONCANONICAL_SIG35_CATEGORY:
        return len(_SIG35_VARIANT_PRIORITY) + 2, text.casefold()
    if normalized in _SIG35_VARIANT_PRIORITY:
        return _SIG35_VARIANT_PRIORITY[normalized], text.casefold()
    if normalized == "control":
        return len(_SIG35_VARIANT_PRIORITY), text.casefold()
    if normalized in {"unknown", "na", "n/a"}:
        return len(_SIG35_VARIANT_PRIORITY) + 1, text.casefold()
    return len(_SIG35_VARIANT_PRIORITY) + 2, text.casefold()


def _numeric_category_value(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_integer_like_value(value: object) -> str | None:
    numeric = _numeric_category_value(value)
    if numeric is None:
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _spacer_length_sort_key(value: object) -> tuple[int, float, str]:
    text = str(value)
    numeric = _numeric_category_value(value)
    if numeric is not None:
        return 0, numeric, text.casefold()
    normalized = normalize_category_key(text)
    if normalized in {"na", "n/a", "unknown"}:
        return 2, 0.0, text.casefold()
    return 1, 0.0, text.casefold()


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    normalized = color.lstrip("#")
    return int(normalized[0:2], 16), int(normalized[2:4], 16), int(normalized[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _interpolate_hex_color(left: str, right: str, fraction: float) -> str:
    left_rgb = _hex_to_rgb(left)
    right_rgb = _hex_to_rgb(right)
    return _rgb_to_hex(
        tuple(int(round(start + ((stop - start) * fraction))) for start, stop in zip(left_rgb, right_rgb, strict=True))
    )


def _color_ramp(color_stops: tuple[str, ...], count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return [color_stops[len(color_stops) // 2]]
    if count == len(color_stops):
        return list(color_stops)
    segments = len(color_stops) - 1
    ramp: list[str] = []
    for index in range(count):
        position = index / max(count - 1, 1)
        segment_index = min(int(position * segments), segments - 1)
        left_position = segment_index / segments
        right_position = (segment_index + 1) / segments
        local_fraction = (
            0.0 if right_position == left_position else (position - left_position) / (right_position - left_position)
        )
        ramp.append(
            _interpolate_hex_color(
                color_stops[segment_index],
                color_stops[segment_index + 1],
                local_fraction,
            )
        )
    return ramp


def ordered_categories(values: Iterable[str], *, column: str | None = None) -> list[str]:
    unique = sorted({str(value) for value in values})
    if str(column or "").strip() == "sig35_variant":
        return sorted(unique, key=_sig35_variant_sort_key)
    if str(column or "").strip() == "spacer_length":
        return sorted(unique, key=_spacer_length_sort_key)
    return sorted(
        unique,
        key=lambda value: (_SEMANTIC_CATEGORY_PRIORITY.get(normalize_category_key(value), 99), value.casefold()),
    )


def categorical_color_map(categories: Iterable[str], *, column: str | None = None) -> dict[str, str]:
    ordered = ordered_categories(categories, column=column)
    if str(column or "").strip() == "spacer_length":
        numeric_categories = [category for category in ordered if _numeric_category_value(category) is not None]
        non_numeric_categories = [category for category in ordered if _numeric_category_value(category) is None]
        color_map = {
            category: color
            for category, color in zip(
                numeric_categories,
                _color_ramp(_SPACER_LENGTH_COLOR_STOPS, len(numeric_categories)),
                strict=True,
            )
        }
        for category in non_numeric_categories:
            color_map[category] = _SIG35_VARIANT_NEUTRAL_COLOR
        return color_map
    color_map: dict[str, str] = {}
    fallback_index = 0
    for category in ordered:
        normalized_category = normalize_category_key(category)
        if str(column or "").strip() == "sig35_variant":
            if normalized_category in _SIG35_VARIANT_COLORS:
                color_map[category] = _SIG35_VARIANT_COLORS[normalized_category]
                continue
            if normalized_category in {"control", "unknown", "na", "n/a"}:
                color_map[category] = _SIG35_VARIANT_NEUTRAL_COLOR
                continue
        semantic_color = _SEMANTIC_CATEGORY_COLORS.get(normalize_category_key(category))
        if semantic_color is not None:
            color_map[category] = semantic_color
            continue
        color_map[category] = PUBLICATION_PALETTE[fallback_index % len(PUBLICATION_PALETTE)]
        fallback_index += 1
    return color_map


def scatter_style(row_count: int) -> ScatterStyle:
    if row_count <= 250:
        return ScatterStyle(point_size=30.0, alpha=0.84, edgecolors="white", linewidths=0.32, rasterized=False)
    if row_count <= 1_000:
        return ScatterStyle(point_size=16.0, alpha=0.66, edgecolors="white", linewidths=0.16, rasterized=False)
    if row_count <= 5_000:
        return ScatterStyle(point_size=6.6, alpha=0.34, edgecolors="none", linewidths=0.0, rasterized=True)
    if row_count <= 20_000:
        return ScatterStyle(point_size=3.4, alpha=0.22, edgecolors="none", linewidths=0.0, rasterized=True)
    return ScatterStyle(point_size=1.7, alpha=0.15, edgecolors="none", linewidths=0.0, rasterized=True)


def humanize_display_text(value: object) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if re.fullmatch(r"[A-Z][A-Z0-9]*(?:\+[A-Z0-9]+)*", text):
        return text.replace("+", " + ")
    normalized = text
    if normalized.startswith("log_likelihood_per_token_"):
        normalized = normalized.replace("log_likelihood_per_token_", "log likelihood per token ")
    normalized = re.sub(r"\becdf\b", "ECDF", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip(" -")
    return humanize_label(normalized)


def _canonical_regulator_token(value: object) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("sig35="):
        return "background"
    if lowered in {"background", "background_only", "control"}:
        return lowered
    token = text.split("_", 1)[0].strip()
    if not token:
        return None
    lowered_token = token.lower()
    return {
        "baer": "baeR",
        "cpxr": "cpxR",
        "lexa": "lexA",
        "background": "background",
        "background_only": "background",
        "control": "control",
    }.get(lowered_token, token)


def compact_design_regulator_composition(value: object) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    direct = _REGULATOR_COMPOSITION_DISPLAY.get(text.lower())
    if direct is not None:
        return direct

    parts: list[str] = []
    for raw_part in text.split("+"):
        token = _canonical_regulator_token(raw_part)
        if token is None or token in parts:
            continue
        parts.append(token)
    if not parts:
        return humanize_display_text(text)

    direct = _REGULATOR_COMPOSITION_DISPLAY.get("+".join(part.lower() for part in parts))
    if direct is not None:
        return direct

    return "+".join(_REGULATOR_COMPOSITION_DISPLAY.get(part.lower(), humanize_label(part)) for part in parts)


def display_category_text(value: object, *, column: str | None = None) -> str:
    normalized_column = str(column or "").strip()
    if normalized_column == "spacer_length":
        formatted = _format_integer_like_value(value)
        return formatted or humanize_display_text(value)
    if normalized_column == "design_regulator_composition":
        return compact_design_regulator_composition(value)
    if normalized_column == "sig35_variant":
        text = str(value or "").strip()
        normalized = normalize_category_key(text)
        if normalized == NONCANONICAL_SIG35_CATEGORY:
            return "Reference/other"
        if normalized in {"", "na", "n/a"}:
            return "NA"
        if normalized in {"control", "unknown"}:
            return humanize_display_text(text)
        resolved = sigma35_variant_display(text)
        if resolved is not None:
            return resolved
        return humanize_label(f"variant {text}")
    return humanize_display_text(value)


def legend_column_count(labels: Iterable[str], *, max_columns: int = 3) -> int:
    normalized = [str(label).strip() for label in labels if str(label).strip()]
    if not normalized:
        return 1
    if len(normalized) <= 3:
        return len(normalized)
    longest = max(len(label) for label in normalized)
    if longest >= 16:
        return min(2, len(normalized))
    return min(max_columns, len(normalized))


def legend_bottom_margin(
    label_count: int,
    *,
    columns: int,
    base_margin: float = 0.11,
    row_step: float = 0.043,
) -> float:
    if label_count <= 0:
        return 0.0
    rows = max(1, ceil(label_count / max(columns, 1)))
    return min(base_margin + (row_step * max(rows - 1, 0)), 0.36)


def legend_layout(
    labels: Iterable[str],
    *,
    plot_id: str | None,
    default_anchor_y: float,
    default_base_margin: float = 0.11,
    row_step: float = 0.043,
    max_columns: int = 3,
    single_row: bool | None = None,
) -> LegendLayout:
    normalized = [str(label).strip() for label in labels if str(label).strip()]
    if not normalized:
        return LegendLayout(columns=1, anchor_y=default_anchor_y, bottom_margin=0.0)

    plot_key = str(plot_id or "").strip()
    use_single_row = plot_key in _SINGLE_ROW_LEGEND_PLOT_IDS if single_row is None else single_row
    columns = max(1, len(normalized)) if use_single_row else legend_column_count(normalized, max_columns=max_columns)
    anchor_y = default_anchor_y
    base_margin = default_base_margin
    if plot_key in _LOWERED_LEGEND_PLOT_IDS:
        anchor_y = max(default_anchor_y - 0.008, 0.004)
        base_margin += 0.015
    bottom_margin = legend_bottom_margin(
        len(normalized),
        columns=columns,
        base_margin=base_margin,
        row_step=row_step,
    )
    return LegendLayout(columns=columns, anchor_y=anchor_y, bottom_margin=bottom_margin)


def compact_candidate_title(value: object) -> str:
    text = humanize_display_text(value)
    lowered = text.casefold()
    if not any(
        token in lowered
        for token in (
            "evo 2",
            "anchor",
            "context",
            "pooled logits",
            "intermediate block",
            "intermediate embedding",
            "concat",
        )
    ):
        return text

    compact = text
    replacements = [
        (r"\bEvo\s*2\b", ""),
        (r"\b60\s*Bp Anchor\b", "60 bp"),
        (r"\b1\s*Kb Construct Context\b", "1 kb ctx"),
        (r"\b1\s*Kb Context\b", "1 kb ctx"),
        (r"\b1\s*Kb Context Anchor Mean\b", "1 kb anchor mean"),
        (r"\bAnchor \+ Anchor-Mean Concat\b", "anchor + anchor-mean"),
        (r"\bAnchor \+ 1\s*Kb Context Concat\b", "anchor + 1 kb ctx"),
        (r"\bIntermediate Block Mean\b", "Block"),
        (r"\bIntermediate Embedding\b", "Block"),
        (r"\bPooled Logits\b", "Logits"),
    ]
    for pattern, replacement in replacements:
        compact = re.sub(pattern, replacement, compact, flags=re.IGNORECASE)
    compact = re.sub(r"\s*·\s*", " · ", compact)
    compact = re.sub(r"\s+", " ", compact).strip(" ·-")
    return compact


def wrap_plot_title(title: object, *, width: int = 28, max_lines: int | None = None) -> str:
    raw_text = str(title or "")
    if not raw_text.strip():
        return ""
    wrapped_lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = humanize_display_text(raw_line)
        if not line:
            continue
        wrapped_lines.extend(
            wrap(
                line,
                width=max(width, 10),
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    lines = wrapped_lines or [humanize_display_text(raw_text)]
    if max_lines is not None and max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        if not lines[-1].endswith("..."):
            lines[-1] = lines[-1].rstrip(". ") + "..."
    return "\n".join(lines)
