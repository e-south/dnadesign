"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/config/style_v1.py

Style v1 schema, preset resolution, and strict style mapping validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..core import SchemaError, ensure


@dataclass(frozen=True)
class LayoutStyle:
    outer_pad_cells: float = 0.50


@dataclass(frozen=True)
class SequenceStyle:
    strand_gap_cells: float = 0.75
    to_kmer_gap_cells: float = 0.45
    bold_consensus_bases: bool = False
    non_consensus_color: str = "#9ca3af"
    tone_quantile_low: float = 0.10
    tone_quantile_high: float = 0.90


@dataclass(frozen=True)
class GlyphStyle:
    round_px: float = 4.0
    edge_width: float = 0.0
    fill_alpha: float = 0.92
    box_height_cells: float = 1.15
    text_color: str = "#ffffff"
    pad_x_px: float = 1.0
    text_y_nudge_cells: float = 0.0
    to_logo_gap_cells: float = 0.30


@dataclass(frozen=True)
class MotifScaleBarStyle:
    enabled: bool = False
    location: str = "top_right"
    font_size: int = 8
    color: str = "#6b7280"
    pad_cells: float = 0.20


@dataclass(frozen=True)
class MotifLetterColoringStyle:
    mode: str = "classic"
    other_color: str = "#d1d5db"
    observed_color_source: str = "nucleotide_palette"


@dataclass(frozen=True)
class MotifLogoStyle:
    height_bits: float = 2.0
    bits_to_cells: float = 0.90
    y_pad_cells: float = 0.35
    letter_x_pad_frac: float = 0.10
    alpha_other: float = 0.65
    alpha_observed: float = 1.0
    colors: Mapping[str, str] = field(
        default_factory=lambda: {
            "A": "#1f77b4",
            "C": "#2ca02c",
            "G": "#ff7f0e",
            "T": "#d62728",
        }
    )
    layout: str = "stack"
    lane_mode: str = "follow_feature_track"
    display_mode: str = "information"
    debug_bounds: bool = False
    scale_bar: MotifScaleBarStyle = field(default_factory=MotifScaleBarStyle)
    letter_coloring: MotifLetterColoringStyle = field(default_factory=MotifLetterColoringStyle)


@dataclass(frozen=True)
class Style:
    dpi: int = 180
    figure_scale: float = 1.0
    font_mono: str = "DejaVu Sans Mono"
    font_label: str = "DejaVu Sans"
    font_size_seq: int = 14
    font_size_label: int = 12

    padding_x: float = 24.0
    padding_y: float = 20.0
    track_spacing: float = 22.0
    baseline_spacing: float = 56.0
    show_reverse_complement: bool = True
    layout: LayoutStyle = field(default_factory=LayoutStyle)
    sequence: SequenceStyle = field(default_factory=SequenceStyle)

    color_sequence: str = "#4b5563"
    color_ticks: str = "#9ca3af"
    overlay_align: str = "left"

    legend: bool = True
    legend_mode: str = "bottom"
    legend_inline_side: str = "auto"
    legend_inline_margin_cells: float = 0.35
    legend_font_size: int = 11
    legend_patch_w: float = 18.0
    legend_patch_h: float = 12.0
    legend_gap_x: float = 14.0
    legend_pad_px: float = 16.0
    legend_height_px: float = 40.0
    legend_gap_patch_text: float = 6.0
    legend_center: bool = True

    connectors: bool = True
    connector_alpha: float = 0.45
    connector_width: float = 0.6
    connector_dash: tuple[float, ...] = (1.0, 3.0)

    palette: Mapping[str, str] = field(default_factory=dict)
    span_link_inner_margin_bp: float = 0.25

    kmer: GlyphStyle = field(default_factory=GlyphStyle)
    motif_logo: MotifLogoStyle = field(default_factory=MotifLogoStyle)

    def __post_init__(self) -> None:
        if isinstance(self.layout, dict):
            object.__setattr__(self, "layout", LayoutStyle(**self.layout))
        if isinstance(self.sequence, dict):
            object.__setattr__(self, "sequence", SequenceStyle(**self.sequence))
        if isinstance(self.kmer, dict):
            object.__setattr__(self, "kmer", GlyphStyle(**self.kmer))
        motif_logo = self.motif_logo
        if isinstance(motif_logo, dict):
            motif_logo = MotifLogoStyle(**motif_logo)
        if isinstance(motif_logo.letter_coloring, dict):
            motif_logo = MotifLogoStyle(
                height_bits=motif_logo.height_bits,
                bits_to_cells=motif_logo.bits_to_cells,
                y_pad_cells=motif_logo.y_pad_cells,
                letter_x_pad_frac=motif_logo.letter_x_pad_frac,
                alpha_other=motif_logo.alpha_other,
                alpha_observed=motif_logo.alpha_observed,
                colors=motif_logo.colors,
                layout=motif_logo.layout,
                lane_mode=motif_logo.lane_mode,
                display_mode=motif_logo.display_mode,
                debug_bounds=motif_logo.debug_bounds,
                scale_bar=motif_logo.scale_bar,
                letter_coloring=MotifLetterColoringStyle(**motif_logo.letter_coloring),
            )
        if isinstance(motif_logo.scale_bar, dict):
            motif_logo = MotifLogoStyle(
                height_bits=motif_logo.height_bits,
                bits_to_cells=motif_logo.bits_to_cells,
                y_pad_cells=motif_logo.y_pad_cells,
                letter_x_pad_frac=motif_logo.letter_x_pad_frac,
                alpha_other=motif_logo.alpha_other,
                alpha_observed=motif_logo.alpha_observed,
                colors=motif_logo.colors,
                layout=motif_logo.layout,
                lane_mode=motif_logo.lane_mode,
                display_mode=motif_logo.display_mode,
                debug_bounds=motif_logo.debug_bounds,
                scale_bar=MotifScaleBarStyle(**motif_logo.scale_bar),
                letter_coloring=motif_logo.letter_coloring,
            )
        object.__setattr__(self, "motif_logo", motif_logo)

        ensure(self.dpi >= 72, "style.dpi must be >= 72", SchemaError)
        ensure(self.figure_scale > 0, "style.figure_scale must be > 0", SchemaError)
        ensure(self.font_size_seq >= 6, "style.font_size_seq must be >= 6", SchemaError)
        ensure(self.font_size_label >= 6, "style.font_size_label must be >= 6", SchemaError)
        ensure(self.padding_x >= 0, "style.padding_x must be >= 0", SchemaError)
        ensure(self.padding_y >= 0, "style.padding_y must be >= 0", SchemaError)
        ensure(self.layout.outer_pad_cells >= 0, "style.layout.outer_pad_cells must be >= 0", SchemaError)
        ensure(self.track_spacing > 0, "style.track_spacing must be > 0", SchemaError)
        ensure(self.baseline_spacing > 0, "style.baseline_spacing must be > 0", SchemaError)
        ensure(
            str(self.overlay_align).lower() in {"left", "center", "right"},
            "style.overlay_align must be 'left', 'center', or 'right'",
            SchemaError,
        )
        ensure(
            str(self.legend_mode).lower() in {"bottom", "inline", "none"},
            "style.legend_mode must be 'bottom', 'inline', or 'none'",
            SchemaError,
        )
        ensure(
            str(self.legend_inline_side).lower() in {"auto", "left", "right"},
            "style.legend_inline_side must be 'auto', 'left', or 'right'",
            SchemaError,
        )
        ensure(
            self.legend_inline_margin_cells >= 0,
            "style.legend_inline_margin_cells must be >= 0",
            SchemaError,
        )
        ensure(self.sequence.strand_gap_cells >= 0, "style.sequence.strand_gap_cells must be >= 0", SchemaError)
        ensure(self.sequence.to_kmer_gap_cells >= 0, "style.sequence.to_kmer_gap_cells must be >= 0", SchemaError)
        dash_raw = self.connector_dash
        if dash_raw is None:
            dash_pattern: tuple[float, ...] = ()
        else:
            ensure(
                isinstance(dash_raw, (tuple, list)),
                "style.connector_dash must be a sequence of numbers",
                SchemaError,
            )
            dash_pattern = tuple(float(value) for value in dash_raw)
        ensure(
            len(dash_pattern) in {0, 2},
            "style.connector_dash must contain exactly 2 values or be empty",
            SchemaError,
        )
        ensure(
            all(value > 0 for value in dash_pattern),
            "style.connector_dash values must be > 0",
            SchemaError,
        )
        object.__setattr__(self, "connector_dash", dash_pattern)
        ensure(
            str(self.sequence.non_consensus_color).strip() != "",
            "style.sequence.non_consensus_color must be non-empty",
            SchemaError,
        )
        ensure(
            0.0 <= self.sequence.tone_quantile_low < 1.0,
            "style.sequence.tone_quantile_low must be in [0, 1)",
            SchemaError,
        )
        ensure(
            0.0 < self.sequence.tone_quantile_high <= 1.0,
            "style.sequence.tone_quantile_high must be in (0, 1]",
            SchemaError,
        )
        ensure(
            self.sequence.tone_quantile_low < self.sequence.tone_quantile_high,
            "style.sequence.tone_quantile_low must be < style.sequence.tone_quantile_high",
            SchemaError,
        )
        ensure(self.kmer.box_height_cells > 0, "style.kmer.box_height_cells must be > 0", SchemaError)
        ensure(self.kmer.pad_x_px >= 0, "style.kmer.pad_x_px must be >= 0", SchemaError)
        ensure(self.kmer.to_logo_gap_cells >= 0, "style.kmer.to_logo_gap_cells must be >= 0", SchemaError)
        ensure(self.span_link_inner_margin_bp >= 0, "style.span_link_inner_margin_bp must be >= 0", SchemaError)
        ensure(self.motif_logo.height_bits > 0, "style.motif_logo.height_bits must be > 0", SchemaError)
        ensure(self.motif_logo.bits_to_cells > 0, "style.motif_logo.bits_to_cells must be > 0", SchemaError)
        ensure(self.motif_logo.y_pad_cells >= 0, "style.motif_logo.y_pad_cells must be >= 0", SchemaError)
        ensure(
            0.0 <= self.motif_logo.letter_x_pad_frac < 1.0,
            "style.motif_logo.letter_x_pad_frac must be in [0, 1)",
            SchemaError,
        )
        ensure(
            0.0 <= self.motif_logo.alpha_other <= 1.0,
            "style.motif_logo.alpha_other must be in [0, 1]",
            SchemaError,
        )
        ensure(
            0.0 <= self.motif_logo.alpha_observed <= 1.0,
            "style.motif_logo.alpha_observed must be in [0, 1]",
            SchemaError,
        )
        ensure(
            str(self.motif_logo.layout).lower() in {"stack", "overlay"},
            "style.motif_logo.layout must be 'stack' or 'overlay'",
            SchemaError,
        )
        ensure(
            str(self.motif_logo.lane_mode).lower() in {"follow_feature_track", "independent"},
            "style.motif_logo.lane_mode must be 'follow_feature_track' or 'independent'",
            SchemaError,
        )
        ensure(
            str(self.motif_logo.display_mode).lower() in {"information", "probability"},
            "style.motif_logo.display_mode must be 'information' or 'probability'",
            SchemaError,
        )
        ensure(isinstance(self.motif_logo.colors, Mapping), "style.motif_logo.colors must be a mapping", SchemaError)
        bad_keys = sorted(set(self.motif_logo.colors.keys()) - {"A", "C", "G", "T"})
        ensure(
            not bad_keys,
            f"style.motif_logo.colors unknown base key(s): {bad_keys}",
            SchemaError,
        )
        missing = sorted({"A", "C", "G", "T"} - set(self.motif_logo.colors.keys()))
        ensure(
            not missing,
            f"style.motif_logo.colors missing base key(s): {missing}",
            SchemaError,
        )
        ensure(
            self.motif_logo.scale_bar.font_size >= 6,
            "style.motif_logo.scale_bar.font_size must be >= 6",
            SchemaError,
        )
        ensure(
            str(self.motif_logo.scale_bar.location).lower() in {"top_right", "bottom_right", "left_of_logo"},
            "style.motif_logo.scale_bar.location must be 'top_right', 'bottom_right', or 'left_of_logo'",
            SchemaError,
        )
        ensure(
            self.motif_logo.scale_bar.pad_cells >= 0,
            "style.motif_logo.scale_bar.pad_cells must be >= 0",
            SchemaError,
        )
        ensure(
            str(self.motif_logo.letter_coloring.mode).lower() in {"classic", "match_window_seq"},
            "style.motif_logo.letter_coloring.mode must be 'classic' or 'match_window_seq'",
            SchemaError,
        )
        ensure(
            str(self.motif_logo.letter_coloring.observed_color_source).lower()
            in {"nucleotide_palette", "feature_fill"},
            "style.motif_logo.letter_coloring.observed_color_source must be 'nucleotide_palette' or 'feature_fill'",
            SchemaError,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "Style":
        if not isinstance(mapping, Mapping):
            raise SchemaError("style must be a mapping/dict")
        data = dict(mapping)
        if "height_factor" in data:
            raise SchemaError("Unknown style key 'height_factor' — use 'style.kmer.box_height_cells'.")
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(data.keys()) - allowed)
        if unknown:
            raise SchemaError(f"Unknown style key(s): {unknown}")
        layout_raw = data.get("layout")
        if layout_raw is not None and not isinstance(layout_raw, Mapping):
            raise SchemaError("style.layout must be a mapping")
        if isinstance(layout_raw, Mapping):
            allowed_layout = {f.name for f in fields(LayoutStyle)}
            unknown_layout = sorted(set(layout_raw.keys()) - allowed_layout)
            if unknown_layout:
                raise SchemaError(f"Unknown style.layout key(s): {unknown_layout}")
        seq_raw = data.get("sequence")
        if seq_raw is not None and not isinstance(seq_raw, Mapping):
            raise SchemaError("style.sequence must be a mapping")
        if isinstance(seq_raw, Mapping):
            allowed_sequence = {f.name for f in fields(SequenceStyle)}
            unknown_sequence = sorted(set(seq_raw.keys()) - allowed_sequence)
            if unknown_sequence:
                raise SchemaError(f"Unknown style.sequence key(s): {unknown_sequence}")
        kmer_raw = data.get("kmer")
        if kmer_raw is not None and not isinstance(kmer_raw, Mapping):
            raise SchemaError("style.kmer must be a mapping")
        if isinstance(kmer_raw, Mapping):
            allowed_kmer = {f.name for f in fields(GlyphStyle)}
            unknown_kmer = sorted(set(kmer_raw.keys()) - allowed_kmer)
            if unknown_kmer:
                raise SchemaError(f"Unknown style.kmer key(s): {unknown_kmer}")
        motif_raw = data.get("motif_logo")
        if motif_raw is not None and not isinstance(motif_raw, Mapping):
            raise SchemaError("style.motif_logo must be a mapping")
        if isinstance(motif_raw, Mapping):
            allowed_motif = {f.name for f in fields(MotifLogoStyle)}
            unknown_motif = sorted(set(motif_raw.keys()) - allowed_motif)
            if unknown_motif:
                raise SchemaError(f"Unknown style.motif_logo key(s): {unknown_motif}")
            motif_colors = motif_raw.get("colors")
            if motif_colors is not None:
                if not isinstance(motif_colors, Mapping):
                    raise SchemaError("style.motif_logo.colors must be a mapping")
                allowed_bases = {"A", "C", "G", "T"}
                unknown_bases = sorted(set(motif_colors.keys()) - allowed_bases)
                if unknown_bases:
                    raise SchemaError(f"Unknown style.motif_logo.colors key(s): {unknown_bases}")
            motif_scale_bar = motif_raw.get("scale_bar")
            if motif_scale_bar is not None:
                if not isinstance(motif_scale_bar, Mapping):
                    raise SchemaError("style.motif_logo.scale_bar must be a mapping")
                allowed_scale_bar = {f.name for f in fields(MotifScaleBarStyle)}
                unknown_scale_bar = sorted(set(motif_scale_bar.keys()) - allowed_scale_bar)
                if unknown_scale_bar:
                    raise SchemaError(f"Unknown style.motif_logo.scale_bar key(s): {unknown_scale_bar}")
            motif_letter_coloring = motif_raw.get("letter_coloring")
            if motif_letter_coloring is not None:
                if not isinstance(motif_letter_coloring, Mapping):
                    raise SchemaError("style.motif_logo.letter_coloring must be a mapping")
                allowed_coloring = {f.name for f in fields(MotifLetterColoringStyle)}
                unknown_coloring = sorted(set(motif_letter_coloring.keys()) - allowed_coloring)
                if unknown_coloring:
                    raise SchemaError(f"Unknown style.motif_logo.letter_coloring key(s): {unknown_coloring}")
        return cls(**data)


def _baserender_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _styles_v1_root() -> Path:
    return _baserender_root() / "styles" / "style_v1"


@lru_cache(maxsize=64)
def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text())
    except Exception as exc:
        raise SchemaError(f"Could not parse style YAML: {path}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise SchemaError(f"style YAML must be a mapping/dict: {path}")
    return dict(raw)


def list_style_presets() -> list[str]:
    styles_dir = _styles_v1_root()
    if not styles_dir.exists():
        return []
    names: set[str] = set()
    names.update(p.stem for p in styles_dir.glob("*.yaml"))
    return sorted(names)


def resolve_preset_path(spec: str | Path | None) -> Path | None:
    if spec is None:
        return None
    raw = Path(spec)
    root = _baserender_root()

    if raw.is_absolute():
        if not raw.exists():
            raise SchemaError(f"Style preset not found: {raw}")
        return raw

    if len(raw.parts) > 1 or raw.suffix.lower() == ".yaml":
        direct = root / raw
        if direct.exists():
            return direct
        styles_guess = _styles_v1_root() / raw.name
        if styles_guess.exists():
            return styles_guess
        raise SchemaError(f"Style preset not found: {spec}")

    for suffix in (".yaml",):
        candidate = _styles_v1_root() / f"{raw}{suffix}"
        if candidate.exists():
            return candidate
    raise SchemaError(f"Style preset not found: {spec}")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(val, Mapping):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def resolve_style(*, preset: str | Path | None, overrides: Mapping[str, object] | None = None) -> Style:
    base_path = resolve_preset_path("presentation_default")
    assert base_path is not None
    merged: dict[str, Any] = _load_yaml_mapping(base_path)

    if preset is not None:
        p = resolve_preset_path(preset)
        assert p is not None
        merged = _deep_merge(merged, _load_yaml_mapping(p))

    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise SchemaError("render.style.overrides must be a mapping")
        merged = _deep_merge(merged, dict(overrides))

    return Style.from_mapping(merged)
