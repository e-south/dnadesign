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
class GlyphStyle:
    round_px: float = 4.0
    edge_width: float = 0.0
    fill_alpha: float = 0.92
    height_factor: float = 1.15
    text_color: str = "#ffffff"
    pad_x_px: float = 1.0
    text_v_align: str = "baseline"
    text_v_offset_px: float = 0.0


@dataclass(frozen=True)
class MotifLogoStyle:
    height_bits: float = 2.0
    height_cells: float = 1.05
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
    debug_bounds: bool = False


@dataclass(frozen=True)
class Style:
    dpi: int = 180
    font_mono: str = "DejaVu Sans Mono"
    font_label: str = "DejaVu Sans"
    font_size_seq: int = 14
    font_size_label: int = 12

    padding_x: float = 24.0
    padding_y: float = 20.0
    track_spacing: float = 22.0
    baseline_spacing: float = 56.0
    show_reverse_complement: bool = True

    color_sequence: str = "#4b5563"
    color_ticks: str = "#9ca3af"

    legend: bool = True
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
    connector_dash: tuple[float, float] = (1.0, 3.0)

    palette: Mapping[str, str] = field(default_factory=dict)
    span_link_inner_margin_bp: float = 0.25

    kmer: GlyphStyle = field(default_factory=GlyphStyle)
    motif_logo: MotifLogoStyle = field(default_factory=MotifLogoStyle)

    def __post_init__(self) -> None:
        if isinstance(self.kmer, dict):
            object.__setattr__(self, "kmer", GlyphStyle(**self.kmer))
        if isinstance(self.motif_logo, dict):
            object.__setattr__(self, "motif_logo", MotifLogoStyle(**self.motif_logo))

        ensure(self.dpi >= 72, "style.dpi must be >= 72", SchemaError)
        ensure(self.font_size_seq >= 6, "style.font_size_seq must be >= 6", SchemaError)
        ensure(self.font_size_label >= 6, "style.font_size_label must be >= 6", SchemaError)
        ensure(self.padding_x >= 0, "style.padding_x must be >= 0", SchemaError)
        ensure(self.padding_y >= 0, "style.padding_y must be >= 0", SchemaError)
        ensure(self.track_spacing > 0, "style.track_spacing must be > 0", SchemaError)
        ensure(self.baseline_spacing > 0, "style.baseline_spacing must be > 0", SchemaError)
        ensure(self.kmer.height_factor > 0, "style.kmer.height_factor must be > 0", SchemaError)
        ensure(self.kmer.pad_x_px >= 0, "style.kmer.pad_x_px must be >= 0", SchemaError)
        ensure(self.span_link_inner_margin_bp >= 0, "style.span_link_inner_margin_bp must be >= 0", SchemaError)
        ensure(self.motif_logo.height_bits > 0, "style.motif_logo.height_bits must be > 0", SchemaError)
        ensure(self.motif_logo.height_cells > 0, "style.motif_logo.height_cells must be > 0", SchemaError)
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
            str(self.kmer.text_v_align).lower() in {"baseline", "center"},
            "style.kmer.text_v_align must be 'baseline' or 'center'",
            SchemaError,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "Style":
        if not isinstance(mapping, Mapping):
            raise SchemaError("style must be a mapping/dict")
        data = dict(mapping)
        if "height_factor" in data:
            raise SchemaError("Unknown style key 'height_factor' — use 'style.kmer.height_factor'.")
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(data.keys()) - allowed)
        if unknown:
            raise SchemaError(f"Unknown style key(s): {unknown}")
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
        return cls(**data)


def _baserender_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


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
    styles_dir = _baserender_root() / "styles"
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
        styles_guess = root / "styles" / raw.name
        if styles_guess.exists():
            return styles_guess
        raise SchemaError(f"Style preset not found: {spec}")

    for suffix in (".yaml",):
        candidate = root / "styles" / f"{raw}{suffix}"
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
