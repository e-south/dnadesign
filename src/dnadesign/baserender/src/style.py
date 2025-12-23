"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/style.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Mapping

from .contracts import SchemaError, ensure


@dataclass(frozen=True)
class GlyphStyle:
    # Rounded rectangle radius (px) for k‑mer boxes.
    round_px: float = 4.0
    # Internal padding inside the rounded box (px).
    pad_px: float = 2.0
    # Outline width (px). 0.0 = facecolor only.
    edge_width: float = 0.0
    # Box fill opacity (1.0 = opaque).
    fill_alpha: float = 0.92  # was 1.00
    # Multiplier of monospaced char height → box height (usually 1.0–1.5).
    height_factor: float = 1.15
    # Text color used atop colored boxes.
    text_color: str = "#ffffff"
    # Extra horizontal padding added to each k-mer box (px on each side).
    pad_x_px: float = 1.0
    # Vertical placement of text inside k‑mer boxes:
    #  - "baseline": keep letters aligned to the reference sequence row
    #  - "center":   center letters within the rounded box (useful when height grows)
    text_v_align: str = "baseline"
    # Optional fine‑tuning (pixels). Positive = up, negative = down.
    text_v_offset_px: float = 0.0


@dataclass(frozen=True)
class Style:
    # Output DPI; affects pixel sizes and font metrics. Must be >= 72.
    dpi: int = 180
    font_mono: str = "DejaVu Sans Mono"
    font_label: str = "DejaVu Sans"
    font_size_seq: int = 14
    font_size_label: int = 12

    # Left/right padding (px).
    padding_x: float = 24.0
    # Top/bottom padding (px). Increasing this increases natural frame height.
    padding_y: float = 20.0
    # Vertical gap (px) between annotation tracks.
    track_spacing: float = 22.0
    # Distance (px) between forward and reverse baselines.
    baseline_spacing: float = 56.0
    show_reverse_complement: bool = True

    color_sequence: str = "#4b5563"
    color_ticks: str = "#9ca3af"

    # Legend (one row)
    legend: bool = True
    legend_font_size: int = 11
    legend_patch_w: float = 18.0
    legend_patch_h: float = 12.0
    legend_gap_x: float = 14.0  # gap between items (after text)
    legend_pad_px: float = 16.0  # bottom padding from figure edge
    legend_height_px: float = 40.0  # reserved drawing area (ensure no overlap
    legend_gap_patch_text: float = 6.0  # patch → text horizontal gap
    legend_center: bool = True

    # Connectors between baselines
    connectors: bool = True
    connector_alpha: float = 0.45
    connector_width: float = 0.6
    connector_dash: tuple[float, float] = (1.0, 3.0)

    # Palette overrides by tag
    palette: Mapping[str, str] = field(default_factory=dict)
    # Sigma link line default inner margin (in bases; can be fractional).
    sigma_link_inner_margin_bp: float = 0.25

    # k-mer glyph style container
    kmer: GlyphStyle = field(default_factory=GlyphStyle)

    # Allow YAML to pass a mapping for `kmer` that we coerce into GlyphStyle.
    def __post_init__(self):
        if isinstance(self.kmer, dict):
            object.__setattr__(self, "kmer", GlyphStyle(**self.kmer))
        # Basic sanity checks: proactive, fail-fast.
        ensure(self.dpi >= 72, "style.dpi must be >= 72", SchemaError)
        ensure(self.font_size_seq >= 6, "style.font_size_seq must be >= 6", SchemaError)
        ensure(self.font_size_label >= 6, "style.font_size_label must be >= 6", SchemaError)
        ensure(self.padding_x >= 0, "style.padding_x must be >= 0", SchemaError)
        ensure(self.padding_y >= 0, "style.padding_y must be >= 0", SchemaError)
        ensure(self.track_spacing > 0, "style.track_spacing must be > 0", SchemaError)
        ensure(self.baseline_spacing > 0, "style.baseline_spacing must be > 0", SchemaError)
        ensure(
            self.kmer.height_factor > 0,
            "style.kmer.height_factor must be > 0",
            SchemaError,
        )
        ensure(self.kmer.pad_x_px >= 0, "style.kmer.pad_x_px must be >= 0", SchemaError)
        ensure(
            self.sigma_link_inner_margin_bp >= 0,
            "style.sigma_link_inner_margin_bp must be >= 0",
            SchemaError,
        )
        # Validate k‑mer text alignment mode
        align = str(self.kmer.text_v_align).lower()
        ensure(
            align in {"baseline", "center"},
            "style.kmer.text_v_align must be 'baseline' or 'center'",
            SchemaError,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "Style":
        """
        Strict constructor from a mapping:
        - rejects unknown top-level keys (prevents typos),
        - rejects misplaced 'height_factor' (must be under 'kmer'),
        - enforces that 'kmer' (if provided) is a mapping.
        """
        if not isinstance(mapping, Mapping):
            raise SchemaError("style must be a mapping/dict")
        top = dict(mapping)
        if "height_factor" in top:
            raise SchemaError("Unknown style key 'height_factor' — use 'style.kmer.height_factor'.")
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(k for k in top.keys() if k not in allowed)
        if unknown:
            raise SchemaError(f"Unknown style key(s): {unknown}")
        kmer_cfg = top.get("kmer")
        if kmer_cfg is not None and not isinstance(kmer_cfg, Mapping):
            raise SchemaError("style.kmer must be a mapping")
        return cls(**top)
