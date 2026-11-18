"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/nucleotide_geometry.py

Shared nucleotide ink metrics and feature-box geometry for layout and drawing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from ..config import Style
from ..core import BoundsError, Feature

_ALPHABET_REFERENCES = {
    "DNA": "ACGTNacgtn",
    "IUPAC_DNA": "ACGTRYSWKMBDHVNacgtryswkmbdhvn",  # pragma: allowlist secret
    "RNA": "ACGUNacgun",
    "PROTEIN": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*",  # pragma: allowlist secret
}


@lru_cache(maxsize=2048)
def glyph_path(text: str, family: str, size_pt: float, weight: str = "normal") -> TextPath:
    return TextPath((0, 0), text, prop=FontProperties(family=family, size=size_pt, weight=weight), usetex=False)


def nucleotide_font_size(style: Style) -> float:
    return float(style.font_size_feature_label if style.font_size_feature_label is not None else style.font_size_seq)


def displayed_nucleotides(feature: Feature) -> str:
    label = feature.label or ""
    return label[::-1] if feature.span.strand == "rev" and feature.kind != "interval_annotation" else label


@dataclass(frozen=True)
class NucleotideRun:
    left_px: float
    right_px: float
    bottom_px: float
    top_px: float
    widest_glyph_px: float

    @property
    def midpoint_y_px(self) -> float:
        return (self.bottom_px + self.top_px) / 2.0

    @property
    def height_px(self) -> float:
        return self.top_px - self.bottom_px


def nucleotide_run(label: str, style: Style, *, cell_width: float) -> NucleotideRun | None:
    """Measure visible ink with each glyph centered on its own sequence column."""
    unit = float(style.dpi) / 72.0
    size = nucleotide_font_size(style)
    boxes = []
    for index, char in enumerate(label):
        if char.isspace():
            continue
        bounds = glyph_path(char, style.font_mono, size).get_extents()
        width = bounds.width * unit
        center = (index + 0.5) * cell_width
        boxes.append((center - width / 2, center + width / 2, bounds.y0 * unit, bounds.y1 * unit, width))
    if not boxes:
        return None
    return NucleotideRun(
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
        max(b[4] for b in boxes),
    )


@dataclass(frozen=True)
class NucleotideBox:
    paint_bounds: tuple[float, float, float, float]
    glyph_midpoint_y_px: float | None


def _required_box_height(run: NucleotideRun, style: Style, cell_height: float) -> float:
    nudge = abs(float(style.kmer.text_y_nudge_cells) * cell_height)
    return run.height_px + 2 * style.kmer.pad_y_px + 2 * nudge


@dataclass(frozen=True)
class NucleotideRequirements:
    minimum_pitch_px: float
    minimum_box_height_px: float


def nucleotide_layout_requirements(
    features: tuple[Feature, ...], style: Style, *, cell_width: float, cell_height: float, alphabet: str
) -> NucleotideRequirements:
    runs = [
        run
        for feature in features
        if feature.kind != "interval_annotation"
        and (run := nucleotide_run(feature.label or "", style, cell_width=cell_width)) is not None
    ]
    if not runs:
        return NucleotideRequirements(0.0, 0.0)
    if alphabet not in _ALPHABET_REFERENCES:
        raise BoundsError(f"cannot reserve nucleotide geometry for unsupported alphabet {alphabet!r}")
    reference = nucleotide_run(_ALPHABET_REFERENCES[alphabet], style, cell_width=cell_width)
    if reference is None:
        raise BoundsError("could not measure the nucleotide alphabet")
    all_runs = [reference, *runs]
    feature_cell = (
        glyph_path("Ag", style.font_mono, nucleotide_font_size(style)).get_extents().height * style.dpi / 72.0
    )
    return NucleotideRequirements(
        max(run.widest_glyph_px + style.kmer.min_glyph_gap_px for run in all_runs),
        max(
            feature_cell * style.kmer.box_height_cells,
            *(_required_box_height(run, style, cell_height) for run in all_runs),
        ),
    )


def nucleotide_box(
    label: str,
    style: Style,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    cell_width: float,
    cell_height: float,
) -> NucleotideBox:
    run = nucleotide_run(label, style, cell_width=cell_width)
    center_x = x + width / 2.0
    box_width = width + 2 * style.kmer.pad_x_px
    box_height = height
    if run is not None:
        center_x = x + (run.left_px + run.right_px) / 2.0
        box_width = max(box_width, run.right_px - run.left_px + 2 * style.kmer.pad_x_px)
        box_height = max(box_height, _required_box_height(run, style, cell_height))
    return NucleotideBox(
        (center_x - box_width / 2, y - box_height / 2, center_x + box_width / 2, y + box_height / 2),
        None if run is None else run.midpoint_y_px,
    )
