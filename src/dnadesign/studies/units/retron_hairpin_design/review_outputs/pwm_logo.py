"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm_logo.py

Study-owned bidirectional TetR PWM sequence-row review triptych rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .plan import PwmTrimPanel
from .pwm_panel_labels import panel_title
from .pwm_panel_metadata import panel_metadata_attributes, panel_subtitle, trim_state_elements
from .pwm_sequence_rows import (
    PANEL_WIDTH,
    PwmLogoColumn,
    PwmLogoLayer,
    render_pwm_sequence_row_panel,
)
from .pwm_typography import SUBTITLE_FONT_SIZE, TITLE_FONT_SIZE, TYPOGRAPHIC_SCALE_ID

INK = "#1F2937"
MUTED = "#667085"
LOGO_STYLE_ID = "baserender_sequence_rows_tetr_dual_site_trim_logo_v7"
PANEL_GUTTER = 34
OUTER_MARGIN_X = 36
TOP_LABEL_HEIGHT = 76
BOTTOM_MARGIN = 10


def write_pwm_logo_triptych(
    columns: Sequence[PwmLogoColumn],
    *,
    parent_sequence: str,
    logo_layers: Sequence[PwmLogoLayer],
    panels: Sequence[PwmTrimPanel],
    source_path: Path,
    svg_path: Path,
    png_path: Path,
) -> None:
    panel_images = [
        render_pwm_sequence_row_panel(
            columns,
            parent_sequence=parent_sequence,
            panel=panel,
            logo_layers=logo_layers,
        )
        for panel in panels
    ]
    png_image = _compose_triptych(panel_images, panels=panels)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_image.save(png_path)
    _write_svg_wrapper(
        png_image,
        columns=columns,
        logo_layers=logo_layers,
        panels=panels,
        source_path=source_path,
        svg_path=svg_path,
    )


def _compose_triptych(
    panel_images: Sequence[Image.Image],
    *,
    panels: Sequence[PwmTrimPanel],
) -> Image.Image:
    panel_height = max(image.height for image in panel_images)
    width = OUTER_MARGIN_X * 2 + len(panel_images) * PANEL_WIDTH + (len(panel_images) - 1) * PANEL_GUTTER
    height = TOP_LABEL_HEIGHT + panel_height + BOTTOM_MARGIN
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(TITLE_FONT_SIZE)
    meta_font = _font(SUBTITLE_FONT_SIZE)
    for index, (image, panel) in enumerate(zip(panel_images, panels, strict=True)):
        x = OUTER_MARGIN_X + index * (PANEL_WIDTH + PANEL_GUTTER)
        panel_center_x = x + PANEL_WIDTH / 2
        _draw_centered_text(draw, panel_title(panel), center_x=panel_center_x, y=6, fill=INK, font=title_font)
        _draw_centered_text(
            draw,
            panel_subtitle(panel),
            center_x=panel_center_x,
            y=37,
            fill=MUTED,
            font=meta_font,
        )
        canvas.paste(image, (x, TOP_LABEL_HEIGHT + (panel_height - image.height) // 2))
    return canvas


def _write_svg_wrapper(
    image: Image.Image,
    *,
    columns: Sequence[PwmLogoColumn],
    logo_layers: Sequence[PwmLogoLayer],
    panels: Sequence[PwmTrimPanel],
    source_path: Path,
    svg_path: Path,
) -> None:
    handle = BytesIO()
    image.save(handle, format="PNG")
    encoded = base64.b64encode(handle.getvalue()).decode("ascii")
    metadata = "\n".join(_metadata_lines(columns, logo_layers=logo_layers, panels=panels, source_path=source_path))
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{image.width}" height="{image.height}" '
                f'viewBox="0 0 {image.width} {image.height}" role="img">',
                metadata,
                f'<image width="{image.width}" height="{image.height}" href="data:image/png;base64,{encoded}"/>',
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _metadata_lines(
    columns: Sequence[PwmLogoColumn],
    *,
    logo_layers: Sequence[PwmLogoLayer],
    panels: Sequence[PwmTrimPanel],
    source_path: Path,
) -> list[str]:
    lines = [
        (
            '<metadata contract="retron_hairpin_teto_pwm_logo_triptych_svg_v1" '
            f'data-logo-style="{LOGO_STYLE_ID}" data-renderer="baserender_sequence_rows" '
            'data-source-rendering="metadata_only" '
            f'data-typographic-scale="{TYPOGRAPHIC_SCALE_ID}" '
            'data-sequence-context="tetr_dual_site_top_bottom_strands" '
            'data-site-coordinate-system="tetr_monotypic_elite_parent_19nt" '
            'data-feature-box="retained_payload_span" '
            'data-visual-layers="full_site_backdrop,retained_payload_overlay,dual_motif_logos,trim_cut_lines" '
            'data-boundary-tick-policy="retained_span_edges_only" '
            'data-retained-span-bracket="retained_payload" data-min-critical-font-size-px="16" '
            'data-full-site-backdrop-0="0..19" '
            'data-letter-coloring="match_window_seq_trim_inclusion" data-scale-bar="2_bits_left_of_logo" '
            f'data-motif-layer-count="{len(logo_layers)}" data-logo-render-span-0="0..19">'
        ),
        f"<pwm-source>{escape(source_path.as_posix())}</pwm-source>",
    ]
    for layer in logo_layers:
        lines.append(
            f'<motif-layer data-motif-layer="{escape(layer.motif_instance_id)}" '
            f'data-strand="{escape(layer.strand)}" data-span-0="{layer.start_0}..{layer.end_0}" '
            f'data-occurrence-rank="{layer.occurrence_rank}"/>'
        )
    for panel in panels:
        lines.append(f"<pwm-panel {panel_metadata_attributes(columns, panel)}>")
        lines.extend(trim_state_elements(columns, panel))
        lines.append("</pwm-panel>")
    lines.append("</metadata>")
    return lines


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    center_x: float,
    y: int,
    fill: str,
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    draw.text((center_x - text_width / 2, y), text, fill=fill, font=font)


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


__all__ = ["PwmLogoColumn", "PwmLogoLayer", "write_pwm_logo_triptych"]
