"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/video/stills.py

Review-still rendering for Retron sequence montage artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..sequence.index import SequenceReviewFrame
from ..sequence.variant_identity import identity_for_frame
from .frame_naming import frame_filename_stem, review_construct_id

EDGE_COLUMN_MAX_RGB_THRESHOLD = 250
EDGE_MAX_COLUMNS = 16
EDGE_MIN_FRACTION = 0.85
STILL_SIZE_PX = (1920, 1080)
TITLE_BAND_HEIGHT_PX = 188
BODY_MARGIN_PX = 36
SOURCE_HEADER_FRACTION = 0.19
TITLE_COLOR = (31, 41, 55)
SUBTITLE_COLOR = (75, 85, 99)


def write_review_stills(
    frames: Sequence[SequenceReviewFrame],
    *,
    stills_dir: Path,
    review_variant_ids: Mapping[str, str],
) -> tuple[Path, ...]:
    _clear_existing_stills(stills_dir)
    return tuple(
        _write_review_still(frame, stills_dir=stills_dir, review_variant_ids=review_variant_ids) for frame in frames
    )


def _write_review_still(frame: SequenceReviewFrame, *, stills_dir: Path, review_variant_ids: Mapping[str, str]) -> Path:
    source = _remove_source_title_band(
        _trim_edge_artifact_columns(Image.open(frame.composition_overview_png).convert("RGB"))
    )
    canvas = Image.new("RGB", STILL_SIZE_PX, color="white")
    body_size = (STILL_SIZE_PX[0] - BODY_MARGIN_PX * 2, STILL_SIZE_PX[1] - TITLE_BAND_HEIGHT_PX - BODY_MARGIN_PX)
    fitted = ImageOps.contain(source, body_size)
    canvas.paste(fitted, ((STILL_SIZE_PX[0] - fitted.width) // 2, TITLE_BAND_HEIGHT_PX))
    _draw_review_title(canvas, frame=frame, review_variant_ids=review_variant_ids)
    stills_dir.mkdir(parents=True, exist_ok=True)
    path = stills_dir / f"{frame_filename_stem(frame, review_variant_ids=review_variant_ids)}.png"
    canvas.save(path)
    return path


def _draw_review_title(
    canvas: Image.Image, *, frame: SequenceReviewFrame, review_variant_ids: Mapping[str, str]
) -> None:
    draw = ImageDraw.Draw(canvas)
    identity = identity_for_frame(frame)
    title = f"{review_construct_id(frame, review_variant_ids=review_variant_ids)} | {identity.payload_label}"
    subtitle = f"{identity.scaffold} scaffold | {identity.insert_nt} nt"
    _draw_centered_text(draw, canvas.width, title, y=62, font=_font(54, bold=True), fill=TITLE_COLOR)
    _draw_centered_text(draw, canvas.width, subtitle, y=126, font=_font(36), fill=SUBTITLE_COLOR)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    width: int,
    text: str,
    *,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    draw.text(((width - text_width) // 2, y - text_height // 2), text, fill=fill, font=font)


@lru_cache(maxsize=8)
def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _clear_existing_stills(stills_dir: Path) -> None:
    if stills_dir.exists():
        for path in stills_dir.glob("*.png"):
            path.unlink()


def _remove_source_title_band(image: Image.Image) -> Image.Image:
    crop_top = round(image.height * SOURCE_HEADER_FRACTION)
    if crop_top <= 0 or crop_top >= image.height:
        return image
    return image.crop((0, crop_top, image.width, image.height))


def _trim_edge_artifact_columns(image: Image.Image) -> Image.Image:
    left = 0
    right = image.width
    while left < min(EDGE_MAX_COLUMNS, right - 1) and _is_edge_artifact_column(image, left):
        left += 1
    while right > max(left + 1, image.width - EDGE_MAX_COLUMNS) and _is_edge_artifact_column(image, right - 1):
        right -= 1
    if left == 0 and right == image.width:
        return image
    return image.crop((left, 0, right, image.height))


def _is_edge_artifact_column(image: Image.Image, x: int) -> bool:
    artifact = sum(1 for y in range(image.height) if max(image.getpixel((x, y))[:3]) < EDGE_COLUMN_MAX_RGB_THRESHOLD)
    return artifact / image.height >= EDGE_MIN_FRACTION


__all__ = ["STILL_SIZE_PX", "write_review_stills"]
