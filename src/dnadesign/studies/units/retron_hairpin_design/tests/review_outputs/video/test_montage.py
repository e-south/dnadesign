"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/video/test_montage.py

Tests for Retron review-output sequence montage still rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.retron_hairpin_design.review_outputs.sequence.index import SequenceReviewFrame
from dnadesign.studies.units.retron_hairpin_design.review_outputs.video.montage import (
    STILL_SIZE_PX,
    write_sequence_montage,
)
from dnadesign.studies.units.retron_hairpin_design.review_outputs.video.stills import _trim_edge_artifact_columns


def test_sequence_montage_trims_compiler_png_edge_lines(tmp_path: Path) -> None:
    from PIL import Image

    source_png = tmp_path / "composition_overview.png"
    _write_source_png_with_dark_edge_lines(source_png)
    cropped = _trim_edge_artifact_columns(Image.open(source_png).convert("RGB"))
    assert _column_is_white(cropped, 0)
    assert _column_is_white(cropped, cropped.width - 1)
    frame = SequenceReviewFrame(
        order=1,
        construct_id="pES-tetr-r26-w00-19",
        msd_design_id="msd-tetr-w00-19-C26-LCGGG-RACAG-MXMX",
        payload_trim_id="TetR_w00_19",
        scaffold_context="retron26",
        variant_role="control",
        rt_mode="wt_eco1",
        composition_overview_png=source_png,
        row={},
    )

    stale_still = tmp_path / "review" / "reviews" / "video" / "stills" / "01_r26-w00-19.png"
    stale_still.parent.mkdir(parents=True)
    stale_still.write_bytes(b"stale")
    write_sequence_montage(
        (frame,),
        out_dir=tmp_path / "review",
        deliverable_plan_id="teto_retained_span_trim_tetr_pwm_elite_v1",
        materialized_root=tmp_path,
        review_variant_ids={"r26-w00-19": "pES-retron-26"},
        video_writer=_fake_video_writer,
    )

    still_path = tmp_path / "review" / "reviews" / "video" / "stills" / "01_pES-retron-26_tetO-w00-19.png"
    assert still_path.is_file()
    assert not stale_still.exists()
    assert _image_size(still_path) == STILL_SIZE_PX and not _has_full_height_dark_column(still_path)


def _write_source_png_with_dark_edge_lines(path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 180), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 48, 272, 132), fill=(80, 130, 180), outline=(20, 80, 120), width=3)
    draw.line((0, 0, 0, 179), fill="black", width=3)
    draw.line((3, 0, 3, 179), fill=(192, 192, 192), width=1)
    draw.line((316, 0, 316, 179), fill=(192, 192, 192), width=1)
    draw.line((319, 0, 319, 179), fill="black", width=3)
    image.save(path)


def _fake_video_writer(*, output_path: Path, **kwargs: object) -> None:
    output_path.write_bytes(b"fake-mp4")


def _column_is_white(image, x: int) -> bool:
    return all(image.getpixel((x, y)) == (255, 255, 255) for y in range(image.height))


def _has_full_height_dark_column(path: Path) -> bool:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    return any(
        sum(1 for y in range(image.height) if max(image.getpixel((x, y))) < 48) / image.height > 0.85
        for x in range(image.width)
    )


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size
