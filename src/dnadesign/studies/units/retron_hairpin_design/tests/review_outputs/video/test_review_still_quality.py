"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/video/test_review_still_quality.py

Review-still quality tests for Retron review output video frames.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dnadesign.studies.units.retron_hairpin_design.review_outputs.video.frame_naming import frame_evidence_label
from dnadesign.studies.units.retron_hairpin_design.review_outputs.video.montage import (
    VIDEO_SIZE_PX,
    SequenceReviewFrame,
    write_sequence_montage,
)


def test_review_still_masks_compiler_title_and_uses_video_resolution(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    _write_source_png_with_stale_title_marker(frame.composition_overview_png)

    write_sequence_montage(
        [frame],
        out_dir=tmp_path / "out",
        deliverable_plan_id="teto_retained_span_trim_tetr_pwm_elite_v1",
        materialized_root=tmp_path / "materialized",
        review_variant_ids={"r180-w02-17": "pES-retron-199"},
        video_writer=_write_mock_video,
    )

    still_path = tmp_path / "out" / "reviews" / "video" / "stills" / "01_pES-retron-199_tetO-w02-17.png"
    assert still_path.name == "01_pES-retron-199_tetO-w02-17.png"
    with Image.open(still_path) as image:
        assert image.size == VIDEO_SIZE_PX
    assert not _has_stale_title_marker(still_path)
    assert _has_top_content_marker(still_path)
    label = frame_evidence_label(frame, review_variant_ids={"r180-w02-17": "pES-retron-199"})
    assert label == "pES-retron-199 | tetO PWM [2,17) | r180 scaffold | 15 nt"


def _frame(tmp_path: Path) -> SequenceReviewFrame:
    source_png = tmp_path / "source" / "composition_overview.png"
    row = {
        "variant_id": "r180-w02-17",
        "construct_id": "pES-tetr-r180-w02-17",
        "msd_design_id": "msd-tetr-r180-w02-17",
        "scaffold_context": "retron180",
        "variant_role": "trim_candidate",
        "payload_trim_id": "TetR_w02_17",
        "payload_trim_display": "mild trim",
        "retained_parent_span_0": "[2, 17]",
        "insert_nt": "15",
        "folding_status": "ok",
        "composition_overview_png": str(source_png),
        "genbank_path": str(tmp_path / "clone" / "sequence.gb"),
        "fasta_path": str(tmp_path / "clone" / "sequence.fa"),
        "csv_path": str(tmp_path / "clone" / "sequence.csv"),
    }
    return SequenceReviewFrame(
        order=1,
        construct_id=row["construct_id"],
        msd_design_id=row["msd_design_id"],
        payload_trim_id=row["payload_trim_id"],
        scaffold_context=row["scaffold_context"],
        variant_role=row["variant_role"],
        rt_mode="wt_eco1",
        composition_overview_png=source_png,
        row=row,
    )


def _write_source_png_with_stale_title_marker(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 2, 239), fill=(30, 30, 30))
    draw.rectangle((317, 0, 319, 239), fill=(30, 30, 30))
    draw.rectangle((10, 12, 60, 30), fill=(255, 0, 180))
    draw.rectangle((110, 12, 210, 30), fill=(255, 0, 180))
    draw.rectangle((70, 48, 250, 58), fill=(0, 170, 120))
    draw.rectangle((32, 92, 288, 152), fill=(88, 143, 126))
    image.save(path)


def _has_stale_title_marker(path: Path) -> bool:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
        return bool(((pixels[..., 0] > 220) & (pixels[..., 1] < 70) & (pixels[..., 2] > 130)).any())


def _has_top_content_marker(path: Path) -> bool:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
        return int(((pixels[..., 1] > 145) & (pixels[..., 0] < 40) & (pixels[..., 2] < 140)).sum()) > 15000


def _write_mock_video(**kwargs) -> None:
    kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
    kwargs["output_path"].write_bytes(b"mock mp4")
