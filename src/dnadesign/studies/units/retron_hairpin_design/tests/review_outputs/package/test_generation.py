"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/package/test_generation.py

Tests for tetO PWM trim review-package generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from dnadesign.studies.units.retron_hairpin_design.review_outputs.handoff.contract import (
    SEQUENCE_HANDOFF_COLUMNS,
)
from dnadesign.studies.units.retron_hairpin_design.review_outputs.service import (
    generate_teto_pwm_trim_rescue_review_outputs,
)

from ...support.paths import repo_root_from
from ...support.review_ids import EXPECTED_TETO_TRIM_REVIEW_VARIANT_IDS
from ...support.review_outputs import fake_video_writer, write_fake_materialized_bundle
from ...support.review_plans import write_review_plan_with_test_pwm


def test_teto_pwm_trim_review_outputs_generate_review_package(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    deliverable_plan_path = write_review_plan_with_test_pwm(tmp_path / "plan", repo_root=repo_root)
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    out_dir = tmp_path / "workbench" / "outputs" / "teto_pwm_trim_rescue_v1"

    result = generate_teto_pwm_trim_rescue_review_outputs(
        deliverable_plan_path=deliverable_plan_path,
        materialized_root=materialized_root,
        out_dir=out_dir,
        repo_root=repo_root,
        video_writer=fake_video_writer,
    )

    assert result.sequence_row_count == 9
    assert result.handoff_verified_count == 9
    assert result.pwm_triptych_svg == out_dir / "reviews" / "pwm" / "teto_pwm_trim_rescue_v1.pwm_trim_triptych.svg"
    assert result.pwm_triptych_png == out_dir / "reviews" / "pwm" / "teto_pwm_trim_rescue_v1.pwm_trim_triptych.png"
    assert result.sequence_montage_mp4 == out_dir / "reviews" / "video" / "teto_pwm_trim_rescue_v1.sequence_montage.mp4"
    assert result.handoff_tsv == out_dir / "reviews" / "handoff" / "teto_pwm_trim_rescue_v1.handoff.tsv"
    assert result.handoff_markdown == (out_dir / "reviews" / "handoff" / "teto-pwm-trim-rescue-v1.handoff.md")
    assert result.benchling_genbank_dir == out_dir / "benchling_genbank"
    assert result.benchling_genbank_index == out_dir / "reviews" / "handoff" / (
        "teto_pwm_trim_rescue_v1.benchling_genbank.tsv"
    )
    assert result.benchling_genbank_count == 6
    assert result.review_manifest_path == out_dir / "reviews" / "review_manifest.json"
    assert result.pwm_triptych_svg.read_text(encoding="utf-8").startswith("<?xml")
    assert result.pwm_triptych_png.read_bytes().startswith(b"\x89PNG")
    with Image.open(result.pwm_triptych_png) as image:
        assert image.size[0] >= 3000
        assert image.size[1] >= 800
    assert result.sequence_montage_mp4.read_bytes() == b"fake-mp4"

    _assert_triptych_contract(result.pwm_triptych_svg.read_text(encoding="utf-8"))
    _assert_review_manifest(result.review_manifest_path)
    _assert_handoff_index(result.handoff_tsv, result.handoff_markdown, out_dir=out_dir)
    _assert_benchling_import(result.benchling_genbank_dir, result.benchling_genbank_index)
    _assert_video_manifest(result.sequence_montage_manifest, out_dir=out_dir)


def _assert_triptych_contract(triptych_svg: str) -> None:
    expected_fragments = [
        'data-logo-style="baserender_sequence_rows_tetr_dual_site_trim_logo_v7"',
        'data-renderer="baserender_sequence_rows"',
        'data-source-rendering="metadata_only"',
        'data-typographic-scale="title_42_subtitle_32_sequence_16_scale_13"',
        'data-sequence-context="tetr_dual_site_top_bottom_strands"',
        'data-site-coordinate-system="tetr_monotypic_elite_parent_19nt"',
        'data-feature-box="retained_payload_span"',
        'data-full-site-backdrop-0="0..19"',
        'data-motif-layer-count="2"',
        'data-motif-layer="tetR:0:17:+:1" data-strand="+"',
        'data-motif-layer="tetR:2:19:-:2" data-strand="-"',
        'data-trim-edge-policy="cut_lines_with_span_in_subtitle"',
        'data-retained-span-bracket="retained_payload"',
        'data-min-critical-font-size-px="16"',
        'data-letter-coloring="match_window_seq_trim_inclusion"',
        'data-scale-bar="2_bits_left_of_logo"',
        'data-logo-render-span-0="0..19"',
        'data-display-title="Full site"',
        'data-compact-subtitle="19 nt | [0,19) | 100% IC"',
        'data-display-title="Trim 02-17"',
        'data-compact-subtitle="15 nt | [2,17) | 96% IC"',
        'data-retained-information-fraction="0.964248"',
        'data-display-title="Trim 03-16"',
        'data-compact-subtitle="13 nt | [3,16) | 92% IC"',
        'data-retained-information-fraction="0.915756"',
        'data-retained-feature-label-5to3="TATATCTGATATA"',
    ]
    for fragment in expected_fragments:
        assert fragment in triptych_svg

    assert 'data-visual-layers="full_site_backdrop,retained_payload_overlay,dual_motif_logos,trim_cut_lines"' in (
        triptych_svg
    )
    assert 'data-retained-edge-cuts-0="3,16"' in triptych_svg
    assert 'data-observed-sequence-5to3="NNNTATATCTGATATANNN"' in triptych_svg
    assert 'data-visible-trim-summary="removed 3+3 nt; retained 13 nt; retained PWM information 91.6%"' in (
        triptych_svg
    )


def _assert_review_manifest(path: Path) -> None:
    review_manifest = json.loads(path.read_text(encoding="utf-8"))
    assert review_manifest["contract"] == "retron_hairpin_review_output_manifest_v1"
    assert review_manifest["deliverable_plan_id"] == "teto_pwm_trim_rescue_v1"
    assert review_manifest["materialized_sequence_rows"] == 9
    assert review_manifest["handoff_verified_count"] == 9
    assert "clone_handoff_verified_count" not in review_manifest
    assert review_manifest["sequence_evidence"] == {
        "folding_status_ok_count": 9,
        "native_structure_png_verified_count": 9,
        "reverse_complement_verified_count": 9,
    }
    assert review_manifest["pwm_triptych"]["payload_trim_ids"] == [
        "TetR_w00_19",
        "TetR_w02_17",
        "TetR_w03_16",
    ]
    assert review_manifest["sequence_montage"]["frame_count"] == 9
    assert review_manifest["sequence_montage"]["review_variant_ids"] == EXPECTED_TETO_TRIM_REVIEW_VARIANT_IDS
    assert review_manifest["source_indexes"]["sequence_index"] == "materialized/manifest/indexes/sequence_index.tsv"
    assert "clone_handoff" not in review_manifest
    assert review_manifest["sequence_handoff"]["index_tsv"] == "reviews/handoff/teto_pwm_trim_rescue_v1.handoff.tsv"
    assert review_manifest["sequence_handoff"]["index_markdown"] == (
        "reviews/handoff/teto-pwm-trim-rescue-v1.handoff.md"
    )
    assert review_manifest["benchling_genbank_import"]["orientation"] == "reverse_complement_only"
    assert review_manifest["benchling_genbank_import"]["verified_count"] == 6
    assert review_manifest["benchling_genbank_import"]["expected_count"] == 6
    assert review_manifest["benchling_genbank_import"]["included_payload_trim_ids"] == [
        "TetR_w02_17",
        "TetR_w03_16",
    ]
    assert review_manifest["benchling_genbank_import"]["assigned_retron_ids"]["r180-w03-16"] == "pES-retron-200"
    assert review_manifest["benchling_genbank_import"]["source_precedent_ids"]["r180-w03-16"] == "pES-retron-180"
    assert review_manifest["benchling_genbank_import"]["directory"] == "benchling_genbank"
    assert review_manifest["benchling_genbank_import"]["index_tsv"] == (
        "reviews/handoff/teto_pwm_trim_rescue_v1.benchling_genbank.tsv"
    )
    assert review_manifest["benchling_genbank_import"]["files"] == [
        "benchling_genbank/pES-retron-195-msd[TetR]-r26-w02-17.gb",
        "benchling_genbank/pES-retron-196-msd[TetR]-r26-w03-16.gb",
        "benchling_genbank/pES-retron-197-msd[TetR]-r43-w02-17.gb",
        "benchling_genbank/pES-retron-198-msd[TetR]-r43-w03-16.gb",
        "benchling_genbank/pES-retron-199-msd[TetR]-r180-w02-17.gb",
        "benchling_genbank/pES-retron-200-msd[TetR]-r180-w03-16.gb",
    ]


def _assert_handoff_index(tsv_path: Path, markdown_path: Path, *, out_dir: Path) -> None:
    rows = list(csv.DictReader(tsv_path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert len(rows) == 9
    assert list(rows[0]) == list(SEQUENCE_HANDOFF_COLUMNS)
    assert rows[0]["variant_id"] == "r26-w00-19"
    assert rows[0]["construct_id"] == "pES-tetr-r26-w00-19"
    assert rows[0]["retained_window"] == "[0,19)"
    assert rows[0]["insert_nt"] == "19"
    assert rows[0]["genbank"].endswith("/sequences/forward.gb")
    assert (out_dir / rows[0]["genbank"]).is_file()
    handoff_markdown = markdown_path.read_text(encoding="utf-8")
    assert "tetO Trim Sequence Handoff" in handoff_markdown
    assert "r26-w00-19" in handoff_markdown
    assert "| Variant | Insert | Context | Files |" in handoff_markdown
    assert "Full machine metadata stays in `sequence_index.tsv`" in handoff_markdown
    assert "pES-retron-teto-trim-001" not in handoff_markdown
    assert " / msd-" not in handoff_markdown
    assert "[GB]" in handoff_markdown
    assert "[RC GB]" in handoff_markdown
    assert "[RC FA]" in handoff_markdown


def _assert_benchling_import(directory: Path, index_path: Path) -> None:
    expected_names = [
        "pES-retron-195-msd[TetR]-r26-w02-17.gb",
        "pES-retron-196-msd[TetR]-r26-w03-16.gb",
        "pES-retron-197-msd[TetR]-r43-w02-17.gb",
        "pES-retron-198-msd[TetR]-r43-w03-16.gb",
        "pES-retron-199-msd[TetR]-r180-w02-17.gb",
        "pES-retron-200-msd[TetR]-r180-w03-16.gb",
    ]
    observed = sorted(path.name for path in directory.iterdir() if not path.name.startswith("."))
    assert observed == expected_names
    assert all((directory / name).suffix == ".gb" for name in expected_names)
    first = (directory / expected_names[0]).read_text(encoding="utf-8")
    assert first.startswith("LOCUS       pES-retron-195")
    assert "derived from pES-retron-26" in first
    assert "reverse-complement MSD handoff" in first
    assert "FEATURES             Location/Qualifiers" in first
    rows = list(csv.DictReader(index_path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert [row["assigned_construct_id"] for row in rows] == [f"pES-retron-{idx}" for idx in range(195, 201)]
    assert rows[4]["variant_id"] == "r180-w02-17"


def _assert_video_manifest(path: Path, *, out_dir: Path) -> None:
    video_manifest = json.loads(path.read_text(encoding="utf-8"))
    assert video_manifest["contract"] == "retron_hairpin_sequence_montage_manifest_v1"
    assert video_manifest["still_count"] == 9
    assert video_manifest["still_resolution_px"] == {"width": 1920, "height": 1080}
    assert video_manifest["video_resolution_px"] == {"width": 1920, "height": 1080}
    assert [frame["payload_trim_id"] for frame in video_manifest["frames"][:3]] == [
        "TetR_w00_19",
        "TetR_w02_17",
        "TetR_w03_16",
    ]
    assert video_manifest["frames"][0]["variant_id"] == "r26-w00-19"
    assert video_manifest["review_variant_ids"] == EXPECTED_TETO_TRIM_REVIEW_VARIANT_IDS
    assert video_manifest["frames"][0]["review_construct_id"] == "pES-retron-26"
    assert video_manifest["frames"][0]["evidence_label"] == "pES-retron-26 | tetO PWM [0,19) | r26 scaffold | 19 nt"
    assert video_manifest["frames"][0]["review_still_png"] == "reviews/video/stills/01_pES-retron-26_tetO-w00-19.png"
    assert Path(video_manifest["frames"][0]["review_still_png"]).name.startswith("01_pES-retron-26_")
    assert video_manifest["frames"][0]["composition_overview_png"].endswith("composition_overview.png")
    assert (out_dir / video_manifest["frames"][0]["review_still_png"]).read_bytes().startswith(b"\x89PNG")
