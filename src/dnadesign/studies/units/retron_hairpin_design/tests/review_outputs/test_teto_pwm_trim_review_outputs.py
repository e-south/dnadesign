"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/test_teto_pwm_trim_review_outputs.py

Tests for tetO PWM trim rescue review-output generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dnadesign.studies.units.retron_hairpin_design.compiler.exceptions import RetronMsdCompilerError
from dnadesign.studies.units.retron_hairpin_design.interfaces.cli import review_outputs as cli_review_outputs_module
from dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app import app
from dnadesign.studies.units.retron_hairpin_design.review_outputs.service import (
    generate_teto_pwm_trim_rescue_review_outputs,
)

from ..support.cli import RUNNER
from ..support.paths import repo_root_from
from ..support.review_outputs import fake_video_writer, write_fake_materialized_bundle


def test_teto_pwm_trim_review_outputs_generate_review_package(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    out_dir = tmp_path / "workbench" / "outputs" / "teto_pwm_trim_rescue_v1"

    result = generate_teto_pwm_trim_rescue_review_outputs(
        deliverable_plan_path=study_dir / "workbench" / "deliverables" / "teto_pwm_trim_rescue_v1.yaml",
        materialized_root=materialized_root,
        out_dir=out_dir,
        repo_root=repo_root,
        video_writer=fake_video_writer,
    )

    assert result.sequence_row_count == 9
    assert result.clone_handoff_verified_count == 9
    assert result.pwm_triptych_svg == out_dir / "reviews" / "pwm" / "teto_pwm_trim_rescue_v1.pwm_trim_triptych.svg"
    assert result.pwm_triptych_png == out_dir / "reviews" / "pwm" / "teto_pwm_trim_rescue_v1.pwm_trim_triptych.png"
    assert result.sequence_montage_mp4 == out_dir / "reviews" / "video" / "teto_pwm_trim_rescue_v1.sequence_montage.mp4"
    assert result.review_manifest_path == out_dir / "reviews" / "review_manifest.json"
    assert result.pwm_triptych_svg.read_text(encoding="utf-8").startswith("<?xml")
    assert result.pwm_triptych_png.read_bytes().startswith(b"\x89PNG")
    assert result.sequence_montage_mp4.read_bytes() == b"fake-mp4"

    triptych_svg = result.pwm_triptych_svg.read_text(encoding="utf-8")
    assert 'data-logo-style="baserender_sequence_rows_tetr_dual_site_trim_logo_v7"' in triptych_svg
    assert 'data-renderer="baserender_sequence_rows"' in triptych_svg
    assert 'data-source-rendering="metadata_only"' in triptych_svg
    assert 'data-typographic-scale="title_24_subtitle_18_sequence_16_boundary_11"' in triptych_svg
    assert 'data-sequence-context="tetr_dual_site_top_bottom_strands"' in triptych_svg
    assert 'data-site-coordinate-system="tetr_monotypic_elite_parent_19nt"' in triptych_svg
    assert 'data-feature-box="retained_payload_span"' in triptych_svg
    assert 'data-visual-layers="full_site_backdrop,retained_payload_overlay,dual_motif_logos,trim_cut_lines"' in (
        triptych_svg
    )
    assert 'data-full-site-backdrop-0="0..19"' in triptych_svg
    assert 'data-motif-layer-count="2"' in triptych_svg
    assert 'data-motif-layer="tetR:0:17:+:1" data-strand="+"' in triptych_svg
    assert 'data-motif-layer="tetR:2:19:-:2" data-strand="-"' in triptych_svg
    assert 'data-boundary-tick-policy="retained_span_edges_only"' in triptych_svg
    assert 'data-retained-span-bracket="retained_payload"' in triptych_svg
    assert 'data-min-critical-font-size-px="16"' in triptych_svg
    assert 'data-letter-coloring="match_window_seq_trim_inclusion"' in triptych_svg
    assert 'data-scale-bar="2_bits_left_of_logo"' in triptych_svg
    assert 'data-logo-render-span-0="0..19"' in triptych_svg
    assert 'data-payload-trim-id="TetR_full"' in triptych_svg
    assert 'data-display-title="Full dual-site"' in triptych_svg
    assert 'data-compact-subtitle="19 nt | [0,19) | 100% IC"' in triptych_svg
    assert 'data-boundary-ticks-0="0,19"' in triptych_svg
    assert 'data-boundary-tick-font-size-px="11"' in triptych_svg
    assert 'data-observed-sequence-5to3="CTCTATATCTGATATAGAG"' in triptych_svg
    assert 'data-retained-feature-span-0="0..19"' in triptych_svg
    assert 'data-retained-feature-label-5to3="CTCTATATCTGATATAGAG"' in triptych_svg
    assert 'data-trim-5p-nt="0" data-trim-3p-nt="0" data-retained-nt="19"' in triptych_svg
    assert 'data-payload-trim-id="TetR_trim_conservative"' in triptych_svg
    assert 'data-display-title="Mild trim"' in triptych_svg
    assert 'data-compact-subtitle="15 nt | [2,17) | 96% IC"' in triptych_svg
    assert 'data-boundary-ticks-0="2,17"' in triptych_svg
    assert 'data-observed-sequence-5to3="NNCTATATCTGATATAGNN"' in triptych_svg
    assert 'data-retained-feature-span-0="2..17"' in triptych_svg
    assert 'data-retained-feature-label-5to3="CTATATCTGATATAG"' in triptych_svg
    assert 'data-trim-5p-nt="2" data-trim-3p-nt="2" data-retained-nt="15"' in triptych_svg
    assert 'data-retained-information-fraction="0.964248"' in triptych_svg
    assert 'data-visible-trim-summary="removed 2+2 nt; retained 15 nt; retained PWM information 96.4%"' in triptych_svg
    assert 'data-payload-trim-id="TetR_trim_aggressive"' in triptych_svg
    assert 'data-display-title="Stronger trim"' in triptych_svg
    assert 'data-compact-subtitle="12 nt | [3,15) | 87% IC"' in triptych_svg
    assert 'data-boundary-ticks-0="3,15"' in triptych_svg
    assert 'data-observed-sequence-5to3="NNNTATATCTGATATNNNN"' in triptych_svg
    assert 'data-retained-feature-span-0="3..15"' in triptych_svg
    assert 'data-retained-feature-label-5to3="TATATCTGATAT"' in triptych_svg
    assert 'data-trim-5p-nt="3" data-trim-3p-nt="4" data-retained-nt="12"' in triptych_svg
    assert 'data-retained-information-fraction="0.867985"' in triptych_svg
    assert 'data-visible-trim-summary="removed 3+4 nt; retained 12 nt; retained PWM information 86.8%"' in triptych_svg
    assert 'data-parent-position="0" data-trim-state="included"' in triptych_svg
    assert 'data-parent-position="18" data-trim-state="included"' in triptych_svg
    assert 'data-parent-position="0" data-trim-state="excluded"' in triptych_svg
    assert 'data-parent-position="1" data-trim-state="excluded"' in triptych_svg
    assert 'data-parent-position="2" data-trim-state="excluded"' in triptych_svg
    assert 'data-parent-position="15" data-trim-state="included"' in triptych_svg
    assert 'data-parent-position="16" data-trim-state="excluded"' in triptych_svg
    assert 'data-parent-position="17" data-trim-state="excluded"' in triptych_svg
    assert 'data-parent-position="18" data-trim-state="excluded"' in triptych_svg
    assert 'data-parent-position="2" data-trim-state="included"' in triptych_svg
    assert 'data-parent-position="16" data-trim-state="included"' in triptych_svg

    review_manifest = json.loads(result.review_manifest_path.read_text(encoding="utf-8"))
    assert review_manifest["contract"] == "retron_hairpin_review_output_manifest_v1"
    assert review_manifest["deliverable_plan_id"] == "teto_pwm_trim_rescue_v1"
    assert review_manifest["materialized_sequence_rows"] == 9
    assert review_manifest["clone_handoff_verified_count"] == 9
    assert review_manifest["sequence_evidence"] == {
        "folding_status_ok_count": 9,
        "native_structure_png_verified_count": 9,
        "reverse_complement_verified_count": 9,
    }
    assert review_manifest["pwm_triptych"]["payload_trim_ids"] == [
        "TetR_full",
        "TetR_trim_conservative",
        "TetR_trim_aggressive",
    ]
    assert review_manifest["sequence_montage"]["frame_count"] == 9
    assert review_manifest["sequence_montage"]["still_count"] == 9
    assert review_manifest["source_indexes"]["sequence_index"] == "materialized/manifest/indexes/sequence_index.tsv"

    video_manifest = json.loads(result.sequence_montage_manifest.read_text(encoding="utf-8"))
    assert video_manifest["contract"] == "retron_hairpin_sequence_montage_manifest_v1"
    assert video_manifest["still_count"] == 9
    assert [frame["payload_trim_id"] for frame in video_manifest["frames"][:3]] == [
        "TetR_full",
        "TetR_trim_conservative",
        "TetR_trim_aggressive",
    ]
    assert video_manifest["frames"][0]["evidence_label"] == "control | retron26 | TetR_full"
    assert video_manifest["frames"][0]["review_still_png"] == "reviews/video/stills/01_control_retron26_TetR_full.png"
    assert "pES-retron-teto-trim-001" not in Path(video_manifest["frames"][0]["review_still_png"]).name
    assert video_manifest["frames"][0]["composition_overview_png"].endswith("composition_overview.png")
    assert (out_dir / video_manifest["frames"][0]["review_still_png"]).read_bytes().startswith(b"\x89PNG")


def test_teto_pwm_trim_review_outputs_fail_fast_on_wrong_row_count(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root, row_count=8)

    with pytest.raises(RetronMsdCompilerError, match="Expected 9 materialized sequence rows"):
        generate_teto_pwm_trim_rescue_review_outputs(
            deliverable_plan_path=study_dir / "workbench" / "deliverables" / "teto_pwm_trim_rescue_v1.yaml",
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )


def test_teto_pwm_trim_review_outputs_fail_fast_on_missing_row_artifact(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    rows = list(
        csv.DictReader(
            (materialized_root / "manifest" / "indexes" / "sequence_index.tsv")
            .read_text(encoding="utf-8")
            .splitlines(),
            delimiter="\t",
        )
    )
    (materialized_root / rows[0]["genbank"]).unlink()

    with pytest.raises(RetronMsdCompilerError, match="Missing materialized review artifact"):
        generate_teto_pwm_trim_rescue_review_outputs(
            deliverable_plan_path=study_dir / "workbench" / "deliverables" / "teto_pwm_trim_rescue_v1.yaml",
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )


def test_teto_pwm_trim_review_outputs_fail_fast_on_non_ok_folding(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    index_path = materialized_root / "manifest" / "indexes" / "sequence_index.tsv"
    rows = list(csv.DictReader(index_path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    rows[0]["folding_status"] = "backend_missing"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RetronMsdCompilerError, match="folding_status == ok"):
        generate_teto_pwm_trim_rescue_review_outputs(
            deliverable_plan_path=study_dir / "workbench" / "deliverables" / "teto_pwm_trim_rescue_v1.yaml",
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )


def test_teto_pwm_trim_review_outputs_fail_fast_on_bad_reverse_complement(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    materialized_root = write_fake_materialized_bundle(tmp_path / "materialized", repo_root=repo_root)
    rows = list(
        csv.DictReader(
            (materialized_root / "manifest" / "indexes" / "sequence_index.tsv")
            .read_text(encoding="utf-8")
            .splitlines(),
            delimiter="\t",
        )
    )
    (materialized_root / rows[0]["reverse_complement_fasta"]).write_text(
        ">bad_reverse_complement\nAAAAAAAAAA\n",
        encoding="utf-8",
    )

    with pytest.raises(RetronMsdCompilerError, match="reverse_complement_fasta does not match"):
        generate_teto_pwm_trim_rescue_review_outputs(
            deliverable_plan_path=study_dir / "workbench" / "deliverables" / "teto_pwm_trim_rescue_v1.yaml",
            materialized_root=materialized_root,
            out_dir=tmp_path / "outputs",
            repo_root=repo_root,
            video_writer=fake_video_writer,
        )


def test_review_outputs_cli_defaults_to_workbench_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    materialized_root = tmp_path / "materialized"

    def fake_generate(**kwargs: object):
        class Result:
            review_root = kwargs["out_dir"]
            pwm_triptych_svg = Path(kwargs["out_dir"]) / "reviews/pwm/teto_pwm_trim_rescue_v1.pwm_trim_triptych.svg"
            pwm_triptych_png = Path(kwargs["out_dir"]) / "reviews/pwm/teto_pwm_trim_rescue_v1.pwm_trim_triptych.png"
            sequence_montage_mp4 = (
                Path(kwargs["out_dir"]) / "reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.mp4"
            )
            sequence_montage_manifest = (
                Path(kwargs["out_dir"]) / "reviews/video/teto_pwm_trim_rescue_v1.sequence_montage.manifest.json"
            )
            review_manifest_path = Path(kwargs["out_dir"]) / "reviews/review_manifest.json"
            sequence_row_count = 9
            clone_handoff_verified_count = 9

        return Result()

    monkeypatch.setattr(cli_review_outputs_module, "generate_teto_pwm_trim_rescue_review_outputs", fake_generate)

    result = RUNNER.invoke(
        app,
        [
            "review-outputs",
            "--study-dir",
            study_dir.as_posix(),
            "--materialized-root",
            materialized_root.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    expected_root = study_dir / "workbench" / "outputs" / "teto_pwm_trim_rescue_v1"
    assert payload["status"] == "ok"
    assert payload["output_dir"] == expected_root.as_posix()
    assert payload["review_manifest_path"] == (expected_root / "reviews" / "review_manifest.json").as_posix()
    assert payload["record_count"] == 9
