"""Tests for generic MSA visualization sidecars."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from dnadesign.aligner.msa.visualization import (
    MsaVisualizationRequest,
    materialize_msa_visualizations,
)
from dnadesign.aligner.tests.msa.visualization._fixtures import (
    TARGET,
    target_hash,
    write_alignment_inputs,
    write_annotation_tracks,
    write_exemplar_rows,
    write_panel_spec,
)


def test_visualization_writes_qc_sidecars_for_profiles(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a", "profile_b"))
    output_root = tmp_path / "visualizations"
    annotation_tracks = write_annotation_tracks(tmp_path)
    exemplar_rows = write_exemplar_rows(tmp_path)
    panel_spec = write_panel_spec(tmp_path)

    result = materialize_msa_visualizations(
        MsaVisualizationRequest(
            alignment_root=alignment_root,
            output_root=output_root,
            profile_ids=("profile_a", "profile_b"),
            target_row_id="target",
            target_sequence_hash=target_hash(TARGET),
            annotation_tracks_yaml=annotation_tracks,
            exemplar_rows_yaml=exemplar_rows,
            panel_spec_yaml=panel_spec,
        )
    )

    assert result.profile_ids == ("profile_a", "profile_b")
    assert result.missing_profile_ids == ()
    assert result.index_manifest_path == output_root / "msa_visualization_index.yaml"
    assert result.position_qc_csv_path == output_root / "msa_position_qc.csv"
    assert result.html_report_path == output_root / "msa_qc.html"
    assert result.profile_exemplar_svg_paths["profile_a"] == output_root / "profile_a.exemplar_windows.svg"
    assert result.profile_alignment_overview_svg_paths["profile_a"] == output_root / "profile_a.alignment_overview.svg"
    assert (
        result.profile_consensus_histogram_svg_paths["profile_a"] == output_root / "profile_a.consensus_histogram.svg"
    )

    index = yaml.safe_load(result.index_manifest_path.read_text(encoding="utf-8"))
    assert index["schema_id"] == "dnadesign.aligner.msa.visualization.index"
    assert index["status"] == "materialized"
    assert index["profile_ids"] == ["profile_a", "profile_b"]
    assert index["annotation_tracks_path"] == str(annotation_tracks)
    assert index["exemplar_rows_path"] == str(exemplar_rows)
    assert index["panel_spec_path"] == str(panel_spec)

    profile_qc = yaml.safe_load(result.profile_qc_paths["profile_a"].read_text(encoding="utf-8"))
    assert profile_qc["record_count"] == 3
    assert profile_qc["alignment_length"] == 21
    assert profile_qc["canonical_position_count"] == 20
    assert profile_qc["inserted_column_count"] == 1
    assert profile_qc["target_row_id"] == "target"

    with result.position_qc_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert rows[0]["profile_id"] == "profile_a"
    assert rows[0]["canonical_position"] == "1"
    svg = result.profile_svg_paths["profile_a"].read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "Motif A" in svg
    assert "Region B" in svg
    exemplar_svg = result.profile_exemplar_svg_paths["profile_a"].read_text(encoding="utf-8")
    assert "Reference target" in exemplar_svg
    assert "Homolog one" in exemplar_svg
    assert "Motif A (1-12)" in exemplar_svg
    assert 'data-feature-id="motif_a"' in exemplar_svg
    assert 'stroke="#d95f02"' in exemplar_svg
    assert ">A</text>" in exemplar_svg
    assert ">.</text>" in exemplar_svg
    overview_svg = result.profile_alignment_overview_svg_paths["profile_a"].read_text(encoding="utf-8")
    assert "selected-row overview" in overview_svg
    assert "Reference target" in overview_svg
    assert "Homolog one" in overview_svg
    assert "Motif A" in overview_svg
    assert 'stroke="#d95f02"' in overview_svg
    assert 'stroke-width="2.00"' in overview_svg
    histogram_svg = result.profile_consensus_histogram_svg_paths["profile_a"].read_text(encoding="utf-8")
    assert "plurality histogram" in histogram_svg
    assert "display-only high-gap trim threshold declared: 0.90" in histogram_svg
    assert "Visualization sidecar" in result.html_report_path.read_text(encoding="utf-8")


def test_visualization_rejects_missing_profiles_by_default(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_b",))

    with pytest.raises(FileNotFoundError, match="profile_a.aligned.fasta"):
        materialize_msa_visualizations(
            MsaVisualizationRequest(
                alignment_root=alignment_root,
                output_root=tmp_path / "visualizations",
                profile_ids=("profile_a", "profile_b"),
                target_row_id="target",
                target_sequence_hash=target_hash(TARGET),
            )
        )


def test_visualization_can_write_explicit_partial_report(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_b",))

    result = materialize_msa_visualizations(
        MsaVisualizationRequest(
            alignment_root=alignment_root,
            output_root=tmp_path / "visualizations",
            profile_ids=("profile_a", "profile_b"),
            target_row_id="target",
            target_sequence_hash=target_hash(TARGET),
            allow_missing_profiles=True,
        )
    )

    assert result.profile_ids == ("profile_b",)
    assert result.missing_profile_ids == ("profile_a",)
    index = yaml.safe_load(result.index_manifest_path.read_text(encoding="utf-8"))
    assert index["status"] == "materialized_partial"
    assert index["missing_profile_ids"] == ["profile_a"]


def test_visualization_rejects_target_hash_drift(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a",))

    with pytest.raises(ValueError, match="target row hash"):
        materialize_msa_visualizations(
            MsaVisualizationRequest(
                alignment_root=alignment_root,
                output_root=tmp_path / "visualizations",
                profile_ids=("profile_a",),
                target_row_id="target",
                target_sequence_hash="sha256:not-the-target",
            )
        )


def test_visualization_rejects_annotation_ranges_outside_target(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a",))
    annotation_tracks = write_annotation_tracks(tmp_path, feature_end=21)

    with pytest.raises(ValueError, match="outside target position range"):
        materialize_msa_visualizations(
            MsaVisualizationRequest(
                alignment_root=alignment_root,
                output_root=tmp_path / "visualizations",
                profile_ids=("profile_a",),
                target_row_id="target",
                target_sequence_hash=target_hash(TARGET),
                annotation_tracks_yaml=annotation_tracks,
            )
        )


def test_visualization_rejects_invalid_annotation_style(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a",))
    annotation_tracks = write_annotation_tracks(tmp_path, fill_opacity=1.5)

    with pytest.raises(ValueError, match="fill_opacity"):
        materialize_msa_visualizations(
            MsaVisualizationRequest(
                alignment_root=alignment_root,
                output_root=tmp_path / "visualizations",
                profile_ids=("profile_a",),
                target_row_id="target",
                target_sequence_hash=target_hash(TARGET),
                annotation_tracks_yaml=annotation_tracks,
            )
        )


def test_visualization_rejects_invalid_annotation_label_position(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a",))
    annotation_tracks = write_annotation_tracks(tmp_path, label_position="diagonal")

    with pytest.raises(ValueError, match="label_position"):
        materialize_msa_visualizations(
            MsaVisualizationRequest(
                alignment_root=alignment_root,
                output_root=tmp_path / "visualizations",
                profile_ids=("profile_a",),
                target_row_id="target",
                target_sequence_hash=target_hash(TARGET),
                annotation_tracks_yaml=annotation_tracks,
            )
        )


def test_visualization_respects_annotation_label_position(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a",))
    annotation_tracks = write_annotation_tracks(tmp_path, label_position="below")
    exemplar_rows = write_exemplar_rows(tmp_path)
    panel_spec = write_panel_spec(tmp_path)

    result = materialize_msa_visualizations(
        MsaVisualizationRequest(
            alignment_root=alignment_root,
            output_root=tmp_path / "visualizations",
            profile_ids=("profile_a",),
            target_row_id="target",
            target_sequence_hash=target_hash(TARGET),
            annotation_tracks_yaml=annotation_tracks,
            exemplar_rows_yaml=exemplar_rows,
            panel_spec_yaml=panel_spec,
        )
    )

    overview_svg = result.profile_alignment_overview_svg_paths["profile_a"].read_text(encoding="utf-8")
    assert "Motif A" in overview_svg
    assert 'data-label-position="below"' in overview_svg


def test_visualization_rejects_missing_exemplar_rows(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a",))
    annotation_tracks = write_annotation_tracks(tmp_path)
    exemplar_rows = write_exemplar_rows(tmp_path, missing=True)

    with pytest.raises(ValueError, match="exemplar row missing_record"):
        materialize_msa_visualizations(
            MsaVisualizationRequest(
                alignment_root=alignment_root,
                output_root=tmp_path / "visualizations",
                profile_ids=("profile_a",),
                target_row_id="target",
                target_sequence_hash=target_hash(TARGET),
                annotation_tracks_yaml=annotation_tracks,
                exemplar_rows_yaml=exemplar_rows,
            )
        )


def test_visualization_rejects_invalid_panel_spec_threshold(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a",))
    annotation_tracks = write_annotation_tracks(tmp_path)
    exemplar_rows = write_exemplar_rows(tmp_path)
    panel_spec = write_panel_spec(tmp_path, high_gap_trim_threshold=1.5)

    with pytest.raises(ValueError, match="high_gap_trim_threshold"):
        materialize_msa_visualizations(
            MsaVisualizationRequest(
                alignment_root=alignment_root,
                output_root=tmp_path / "visualizations",
                profile_ids=("profile_a",),
                target_row_id="target",
                target_sequence_hash=target_hash(TARGET),
                annotation_tracks_yaml=annotation_tracks,
                exemplar_rows_yaml=exemplar_rows,
                panel_spec_yaml=panel_spec,
            )
        )


def test_visualization_rejects_overview_without_exemplar_rows(tmp_path: Path) -> None:
    alignment_root = write_alignment_inputs(tmp_path, profile_ids=("profile_a",))
    panel_spec = write_panel_spec(tmp_path)

    with pytest.raises(ValueError, match="alignment overview requires exemplar rows"):
        materialize_msa_visualizations(
            MsaVisualizationRequest(
                alignment_root=alignment_root,
                output_root=tmp_path / "visualizations",
                profile_ids=("profile_a",),
                target_row_id="target",
                target_sequence_hash=target_hash(TARGET),
                panel_spec_yaml=panel_spec,
            )
        )
