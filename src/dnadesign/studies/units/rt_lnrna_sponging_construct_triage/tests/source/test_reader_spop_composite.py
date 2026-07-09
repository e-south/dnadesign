"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_spop_composite.py

Composite Reader SPOP condition matrix and retron structure plot checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.condition_matrix import (
    ReaderSpopConditionColumn,
    ReaderSpopConditionMatrix,
    ReaderSpopConditionRow,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.conditions import (
    BASELINE_CONDITION_KEY,
    BASELINE_ROLE,
    IPTG_DOSE_ROLE,
    POSITIVE_CONTROL_ROLE,
    condition_key_for_iptg_dose,
    condition_key_for_positive_control,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.render import (
    CompositeRenderError,
    _deviating_structure_text_indices,
    _structure_vector_data_size,
    render_spop_condition_structure_plot,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.row_categories import (
    category_for_assay_subject,
    category_spans_for_variants,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.structure_manifest import (
    RetronStructureManifestError,
    RetronStructureThumbnailRow,
    build_retron_structure_thumbnail_manifest,
    write_retron_structure_thumbnail_manifest,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.structure_svg import (
    oriented_structure_geometry,
)

from .hairpin_structure_fixtures import write_hairpin_structure_fixture


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_retron_structure_thumbnail_manifest_resolves_hairpin_195_200_assets(tmp_path: Path) -> None:
    repo_root, hairpin_output_dir = write_hairpin_structure_fixture(tmp_path)
    rows = build_retron_structure_thumbnail_manifest(
        repo_root=repo_root,
        hairpin_output_dir=hairpin_output_dir,
        assay_subject_keys=(
            "retron26",
            "retron43",
            "retron195",
            "retron196",
            "retron197",
            "retron198",
            "retron199",
            "retron200",
        ),
    )
    by_key = {row.assay_subject_key: row for row in rows}

    assert by_key["retron195"].display_variant_id == "pES-retron-195"
    assert by_key["retron195"].source_precedent_id == "pES-retron-195"
    assert by_key["retron195"].folding_status == "ok"
    assert by_key["retron195"].structure_png_path.endswith("plots/secondary_structure.native.png")
    assert by_key["retron195"].structure_svg_path.endswith("manifest/visual/secondary_structure/native.svg")
    assert by_key["retron195"].structure_annotation_manifest_path.endswith(
        "manifest/visual/secondary_structure/annotation_manifest.json"
    )
    assert (repo_root / by_key["retron195"].structure_png_path).exists()
    assert (repo_root / by_key["retron195"].structure_svg_path).exists()
    assert (repo_root / by_key["retron195"].structure_annotation_manifest_path).exists()
    assert by_key["retron195"].left_base_sequence == "CGGG"
    assert by_key["retron195"].right_base_sequence == "ACAG"
    assert by_key["retron195"].stem_length_bp == 15
    assert by_key["retron195"].foldback_sequence == "AGGC"
    assert by_key["retron43"].stem_length_bp == 26
    assert by_key["retron200"].source_precedent_id == "pES-retron-200"
    assert by_key["retron200"].stem_length_bp == 16
    assert by_key["retron200"].primitive_source_path.endswith("variants/pes-retron-200-msd-region.yaml")
    assert by_key["retron197"].stem_length_bp == 22
    assert by_key["retron197"].primitive_warning == ""
    assert by_key["retron197"].payload_pairing_status == "canonical_wc"
    assert by_key["retron197"].foldback_pairing_status == "canonical_wc"
    assert by_key["retron198"].primitive_warning == ""
    assert by_key["retron200"].sequence_sha256


def test_retron_structure_thumbnail_manifest_writes_parquet(tmp_path: Path) -> None:
    repo_root, hairpin_output_dir = write_hairpin_structure_fixture(tmp_path / "fixture")
    rows = build_retron_structure_thumbnail_manifest(
        repo_root=repo_root,
        hairpin_output_dir=hairpin_output_dir,
        assay_subject_keys=("retron195",),
    )

    path = write_retron_structure_thumbnail_manifest(rows, output_dir=tmp_path)

    written = pq.read_table(path).to_pylist()
    assert written[0]["assay_subject_key"] == "retron195"
    assert written[0]["structure_status"] == "available"
    assert written[0]["left_base_sequence"] == "CGGG"
    assert written[0]["stem_length_bp"] == 15
    assert written[0]["foldback_sequence"] == "AGGC"
    assert written[0]["payload_pairing_status"] == "canonical_wc"


def test_structure_svg_geometry_returns_horizontal_vector_elements(tmp_path: Path) -> None:
    repo_root, hairpin_output_dir = write_hairpin_structure_fixture(tmp_path)
    rows = build_retron_structure_thumbnail_manifest(
        repo_root=repo_root,
        hairpin_output_dir=hairpin_output_dir,
        assay_subject_keys=("retron195",),
    )
    geometry = oriented_structure_geometry(
        (repo_root / rows[0].structure_svg_path).as_posix(),
        annotation_manifest_path=(repo_root / rows[0].structure_annotation_manifest_path).as_posix(),
    )
    min_x, max_x, min_y, max_y = geometry.bounds

    assert len(geometry.lines) > 10
    assert len(geometry.texts) > 40
    assert max_x - min_x > max_y - min_y
    semantic_colors = {line.semantic: line.color for line in geometry.lines if line.semantic}
    line_kinds = {line.kind for line in geometry.lines}
    assert {"backbone", "basepair"} <= line_kinds
    assert semantic_colors["stem_base_left"] != semantic_colors["flank_5p"]
    assert semantic_colors["stem_base_right"] != semantic_colors["flank_3p"]
    assert "payload_primary" in semantic_colors
    assert "payload_complement" in semantic_colors
    semantic_texts = {text.semantic: text.color for text in geometry.texts if text.semantic}
    assert semantic_texts["payload_primary"] == semantic_colors["payload_primary"]


def test_structure_vector_data_size_preserves_svg_display_aspect() -> None:
    width_data, height_data = _structure_vector_data_size(
        source_width=400.0,
        source_height=100.0,
        x_pixels_per_data=1000.0,
        y_pixels_per_data=100.0,
        max_width_data=0.96,
        max_height_data=0.84,
    )

    assert width_data <= 0.96
    assert height_data <= 0.84
    assert (width_data * 1000.0) / (height_data * 100.0) == pytest.approx(4.0)


def test_structure_deviation_indices_align_against_retron26_reference() -> None:
    assert _deviating_structure_text_indices(reference_sequence="ACGT", variant_sequence="ACGA") == (3,)
    assert _deviating_structure_text_indices(reference_sequence="ACGT", variant_sequence="ACGTT") == (4,)


def test_reader_spop_row_categories_map_variant_groups() -> None:
    assert category_for_assay_subject("retron26").label == "GUU reference"
    assert category_for_assay_subject("retron45").label == "Stem-base context"
    assert category_for_assay_subject("retron47").label == "Sso7d-RT fusions"
    assert category_for_assay_subject("retron49").label == "Evo2 RT mutants"
    assert category_for_assay_subject("retron172").label == "Foldback cores"
    assert category_for_assay_subject("retron177").label == "Stem/cap wobbles"
    assert category_for_assay_subject("retron195").label == "tetO truncations"

    spans = category_spans_for_variants(("retron26", "retron45", "retron46", "retron47"))

    assert [(span.label, span.start_index, span.stop_index) for span in spans] == [
        ("GUU reference", 0, 1),
        ("Stem-base context", 1, 3),
        ("Sso7d-RT fusions", 3, 4),
    ]


def test_retron_structure_thumbnail_manifest_reports_missing_handoff_file(tmp_path: Path) -> None:
    hairpin_root = tmp_path / "hairpin"
    reviews_dir = hairpin_root / "reviews"
    reviews_dir.mkdir(parents=True)
    (reviews_dir / "review_manifest.json").write_text(
        json.dumps({"sequence_montage": {"review_variant_ids": {"trim_195": "pES-retron-195"}}}),
        encoding="utf-8",
    )

    with pytest.raises(RetronStructureManifestError, match="handoff"):
        build_retron_structure_thumbnail_manifest(
            repo_root=tmp_path,
            assay_subject_keys=("retron195",),
            hairpin_output_dir=Path("hairpin"),
        )


def test_reader_spop_composite_smoke_renders_missing_condition_cells(tmp_path: Path) -> None:
    matrix = ReaderSpopConditionMatrix(
        rows=(
            _row(
                "retron26",
                BASELINE_CONDITION_KEY,
                BASELINE_ROLE,
                0.0,
                0.0,
                100.0,
                viability_relative_to_baseline=1.0,
            ),
            _row(
                "retron26",
                condition_key_for_positive_control(20.0),
                POSITIVE_CONTROL_ROLE,
                20.0,
                0.0,
                500.0,
                viability_relative_to_baseline=0.88,
            ),
            _row(
                "retron26",
                condition_key_for_iptg_dose(500.0),
                IPTG_DOSE_ROLE,
                0.0,
                500.0,
                460.0,
                viability_relative_to_baseline=0.92,
            ),
            _row(
                "retron195",
                BASELINE_CONDITION_KEY,
                BASELINE_ROLE,
                0.0,
                0.0,
                100.0,
                viability_relative_to_baseline=1.0,
            ),
            _row(
                "retron195",
                condition_key_for_positive_control(20.0),
                POSITIVE_CONTROL_ROLE,
                20.0,
                0.0,
                500.0,
                viability_relative_to_baseline=0.91,
            ),
        ),
        condition_columns=(
            ReaderSpopConditionColumn(
                condition_key=BASELINE_CONDITION_KEY,
                condition_role=BASELINE_ROLE,
                atc_nM=0.0,
                iptg_uM=0.0,
            ),
            ReaderSpopConditionColumn(
                condition_key=condition_key_for_positive_control(20.0),
                condition_role=POSITIVE_CONTROL_ROLE,
                atc_nM=20.0,
                iptg_uM=0.0,
            ),
            ReaderSpopConditionColumn(
                condition_key=condition_key_for_iptg_dose(500.0),
                condition_role=IPTG_DOSE_ROLE,
                atc_nM=0.0,
                iptg_uM=500.0,
            ),
        ),
        missing_cell_count=1,
        source_reader_experiment_ids=("demo_reader_experiment",),
    )
    repo_root, hairpin_output_dir = write_hairpin_structure_fixture(tmp_path / "fixture")
    thumbnail_rows = build_retron_structure_thumbnail_manifest(
        repo_root=repo_root,
        hairpin_output_dir=hairpin_output_dir,
        assay_subject_keys=("retron26", "retron195"),
    )

    manifest = render_spop_condition_structure_plot(
        condition_matrix=matrix,
        thumbnail_rows=thumbnail_rows,
        output_dir=tmp_path,
        repo_root=repo_root,
    )

    assert Path(manifest.plot_png_path).exists()
    assert Path(manifest.plot_svg_path).exists()
    assert Path(manifest.plot_png_path).stat().st_size > 1000
    svg_text = Path(manifest.plot_svg_path).read_text(encoding="utf-8")
    assert svg_text.count("data:image/png;base64") <= 4
    assert manifest.variant_count == 2
    assert manifest.condition_count == 3
    assert manifest.missing_cell_count == 1
    payload = json.loads(Path(manifest.manifest_path).read_text(encoding="utf-8"))
    assert payload["contract"] == "rt_lnrna_spop_condition_structure_plot_manifest_v1"
    assert payload["plot_premise"] == "Retron edits shift activation and growth"
    assert not payload["plot_premise"].endswith(".")
    assert payload["panel_order"] == [
        "Experiment group",
        "OD600 rel.",
        "RFP/OD600 activation",
        "MSD primitives",
        "MSD structure",
    ]
    assert payload["row_category_band_position"] == "left_of_heatmaps"
    assert payload["row_category_band_shape"] == "rounded_rectangles"
    assert payload["row_category_count"] == 2
    assert payload["row_category_spans"][0]["label"] == "GUU reference"
    assert payload["row_category_spans"][0]["assay_subject_keys"] == ["retron26"]
    assert payload["row_category_spans"][1]["label"] == "tetO truncations"
    assert payload["row_category_spans"][1]["assay_subject_keys"] == ["retron195"]
    assert payload["missing_cell_rendering"] == "white_not_zero"
    assert payload["missing_tile_color"] == "#ffffff"
    assert payload["heatmap_tile_aspect"] == "square"
    assert payload["condition_tick_label_style"] == "compact_aTc_IPTG"
    assert payload["condition_tick_label_presence"] == "both_heatmaps"
    assert payload["value_palette"] == "pastel_cold_to_warm_activation"
    assert payload["zero_value_color"] != payload["missing_tile_color"]
    assert payload["x_axis_label"] == ""
    assert payload["y_axis_label"] == "lnRNA variants in retron Eco1 system"
    assert payload["structure_thumbnail_orientation"] == "rightward_horizontal_cap_right"
    assert payload["structure_nucleotide_text_orientation"] == "upright"
    assert payload["structure_thumbnail_horizontal_flip"] is True
    assert payload["structure_thumbnail_frame"] == "none"
    assert payload["structure_thumbnail_crop"] == "trim_white_margin"
    assert payload["structure_thumbnail_crop_margin_px"] <= 2
    assert payload["structure_thumbnail_interpolation"] == "lanczos"
    assert 0.11 <= payload["structure_thumbnail_zoom"] <= 0.13
    assert payload["structure_rendering_mode"] == "matplotlib_vector_primitives_from_viennarna_svg"
    assert payload["structure_vector_aspect_policy"] == "preserve_native_svg_aspect_ratio"
    assert payload["structure_deviation_reference_variant"] == "retron26"
    assert payload["structure_deviation_highlight_mode"] == "variant_text_indices_with_primitive_hue_fill"
    assert payload["structure_deviation_legend_label"] == "Differs from retron 26"
    assert payload["structure_deviation_legend_label_align"] == "center"
    assert payload["structure_deviation_marker_size"] >= 8.0
    assert 0.3 <= payload["structure_deviation_marker_alpha"] <= 0.5
    assert payload["structure_deviation_marker_edge_alpha"] >= 0.85
    assert payload["structure_deviation_legend_fontsize"] == pytest.approx(7.0)
    assert payload["structure_deviation_legend_items"][0] == ["stem_base_left", "Stem"]
    assert 0.27 <= payload["structure_panel_content_center_x"] <= 0.35
    assert payload["structure_panel_separator_xmax"] <= 0.72
    assert payload["structure_deviation_legend_label_x"] == pytest.approx(payload["structure_panel_content_center_x"])
    assert payload["structure_deviation_legend_fontsize"] >= 6.6
    assert payload["structure_deviation_legend_item_positions"][0] <= 0.08
    assert payload["structure_deviation_legend_item_positions"][-1] <= 0.62
    assert payload["structure_vector_text_fontsize"] >= 2.6
    assert payload["structure_vector_line_width_mode"] == "semantic_backbone_with_quiet_inset_basepair_edges"
    assert payload["structure_vector_basepair_line_width_scale"] < payload["structure_vector_backbone_line_width_scale"]
    assert 0.08 <= payload["structure_vector_basepair_endpoint_inset"] <= 0.14
    assert payload["typography_profile"] == "publication_dense_v1"
    assert payload["plot_title_fontsize"] >= 14.0
    assert payload["panel_title_fontsize"] >= 10.5
    assert payload["variant_label_fontsize"] >= 9.0
    assert payload["primitive_text_fontsize"] >= 6.8
    assert payload["plot_dpi"] >= 450
    assert payload["colorbar_orientation"] == "horizontal_bottom"
    assert payload["label_collision_policy"] == "tight_cbar_row_close_to_compact_condition_ticks"
    assert payload["colorbar_height_ratio"] <= 0.006
    assert payload["layout_density"] == "compact_adjacent_panels"
    assert payload["primitive_column_order"] == [
        "left_base_sequence",
        "stem_length_bp",
        "foldback_sequence",
        "right_base_sequence",
    ]
    assert (
        payload["primitive_column_source"] == "retron_hairpin_materialized_features_and_decomposed_msd_region_records"
    )
    assert payload["primitive_stem_length_basis"] == "payload_primary_interval_plus_snapback_foldback_return_bp"
    assert payload["od600_panel_label"] == "OD600 rel."
    assert payload["od600_panel_basis"] == "condition_aligned_viability_relative_to_baseline"
    assert payload["od600_panel_palette"] == "pastel_cold_to_warm_growth"
    assert payload["od600_panel_condition_count"] == 3
    assert payload["color_scale"] == {"vmin": 0.0, "vmax": 1.0, "clip": True}
    assert payload["normalization_scope"] == "within_reader_observation_not_cross_experiment_absolute"
    assert "baseline=0" in payload["normalization_basis"]
    assert "aTc positive control=1" in payload["normalization_basis"]
    assert payload["missing_structure_summary"] == {
        "available": 2,
        "missing": 0,
        "by_status": {"available": 2},
        "missing_assay_subject_keys": [],
        "explanation": (
            "Rows marked missing are absent from the configured retron-hairpin "
            "materialized structure source, not silently plotted as zero."
        ),
    }


def test_reader_spop_composite_rejects_stale_available_thumbnail(tmp_path: Path) -> None:
    matrix = ReaderSpopConditionMatrix(
        rows=(_row("retron195", BASELINE_CONDITION_KEY, BASELINE_ROLE, 0.0, 0.0, 100.0),),
        condition_columns=(
            ReaderSpopConditionColumn(
                condition_key=BASELINE_CONDITION_KEY,
                condition_role=BASELINE_ROLE,
                atc_nM=0.0,
                iptg_uM=0.0,
            ),
        ),
        missing_cell_count=0,
        source_reader_experiment_ids=("demo_reader_experiment",),
    )

    with pytest.raises(CompositeRenderError, match="thumbnail is missing"):
        render_spop_condition_structure_plot(
            condition_matrix=matrix,
            thumbnail_rows=(
                RetronStructureThumbnailRow(
                    assay_subject_key="retron195",
                    display_variant_id="pES-retron-195",
                    hairpin_variant_id="trim_195",
                    construct_id="construct_195",
                    source_precedent_id="pES-retron-26",
                    sequence_sha256="sha256:fixture",
                    sequence_length_nt=100,
                    folding_status="ok",
                    structure_status="available",
                    structure_png_path="missing/secondary_structure.native.png",
                    composition_png_path="",
                    source_bundle_path="",
                    review_manifest_path="",
                ),
            ),
            output_dir=tmp_path,
            repo_root=tmp_path,
        )


def _row(
    assay_subject_key: str,
    condition_key: str,
    condition_role: str,
    atc_nM: float,
    iptg_uM: float,
    rfp_over_od600: float,
    *,
    viability_relative_to_baseline: float | None = None,
) -> ReaderSpopConditionRow:
    normalized = 0.0 if condition_role == BASELINE_ROLE else 1.0 if condition_role == POSITIVE_CONTROL_ROLE else 0.9
    return ReaderSpopConditionRow(
        observation_id=f"obs::{assay_subject_key}",
        assay_subject_key=assay_subject_key,
        reader_design_id=f"pES-{assay_subject_key}; pBbS2c-rfp",
        reader_experiment_id="demo_reader_experiment",
        condition_key=condition_key,
        condition_role=condition_role,
        atc_nM=atc_nM,
        iptg_uM=iptg_uM,
        normalized_derepression=normalized,
        rfp_over_od600=rfp_over_od600,
        viability_relative_to_baseline=viability_relative_to_baseline,
        replicate_count=3,
        construct_subject_id=None,
        construct_subject_bridge_status="missing_construct_sequence_authority",
        qc_flags=(),
        value_basis="test_fixture",
    )
