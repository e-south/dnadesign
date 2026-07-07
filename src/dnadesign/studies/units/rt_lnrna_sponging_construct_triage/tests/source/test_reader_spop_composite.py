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
    render_spop_condition_structure_plot,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.structure_manifest import (
    RetronStructureManifestError,
    RetronStructureThumbnailRow,
    build_retron_structure_thumbnail_manifest,
    write_retron_structure_thumbnail_manifest,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_retron_structure_thumbnail_manifest_resolves_hairpin_195_200_assets() -> None:
    repo_root = _repo_root()
    rows = build_retron_structure_thumbnail_manifest(
        repo_root=repo_root,
        assay_subject_keys=("retron26", "retron195", "retron196", "retron197", "retron198", "retron199", "retron200"),
    )
    by_key = {row.assay_subject_key: row for row in rows}

    assert by_key["retron195"].display_variant_id == "pES-retron-195"
    assert by_key["retron195"].source_precedent_id == "pES-retron-26"
    assert by_key["retron195"].folding_status == "ok"
    assert by_key["retron195"].structure_png_path.endswith("plots/secondary_structure.native.png")
    assert (repo_root / by_key["retron195"].structure_png_path).exists()
    assert by_key["retron200"].source_precedent_id == "pES-retron-180"
    assert by_key["retron200"].sequence_sha256


def test_retron_structure_thumbnail_manifest_writes_parquet(tmp_path: Path) -> None:
    repo_root = _repo_root()
    rows = build_retron_structure_thumbnail_manifest(repo_root=repo_root, assay_subject_keys=("retron195",))

    path = write_retron_structure_thumbnail_manifest(rows, output_dir=tmp_path)

    written = pq.read_table(path).to_pylist()
    assert written[0]["assay_subject_key"] == "retron195"
    assert written[0]["structure_status"] == "available"


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
            _row("retron26", BASELINE_CONDITION_KEY, BASELINE_ROLE, 0.0, 0.0, 100.0),
            _row("retron26", condition_key_for_positive_control(20.0), POSITIVE_CONTROL_ROLE, 20.0, 0.0, 500.0),
            _row("retron26", condition_key_for_iptg_dose(500.0), IPTG_DOSE_ROLE, 0.0, 500.0, 460.0),
            _row("retron195", BASELINE_CONDITION_KEY, BASELINE_ROLE, 0.0, 0.0, 100.0),
            _row("retron195", condition_key_for_positive_control(20.0), POSITIVE_CONTROL_ROLE, 20.0, 0.0, 500.0),
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
    thumbnail_rows = build_retron_structure_thumbnail_manifest(
        repo_root=_repo_root(),
        assay_subject_keys=("retron26", "retron195"),
    )

    manifest = render_spop_condition_structure_plot(
        condition_matrix=matrix,
        thumbnail_rows=thumbnail_rows,
        output_dir=tmp_path,
    )

    assert Path(manifest.plot_png_path).exists()
    assert Path(manifest.plot_svg_path).exists()
    assert Path(manifest.plot_png_path).stat().st_size > 1000
    assert manifest.variant_count == 2
    assert manifest.condition_count == 3
    assert manifest.missing_cell_count == 1
    payload = json.loads(Path(manifest.manifest_path).read_text(encoding="utf-8"))
    assert payload["contract"] == "rt_lnrna_spop_condition_structure_plot_manifest_v1"
    assert payload["missing_cell_rendering"] == "masked_gray_not_zero"
    assert payload["heatmap_tile_aspect"] == "square"
    assert payload["value_palette"] == "white_to_darker_seagreen"
    assert payload["x_axis_label"] == ""
    assert payload["y_axis_label"] == "lnRNA variants in retron Eco1 system"
    assert payload["structure_thumbnail_orientation"] == "rightward_horizontal_cap_right"
    assert payload["structure_thumbnail_frame"] == "none"
    assert payload["structure_thumbnail_crop"] == "trim_white_margin"
    assert payload["structure_thumbnail_zoom"] >= 0.12
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
            "materialized review manifest, not silently plotted as zero."
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
        viability_relative_to_baseline=None,
        replicate_count=3,
        construct_subject_id=None,
        construct_subject_bridge_status="missing_construct_sequence_authority",
        qc_flags=(),
        value_basis="test_fixture",
    )
