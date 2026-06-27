"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/foldcheck_review/test_materialization.py

Eco1 fold-check review materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review import (
    materialize_foldcheck_review,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_review.fixtures import (
    write_review_inputs,
)


def test_foldcheck_review_materializes_ranking_panel_and_atlas_subset(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=True)

    result = materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)

    ranking_rows = pq.read_table(result.ranking_path).to_pylist()
    assert ranking_rows[0]["candidate_id"] == "thread_candidate_best_rmsd"
    assert ranking_rows[0]["wt_runtime_ca_rmsd"] == 0.8
    assert ranking_rows[0]["cryoem_mapped_ca_rmsd_status"] == "available"
    assert "strong_fold_preserved" in {row["review_class"] for row in ranking_rows}
    assert "structural_outlier" in {row["review_class"] for row in ranking_rows}
    assert "low_confidence" in {row["review_class"] for row in ranking_rows}

    panel = yaml.safe_load(result.structure_panel_path.read_text(encoding="utf-8"))
    assert panel["schema_id"] == "eco1_rt.foldcheck_structure_panel"
    assert panel["path_policy"] == "local_paths_manifest_relative"
    selected = {entry["candidate_id"]: entry for entry in panel["selected_structures"]}
    assert selected["wild_type"]["copy_status"] == "copied"
    assert selected["thread_candidate_worst_rmsd"]["selection_stratum"] == "rmsd_outlier"
    assert selected["thread_candidate_low_plddt"]["selection_stratum"] == "low_plddt"
    assert _resolve_manifest_path(
        result.structure_panel_path,
        selected["thread_candidate_best_rmsd"]["local_model_artifact_path"],
    ).is_file()

    chimerax_script = result.chimerax_script_path.read_text(encoding="utf-8")
    assert "matchmaker" in chimerax_script
    assert "thread_candidate_worst_rmsd" in chimerax_script
    assert str(tmp_path) not in chimerax_script

    atlas_subset = yaml.safe_load(result.atlas_subset_manifest_path.read_text(encoding="utf-8"))
    assert atlas_subset["schema_id"] == "eco1_rt.atlas_subset_manifest"
    assert atlas_subset["atlas_policy"]["candidate_acceptance_gate"] is False
    assert atlas_subset["atlas_policy"]["default_fold_on_miss"] is False
    assert "thread_candidate_best_plddt" in atlas_subset["selected_sequence_ids"]

    full_set = yaml.safe_load(result.full_structure_set_path.read_text(encoding="utf-8"))
    assert full_set["schema_id"] == "eco1_rt.foldcheck_full_structure_set"
    assert full_set["path_policy"] == "local_paths_manifest_relative"
    assert full_set["structure_count"] == 7
    assert full_set["copy_summary"] == {"copied": 7}
    all_structures = {entry["candidate_id"]: entry for entry in full_set["structures"]}
    assert all_structures["thread_candidate_best_rmsd"]["copy_status"] == "copied"
    assert _resolve_manifest_path(
        result.full_structure_set_path,
        all_structures["thread_candidate_best_rmsd"]["local_model_artifact_path"],
    ).is_file()

    full_chimerax_script = result.full_chimerax_script_path.read_text(encoding="utf-8")
    assert "thread_candidate_best_rmsd_full_fold_set" in full_chimerax_script
    assert "thread_candidate_intermediate_full_fold_set" in full_chimerax_script
    assert str(tmp_path) not in full_chimerax_script

    visual_manifest = yaml.safe_load(result.visual_manifest_path.read_text(encoding="utf-8"))
    assert visual_manifest["schema_id"] == "eco1_rt.foldcheck_review_visual_manifest"
    assert visual_manifest["status"] == "materialized"
    assert visual_manifest["notebook"]["path"] == "notebooks/eco1_foldcheck_review.py"
    assert not Path(visual_manifest["notebook"]["path"]).is_absolute()
    assert visual_manifest["notebook"]["scope"] == "eco1_rt_repack fold-check review"
    assert len(visual_manifest["plots"]) >= 3
    plot_ids = {plot["plot_id"] for plot in visual_manifest["plots"]}
    assert "cryoem_vs_runtime_rmsd" in plot_ids
    assert "review_class_counts" in plot_ids
    for plot in visual_manifest["plots"]:
        assert not Path(plot["path"]).is_absolute()
        assert plot["alt_text"].strip()
        assert plot["description"].strip()
        assert plot["interpretation_limit"].strip()
        plot_path = _resolve_manifest_path(result.visual_manifest_path, plot["path"])
        svg_text = plot_path.read_text(encoding="utf-8")
        svg_root = ET.parse(plot_path).getroot()
        assert "<title" in svg_text
        assert "<desc" in svg_text
        assert svg_root.findall(".//{http://www.w3.org/2000/svg}text")

    notebook_text = result.notebook_path.read_text(encoding="utf-8")
    assert "marimo.App" in notebook_text
    assert "review_visual_manifest.yaml" in notebook_text
    assert "manifest_root = manifest_path.parent" in notebook_text
    assert "def resolve_manifest_path(" in notebook_text
    assert "_resolve_manifest_path(" not in notebook_text
    assert "alt_text" in notebook_text
    assert 'label="Review surface"' in notebook_text
    assert "visual_surface_ui = mo.ui.dropdown" in notebook_text
    assert "selected_plot = plot_lookup.get(" in notebook_text
    assert "mo.accordion(" in notebook_text
    assert "What this visual shows" in notebook_text
    assert "Interpretation limit" in notebook_text
    assert "Plot inventory" in notebook_text
    assert "\n    visual_surface_ui\n" not in notebook_text
    for cell in notebook_text.split("@app.cell"):
        if "visual_surface_ui = mo.ui.dropdown(" in cell:
            assert ".value" not in cell


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def test_foldcheck_review_marks_cryoem_rmsd_unavailable_for_external_models(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=False)

    result = materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)

    ranking_rows = pq.read_table(result.ranking_path).to_pylist()
    candidate_row = next(row for row in ranking_rows if row["candidate_id"] == "thread_candidate_best_rmsd")
    assert candidate_row["cryoem_mapped_ca_rmsd"] is None
    assert candidate_row["cryoem_mapped_ca_rmsd_status"] == "model_artifact_not_local"

    panel = yaml.safe_load(result.structure_panel_path.read_text(encoding="utf-8"))
    selected = {entry["candidate_id"]: entry for entry in panel["selected_structures"]}
    assert selected["thread_candidate_best_rmsd"]["copy_status"] == "source_not_local"
    assert "source_model_artifact_path" in selected["thread_candidate_best_rmsd"]

    full_set = yaml.safe_load(result.full_structure_set_path.read_text(encoding="utf-8"))
    assert full_set["copy_summary"] == {"source_not_local": 7}
