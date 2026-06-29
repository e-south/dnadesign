"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_materialization.py

Eco1 review-deliverable materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.notebook_assertions import (
    assert_manifest_visual_contract,
    assert_review_notebook_contract,
)


def test_review_deliverables_materialize_manifest_figures_and_notebook(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "eco1_rt.review_deliverables"
    assert manifest["status"] == "materialized"
    assert manifest["deliverable_count"] == len(manifest["deliverables"])
    assert manifest["visual_policy"]["requires_alt_text"] is True
    assert manifest["notebook"]["path"] == "notebooks/eco1_review_deliverables.py"
    assert not Path(manifest["notebook"]["path"]).is_absolute()

    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    expected_rendered = {
        "msa_plurality_mask_panel",
        "linear_mask_tracks",
        "proteinmpnn_score_mutation_burden",
        "proteinmpnn_mutation_density",
        "proteinmpnn_tao_style_fold_validation",
        "mask_structure_context_script",
        "mask_structure_context_orientation_template",
        "msa_plurality_vs_esmc_entropy",
        "msa_plurality_vs_best_alt_llr",
        "msa_esmc_constraint_tracks",
    }
    expected_linked_model_constraint = {
        "wt_esmc_entropy_by_position",
        "wt_esmc_fraction_negative_alternate_llr",
        "wt_esmc_substitution_llr_heatmap",
    }
    assert expected_rendered.issubset(deliverables)
    assert expected_linked_model_constraint.issubset(deliverables)
    assert deliverables["mask_structure_context_png"]["status"] == "skipped_optional_render_disabled"
    assert deliverables["foldcheck_review_fold_metric_scatter"]["status"] == "linked_existing"
    assert deliverables["foldcheck_review_structure_overlay_panel"]["status"] == "linked_existing"
    assert deliverables["foldcheck_review_structure_overlay_skipped"]["status"] == "skipped_runtime_unavailable"
    assert deliverables["wt_esmc_entropy_by_position"]["status"] == "linked_existing"
    assert deliverables["msa_plurality_vs_esmc_entropy"]["status"] == "rendered"

    assert_manifest_visual_contract(
        manifest_path=result.manifest_path,
        manifest=manifest,
        deliverables=deliverables,
        expected_rendered=expected_rendered,
    )

    msa_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["msa_plurality_mask_panel"]["path"],
    ).read_text(encoding="utf-8")
    assert "The 4-record clade 9 MSA supplies the 25% plurality mask" in msa_text
    assert "display subset" in deliverables["msa_plurality_mask_panel"]["description"]
    assert "full clade 9 denominator" in deliverables["msa_plurality_mask_panel"]["description"]
    assert "ec86_clade9_conservation_v1__" not in msa_text
    assert "WT plurality &gt;=25% in full clade 9" in msa_text
    assert "C9 row 001" in msa_text

    mask_track_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["linear_mask_tracks"]["path"],
    ).read_text(encoding="utf-8")
    assert "Protected evidence defines fixed residues and the design canvas" in mask_track_text
    assert "WT residue" not in mask_track_text
    assert "Ec86 positions 1-6" in mask_track_text
    assert "Mask evidence track" in mask_track_text
    assert "M" in mask_track_text
    assert "K" in mask_track_text

    diversity_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["proteinmpnn_score_mutation_burden"]["path"],
    ).read_text(encoding="utf-8")
    assert "ProteinMPNN proposes sequence diversity inside the mutable canvas" in diversity_text
    assert "Sequence identity to Ec86 WT (%)" in diversity_text
    assert "Accepted designs retain a minority of WT residues." not in diversity_text
    assert "Sampling temperature" in diversity_text
    assert "Reported ProteinMPNN score" in diversity_text

    mutation_density_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["proteinmpnn_mutation_density"]["path"],
    ).read_text(encoding="utf-8")
    assert "RT1" in mutation_density_text
    assert "NAxxH" in mutation_density_text

    tao_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["proteinmpnn_tao_style_fold_validation"]["path"],
    ).read_text(encoding="utf-8")
    assert "ProteinMPNN designs cluster by ColabFold RMSD and pLDDT" in tao_text
    assert "WT-runtime C-alpha RMSD" in tao_text
    assert "Mean pLDDT" in tao_text
    assert "Tao-style" in deliverables["proteinmpnn_tao_style_fold_validation"]["description"]
    assert "single active mask policy" in deliverables["proteinmpnn_tao_style_fold_validation"]["interpretation_limit"]

    linked_fold_plot = _resolve_manifest_path(
        result.manifest_path,
        deliverables["foldcheck_review_fold_metric_scatter"]["path"],
    )
    assert linked_fold_plot.exists()
    assert linked_fold_plot.parent.name == "plots"
    linked_structure_overlay = _resolve_manifest_path(
        result.manifest_path,
        deliverables["foldcheck_review_structure_overlay_panel"]["path"],
    )
    assert linked_structure_overlay.exists()
    assert linked_structure_overlay.suffix == ".png"

    linked_esmc_plot = _resolve_manifest_path(
        result.manifest_path,
        deliverables["wt_esmc_substitution_llr_heatmap"]["path"],
    )
    assert linked_esmc_plot.exists()
    assert linked_esmc_plot.parent.name == "plots"
    linked_esmc_text = linked_esmc_plot.read_text(encoding="utf-8")
    assert "<title" in linked_esmc_text
    assert "<desc" in linked_esmc_text
    assert (
        deliverables["wt_esmc_substitution_llr_heatmap"]["title"]
        == "ESMC masked-marginal scores form a WT substitution matrix"
    )
    assert "LLR = log P(alternate) - log P(WT)" in deliverables["wt_esmc_substitution_llr_heatmap"]["method_summary"]
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["evidence_summary"]["substitution_llr_rows"] == 114

    esmc_scatter_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["msa_plurality_vs_esmc_entropy"]["path"],
    ).read_text(encoding="utf-8")
    assert "High clade 9 plurality usually corresponds to low ESMC entropy" in esmc_scatter_text
    assert "Pearson r =" in esmc_scatter_text
    assert "R2 =" in esmc_scatter_text
    assert "25% plurality threshold" in esmc_scatter_text
    assert "model-derived audit" in deliverables["msa_plurality_vs_esmc_entropy"]["interpretation_limit"]

    chimerax_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["mask_structure_context_script"]["path"],
    ).read_text(encoding="utf-8")
    assert "eco1_rt_clade9_plurality25_direct_contact5a_v1" in chimerax_text
    assert "set bgColor white" in chimerax_text
    assert "camera ortho" in chimerax_text
    assert '2dlabels text "Ec86 reference"' in chimerax_text
    assert "view orient" in chimerax_text
    assert "# orientation_preset_id: ec86_reference_thumb_down_v1" in chimerax_text
    assert "design canvas" in chimerax_text
    assert "color" in chimerax_text
    assert str(tmp_path) not in chimerax_text

    orientation_text = _resolve_manifest_path(
        result.manifest_path,
        deliverables["mask_structure_context_orientation_template"]["path"],
    ).read_text(encoding="utf-8")
    assert "Manual orientation handoff" in orientation_text
    assert "save mask_structure_context_orientation.cxs" in orientation_text
    assert "exit" not in orientation_text
    assert str(tmp_path) not in orientation_text

    assert_review_notebook_contract(result.notebook_path.read_text(encoding="utf-8"))


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
