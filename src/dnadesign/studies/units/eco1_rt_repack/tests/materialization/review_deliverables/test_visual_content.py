"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_visual_content.py

Eco1 review-deliverable visual content tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.chimerax_assertions import (
    assert_chimerax_context_scripts,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.notebook_assertions import (
    assert_review_notebook_contract,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    resolve_manifest_path,
)

from .proteinmpnn_visual_assertions import assert_proteinmpnn_visual_content


def test_review_deliverable_visual_content_is_plain_and_linked(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    _assert_mask_and_msa_content(result.manifest_path, deliverables)
    assert_proteinmpnn_visual_content(result.manifest_path, deliverables)
    _assert_linked_fold_and_esmc_content(result.manifest_path, deliverables)
    _assert_selection_content(deliverables)
    assert_chimerax_context_scripts(
        manifest_path=result.manifest_path,
        deliverables=deliverables,
        forbidden_path_text=str(tmp_path),
    )
    assert_review_notebook_contract(result.notebook_path.read_text(encoding="utf-8"))


def _assert_mask_and_msa_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    msa_text = _read_deliverable(manifest_path, deliverables, "msa_plurality_mask_panel")
    assert "The 4-record clade 9 MSA shows the active 25% WT-plurality mask denominator" in msa_text
    assert "all accepted clade 9 alignment rows" in str(deliverables["msa_plurality_mask_panel"]["description"])
    assert "current conservation mask uses this clade 9 denominator" in str(
        deliverables["msa_plurality_mask_panel"]["description"]
    )
    assert deliverables["msa_plurality_mask_panel"]["evidence_summary"]["current_mask_denominator"] is True
    assert "ec86_clade9_conservation_v1__" not in msa_text
    assert "WT plurality &gt;=25% (clade 9)" in msa_text
    assert "WT plurality &gt;=50% (design-class threshold)" in msa_text
    assert "Subtype II-A3/42_1 rows" in msa_text
    assert "C9 001 fig|fixture.1.peg.1" in msa_text
    assert "Mask-protected" in msa_text

    subtype_text = _read_deliverable(manifest_path, deliverables, "msa_subtype_plurality_panel")
    assert "The 3-record Eco1 subtype II-A3/42_1 MSA shows the narrower subtype conservation context" in subtype_text
    assert "all accepted II-A3/42_1 subtype alignment rows" in str(
        deliverables["msa_subtype_plurality_panel"]["description"]
    )
    assert "does not replace the clade 9 denominator" in str(deliverables["msa_subtype_plurality_panel"]["description"])
    assert deliverables["msa_subtype_plurality_panel"]["evidence_summary"]["current_mask_denominator"] is False
    assert "II-A3 002 fig|fixture.2.peg.1" in subtype_text
    assert "WT plurality &gt;=25% (Eco1 subtype II-A3/42_1)" in subtype_text
    assert "WT plurality &gt;=50% (design-class threshold)" in subtype_text

    mask_text = _read_deliverable(manifest_path, deliverables, "linear_mask_tracks")
    assert "Residue mask evidence across Ec86 RT" in mask_text
    assert "ProteinMPNN-designable residues" in mask_text
    assert "WT residue" not in mask_text
    assert "Ec86 positions 1-6" not in mask_text
    assert "Mask evidence track" not in mask_text
    assert "M" in mask_text
    assert "K" in mask_text

    design_class_mask_text = _read_deliverable(manifest_path, deliverables, "design_class_mask_overview")
    assert "Design-class mask rules show which residues were fixed or designable" in design_class_mask_text
    assert "clade 9 p25, 5 A" in design_class_mask_text
    assert "clade 9 p50, 5 A" in design_class_mask_text
    assert "subtype p50, 5 A" in design_class_mask_text
    assert "25% WT plurality" in design_class_mask_text
    assert "50% WT plurality" in design_class_mask_text
    assert "ProteinMPNN-designable residues" in design_class_mask_text
    assert "current baseline only" not in design_class_mask_text.lower()


def _assert_linked_fold_and_esmc_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    linked_fold_plot = resolve_manifest_path(
        manifest_path,
        str(deliverables["foldcheck_review_fold_metric_scatter"]["path"]),
    )
    assert linked_fold_plot.exists()
    assert linked_fold_plot.parent.name == "plots"
    linked_structure_overlay = resolve_manifest_path(
        manifest_path,
        str(deliverables["foldcheck_review_structure_overlay_panel"]["path"]),
    )
    assert linked_structure_overlay.exists()

    linked_esmc_plot = resolve_manifest_path(
        manifest_path,
        str(deliverables["wt_esmc_substitution_llr_heatmap"]["path"]),
    )
    assert linked_esmc_plot.exists()
    assert linked_esmc_plot.parent.name == "plots"
    linked_esmc_text = linked_esmc_plot.read_text(encoding="utf-8")
    assert "<title" in linked_esmc_text
    assert "<desc" in linked_esmc_text
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["title"] == (
        "ESMC masked-marginal scores form a WT substitution matrix"
    )
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["render_mode"] == "wide_visual"
    assert "LLR = log P(alternate) - log P(WT)" in str(
        deliverables["wt_esmc_substitution_llr_heatmap"]["method_summary"]
    )
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["evidence_summary"]["substitution_llr_rows"] == 114

    esmc_scatter_text = _read_deliverable(manifest_path, deliverables, "msa_plurality_vs_esmc_entropy")
    assert "Clade 9 plurality is inversely related to ESMC entropy" in esmc_scatter_text
    assert "Pearson r =" in esmc_scatter_text
    assert "R2 =" in esmc_scatter_text
    assert "Linear fit" in esmc_scatter_text
    assert "25% plurality threshold" not in esmc_scatter_text
    assert "model check of the WT sequence context" in str(
        deliverables["msa_plurality_vs_esmc_entropy"]["interpretation_limit"]
    )


def _assert_selection_content(deliverables: dict[str, dict[str, object]]) -> None:
    funnel = deliverables["selection_funnel_summary"]
    assert funnel["path"].endswith("design_classes/selection/selection_readiness_manifest.yaml")
    assert "row counts, gate counts, selected IDs, and selection policy" in str(funnel["description"])
    assert "ESMC and SAE are review annotations, not panel-selection evidence" in str(funnel["interpretation_limit"])
    assert "backend" not in str(funnel).lower()
    assert "generated" not in str(funnel).lower()

    readiness = deliverables["selection_handoff_readiness"]
    assert "candidate_handoff.yaml is absent" in str(readiness["description"])
    assert "construct subject" in str(readiness["description"])
    assert "no assay acceptance gate" in str(readiness["interpretation_limit"])
    assert "backend" not in str(readiness).lower()
    assert "generated" not in str(readiness).lower()


def _read_deliverable(manifest_path: Path, deliverables: dict[str, dict[str, object]], deliverable_id: str) -> str:
    path = resolve_manifest_path(manifest_path, str(deliverables[deliverable_id]["path"]))
    return path.read_text(encoding="utf-8")
