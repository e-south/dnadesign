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

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
    SECTION_FEASIBILITY_AND_HANDOFF,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    write_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.materialization_expected import (
    EXPECTED_LINKED_MODEL_CHECK_DELIVERABLE_IDS,
    EXPECTED_RENDERED_DELIVERABLE_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.notebook_assertions import (
    assert_manifest_visual_contract,
)


def test_review_deliverables_materialize_manifest_figures_and_notebook(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    stale_linear_mask_track = tmp_path / "review_deliverables" / "mask_structure_context" / "linear_mask_tracks.svg"
    stale_linear_mask_track.parent.mkdir(parents=True, exist_ok=True)
    stale_linear_mask_track.write_text("<svg>retired baseline-only mask track</svg>\n", encoding="utf-8")
    stale_planned_handoff = tmp_path / "review_deliverables" / "feasibility_and_handoff" / "planned.md"
    stale_planned_handoff.parent.mkdir(parents=True, exist_ok=True)
    stale_planned_handoff.write_text(
        "# Feasibility and RT-only handoff remain planned\n\n"
        "The review bundle stops before synthesis feasibility and downstream handoff.\n",
        encoding="utf-8",
    )

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    assert not stale_linear_mask_track.exists()
    assert not stale_planned_handoff.exists()

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "eco1_rt.review_deliverables"
    assert manifest["status"] == "materialized_complete"
    assert manifest["deliverable_count"] == len(manifest["deliverables"])
    assert manifest["visual_policy"]["requires_alt_text"] is True
    assert manifest["notebook"]["path"] == "notebooks/eco1_review_deliverables.py"
    assert not Path(manifest["notebook"]["path"]).is_absolute()

    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    assert EXPECTED_RENDERED_DELIVERABLE_IDS.issubset(deliverables)
    assert "linear_mask_tracks" not in deliverables
    assert EXPECTED_LINKED_MODEL_CHECK_DELIVERABLE_IDS.issubset(deliverables)
    assert deliverables["mask_structure_context_png"]["status"] == "skipped_optional_render_disabled"
    assert "proteinmpnn_tao_style_fold_validation" not in deliverables
    assert "foldcheck_review_fold_metric_scatter" not in deliverables
    assert "foldcheck_review_cryoem_vs_runtime_rmsd" not in deliverables
    assert deliverables["foldcheck_review_structure_overlay_panel"]["status"] == "linked_existing"
    assert deliverables["foldcheck_review_structure_overlay_skipped"]["status"] == "skipped_runtime_unavailable"
    assert deliverables["mask_structure_browser_manifest"]["status"] == "rendered"
    assert deliverables["mask_structure_browser_manifest"]["title"] == "The Ec86 structure maps each fixed-mask rule"
    assert deliverables["interactive_structure_browser_manifest"]["status"] == "rendered"
    assert deliverables["interactive_structure_browser_manifest"]["title"] == (
        "Folded candidates can be inspected one at a time"
    )
    assert deliverables["selected_panel_structure_browser_manifest"]["status"] == "rendered"
    assert deliverables["selected_panel_structure_browser_manifest"]["title"] == (
        "Selected structures can be inspected one at a time"
    )
    assert deliverables["wt_esmc_entropy_by_position"]["status"] == "linked_existing"
    assert deliverables["wt_esmc_entropy_by_position"]["section"] == SECTION_CONSTRAINT_EVIDENCE
    assert deliverables["msa_plurality_vs_esmc_entropy"]["status"] == "rendered"
    assert deliverables["proteinmpnn_score_mutation_burden"]["section"] == SECTION_DESIGNS_AND_FOLD_TRIAGE
    assert deliverables["proteinmpnn_score_mutation_burden"]["role"] == "review_only"
    assert deliverables["proteinmpnn_residue_frequency_heatmap"]["role"] == "manuscript_facing"
    assert deliverables["proteinmpnn_residue_frequency_heatmap"]["artifact_kind"] == (
        "proteinmpnn_residue_frequency_bundle"
    )
    assert len(deliverables["proteinmpnn_residue_frequency_heatmap"]["evidence_summary"]["design_class_views"]) == 6
    assert "proteinmpnn_variant_similarity_heatmap" not in deliverables
    assert deliverables["design_class_mask_overview"]["section"] == SECTION_CONSTRAINT_EVIDENCE
    assert deliverables["design_class_mask_overview"]["role"] == "manuscript_facing"
    assert deliverables["design_class_mask_overview"]["render_mode"] == "compact_wide_visual"
    assert deliverables["expanded_proteinmpnn_fold_validation"]["section"] == SECTION_DESIGNS_AND_FOLD_TRIAGE
    assert deliverables["expanded_proteinmpnn_fold_validation"]["role"] == "manuscript_facing"
    assert deliverables["foldcheck_review_review_class_counts"]["status"] == "rendered"
    assert deliverables["foldcheck_review_review_class_counts"]["role"] == "manuscript_facing"
    assert deliverables["foldcheck_review_review_class_counts"]["title"] == (
        "Each fixed mask keeps foldable candidates"
    )
    assert deliverables["foldcheck_review_review_class_counts"]["source_tables"] == [
        "design_classes/candidate_pool.parquet",
        "design_classes/foldcheck_review/foldcheck_candidate_ranking.parquet",
    ]
    assert deliverables["foldcheck_review_review_class_counts"]["evidence_summary"]["design_class_count"] == 6
    assert deliverables["interactive_structure_browser_manifest"]["section"] == SECTION_DESIGNS_AND_FOLD_TRIAGE
    assert deliverables["selected_panel_structure_browser_manifest"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    for browser_id in (
        "mask_structure_browser_manifest",
        "interactive_structure_browser_manifest",
        "selected_panel_structure_browser_manifest",
        "biohub_esmc_sae_structure_browser_manifest",
    ):
        browser_payload = yaml.safe_load(
            (result.manifest_path.parent / deliverables[browser_id]["path"]).read_text(encoding="utf-8")
        )
        for field in ("title", "alt_text", "description", "interpretation_limit"):
            assert str(browser_payload.get(field) or "").strip()
    assert deliverables["selection_design_class_gate_counts"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    assert deliverables["selection_design_class_gate_counts"]["status"] == "linked_existing"
    assert deliverables["selection_primary_panel_sankey"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    assert deliverables["selection_primary_panel_sankey"]["status"] == "linked_existing"
    assert "selection_class_local_percentiles" not in deliverables
    assert deliverables["selection_six_sequence_distance"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    assert deliverables["selection_six_sequence_distance"]["status"] == "linked_existing"
    assert deliverables["selection_selected_substitutions_across_rt"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    assert deliverables["selection_selected_substitutions_across_rt"]["status"] == "linked_existing"
    assert deliverables["selection_regional_mutation_burden"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    assert deliverables["selection_regional_mutation_burden"]["status"] == "linked_existing"
    assert deliverables["selection_na_facing_chemistry_balance"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    assert deliverables["selection_na_facing_chemistry_balance"]["status"] == "linked_existing"
    assert deliverables["selection_funnel_summary"]["artifact_kind"] == "selection_funnel_summary"
    assert deliverables["selection_funnel_summary"]["status"] == "linked_existing"
    assert deliverables["selection_panel_table"]["artifact_kind"] == "selection_panel_table"
    assert deliverables["selection_panel_table"]["status"] == "linked_existing"
    assert deliverables["selection_handoff_sequences"]["artifact_kind"] == "candidate_handoff_sequence_csv"
    assert deliverables["selection_handoff_sequences"]["status"] == "linked_existing"
    assert deliverables["selection_handoff_readiness"]["artifact_kind"] == "handoff_readiness"
    assert deliverables["selection_handoff_readiness"]["status"] == "linked_existing"
    assert "selection_handoff_boundary" not in deliverables

    assert_manifest_visual_contract(
        manifest_path=result.manifest_path,
        manifest=manifest,
        deliverables=deliverables,
        expected_rendered=EXPECTED_RENDERED_DELIVERABLE_IDS,
    )


def test_review_deliverables_require_canonical_selection_manifest(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    (tmp_path / "design_classes" / "selection" / "selection_readiness_manifest.yaml").unlink()

    try:
        materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    except FileNotFoundError as exc:
        assert "design_classes/selection/selection_readiness_manifest.yaml" in str(exc)
    else:  # pragma: no cover - pytest assertion path
        raise AssertionError("review deliverables should require the panel-selection manifest")


def test_review_manifest_rejects_missing_visual_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required metadata field: alt_text"):
        write_manifest(
            tmp_path / "review_deliverables" / "review_deliverable_manifest.yaml",
            notebook_path=tmp_path / "review_deliverables" / "notebooks" / "eco1_review_deliverables.py",
            deliverables=[
                {
                    "deliverable_id": "metadata_incomplete_visual",
                    "title": "Metadata incomplete visual",
                    "section": "Panel selection",
                    "artifact_kind": "svg",
                    "status": "rendered",
                    "role": "manuscript_facing",
                    "render_mode": "wide_visual",
                    "path": str(tmp_path / "plot.svg"),
                    "source_tables": ["source.parquet"],
                    "input_hashes": {"source": "sha256:test"},
                    "alt_text": "",
                    "description": "A test row missing required alt text.",
                    "interpretation_limit": "This row is not valid.",
                }
            ],
        )
