"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/proteinmpnn_visual_assertions.py

ProteinMPNN visual-content assertions for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def assert_proteinmpnn_visual_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    """Assert v3 proposal visuals are present without restoring v1 design-class views."""

    assert "proteinmpnn_score_mutation_burden" not in deliverables
    assert "proteinmpnn_residue_frequency_heatmap" not in deliverables
    assert "proteinmpnn_policy_proposal_spread" in deliverables
    proposal_path = manifest_path.parent / str(deliverables["proteinmpnn_policy_proposal_spread"]["path"])
    proposal_text = proposal_path.read_text(encoding="utf-8")
    assert "Mean NLL = -log P(aa | backbone)" in proposal_text
    assert "Open positions" in proposal_text
    assert "All modeled positions" in proposal_text
    assert "proteinmpnn_policy_residue_frequency" in deliverables
    frequency = deliverables["proteinmpnn_policy_residue_frequency"]
    views = list(dict(frequency["evidence_summary"])["policy_views"])
    assert [str(view["label"]) for view in views] == ["Distal", "Peripheral", "Combined"]
    assert all((manifest_path.parent / str(view["path"])).exists() for view in views)
    assert "expanded_proteinmpnn_fold_validation" not in deliverables
    assert "foldcheck_review_review_class_counts" not in deliverables
    assert "proteinmpnn_variant_similarity_heatmap" not in deliverables
    assert "proteinmpnn_tao_style_fold_validation" not in deliverables
    assert "foldcheck_review_fold_metric_scatter" not in deliverables
    assert "foldcheck_review_cryoem_vs_runtime_rmsd" not in deliverables
