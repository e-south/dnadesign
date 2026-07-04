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

from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    resolve_manifest_path,
)


def assert_proteinmpnn_visual_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    """Assert baseline and expanded ProteinMPNN visuals stay plain and scoped."""

    diversity_text = _read_deliverable(manifest_path, deliverables, "proteinmpnn_score_mutation_burden")
    assert "Baseline ProteinMPNN proposal scores and mutation burden" in diversity_text
    assert "Sequence identity to Ec86 WT (%)" in diversity_text
    assert "Accepted designs retain a minority of WT residues." not in diversity_text
    assert "Sampling temperature" in diversity_text
    assert "ProteinMPNN score" in diversity_text
    assert "Global score" in diversity_text

    mutation_density_text = _read_deliverable(manifest_path, deliverables, "proteinmpnn_mutation_density")
    assert "Baseline ProteinMPNN mutation density across allowed residues" in mutation_density_text
    assert "RT1-RT7 annotation intervals" in mutation_density_text
    assert "Retained DNA/RNA &lt;=5 A" in mutation_density_text
    assert "Motif anchors: NAxxH/YADD/VTG" in mutation_density_text

    similarity_text = _read_deliverable(manifest_path, deliverables, "proteinmpnn_variant_similarity_heatmap")
    assert "Baseline ProteinMPNN variants are mapped against the Ec86 WT sequence" in similarity_text
    assert "Same as WT" in similarity_text
    assert "Different from WT" in similarity_text
    assert "Missing backbone context" in similarity_text
    assert "canonical positions" in str(deliverables["proteinmpnn_variant_similarity_heatmap"]["description"])
    assert "descriptive sequence-similarity view" in str(
        deliverables["proteinmpnn_variant_similarity_heatmap"]["interpretation_limit"]
    )

    tao_text = _read_deliverable(manifest_path, deliverables, "proteinmpnn_tao_style_fold_validation")
    assert "Baseline ProteinMPNN designs cluster by ColabFold RMSD and pLDDT" in tao_text
    assert "WT-runtime C-alpha RMSD" in tao_text
    assert "Mean pLDDT" in tao_text
    assert "baseline" in str(deliverables["proteinmpnn_tao_style_fold_validation"]["description"]).lower()
    assert "single active mask policy" in str(
        deliverables["proteinmpnn_tao_style_fold_validation"]["interpretation_limit"]
    )

    expanded_text = _read_deliverable(manifest_path, deliverables, "expanded_proteinmpnn_fold_validation")
    assert "Expanded ProteinMPNN design classes cluster by ColabFold RMSD and pLDDT" in expanded_text
    assert "Design class" in expanded_text
    assert "Sampling temperature" in expanded_text
    assert "Temperature 0.1" in expanded_text
    assert "Temperature 0.3" in expanded_text
    assert "Selected panel" in expanded_text
    assert "WT-runtime C-alpha RMSD" in expanded_text
    assert "Mean pLDDT" in expanded_text
    assert "576" not in expanded_text or "Baseline" not in expanded_text
    assert "design class as color" in str(deliverables["expanded_proteinmpnn_fold_validation"]["description"])


def _read_deliverable(manifest_path: Path, deliverables: dict[str, dict[str, object]], deliverable_id: str) -> str:
    path = resolve_manifest_path(manifest_path, str(deliverables[deliverable_id]["path"]))
    return path.read_text(encoding="utf-8")
