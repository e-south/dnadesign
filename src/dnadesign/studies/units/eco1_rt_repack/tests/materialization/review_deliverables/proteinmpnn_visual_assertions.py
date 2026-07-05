"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/proteinmpnn_visual_assertions.py

ProteinMPNN visual-content assertions for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    resolve_manifest_path,
)


def assert_proteinmpnn_visual_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    """Assert baseline and expanded ProteinMPNN visuals stay plain and scoped."""

    diversity_text = _read_deliverable(manifest_path, deliverables, "proteinmpnn_score_mutation_burden")
    assert "ProteinMPNN scores describe proposal spread" in diversity_text
    assert "Sequence identity to Ec86 WT (%)" in diversity_text
    assert "Accepted designs retain a minority of WT residues." not in diversity_text
    assert "Sampling temperature" in diversity_text
    assert "ProteinMPNN score" in diversity_text
    assert "Global score" in diversity_text

    assert "proteinmpnn_variant_similarity_heatmap" not in deliverables
    assert "proteinmpnn_tao_style_fold_validation" not in deliverables
    assert "foldcheck_review_fold_metric_scatter" not in deliverables
    assert "foldcheck_review_cryoem_vs_runtime_rmsd" not in deliverables
    residue_frequency_text = _read_deliverable(manifest_path, deliverables, "proteinmpnn_residue_frequency_heatmap")
    assert "ProteinMPNN samples amino acids within each fixed mask" in residue_frequency_text
    assert "WT residue" in residue_frequency_text
    assert "Variants with this amino acid" in residue_frequency_text
    assert "Clade 9 25% + 5 A" in residue_frequency_text
    assert "Baseline ProteinMPNN mutation density across allowed residues" not in residue_frequency_text
    class_views = deliverables["proteinmpnn_residue_frequency_heatmap"]["evidence_summary"]["design_class_views"]
    assert len(class_views) == 6
    assert {view["label"] for view in class_views} >= {
        "Clade 9 25% + 5 A",
        "II-A3/42_1 50% + 5 A",
    }

    expanded_text = _read_deliverable(manifest_path, deliverables, "expanded_proteinmpnn_fold_validation")
    assert "Expanded designs preserve the RT fold" in expanded_text
    assert "Design class" in expanded_text
    assert "Sampling temperature" in expanded_text
    assert "Temperature 0.1" in expanded_text
    assert "Temperature 0.3" in expanded_text
    assert "Selected panel" in expanded_text
    assert "WT-runtime C-alpha RMSD" in expanded_text
    assert "Mean pLDDT" in expanded_text
    _assert_top_marginal_matches_square_panel(expanded_text)
    assert "576" not in expanded_text or "Baseline" not in expanded_text
    assert "design class as color" in str(deliverables["expanded_proteinmpnn_fold_validation"]["description"])

    fold_bin_text = _read_deliverable(manifest_path, deliverables, "foldcheck_review_review_class_counts")
    assert "Each fixed mask keeps foldable candidates" in fold_bin_text
    assert "clade 9 p25, 5 A" in fold_bin_text
    assert "clade 9 p25, 10 A" in fold_bin_text
    assert "subtype p50, 5 A" in fold_bin_text
    assert "Strong fold" in fold_bin_text
    assert "Review band" in fold_bin_text
    assert "Fold-review thresholds separate preserved folds" not in fold_bin_text
    assert "class-specific failures" in str(deliverables["foldcheck_review_review_class_counts"]["description"])


def _read_deliverable(manifest_path: Path, deliverables: dict[str, dict[str, object]], deliverable_id: str) -> str:
    path = resolve_manifest_path(manifest_path, str(deliverables[deliverable_id]["path"]))
    return path.read_text(encoding="utf-8")


def _assert_top_marginal_matches_square_panel(svg_text: str) -> None:
    rects = [
        tuple(float(value) for value in match)
        for match in re.findall(
            r'<clipPath id="[^"]+">\s*<rect x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" height="([0-9.]+)"',
            svg_text,
        )
    ]
    square_panels = [rect for rect in rects if abs(rect[2] - rect[3]) <= 0.5 and rect[2] > 100.0]
    assert square_panels, "expanded fold SVG should include a square main panel"
    main_panel = square_panels[0]
    top_marginals = [rect for rect in rects if rect[1] < main_panel[1] and rect[2] > 100.0 and rect[3] < main_panel[3]]
    assert top_marginals, "expanded fold SVG should include a top marginal histogram"
    top_marginal = top_marginals[0]
    assert abs(top_marginal[0] - main_panel[0]) <= 0.5
    assert abs(top_marginal[2] - main_panel[2]) <= 0.5
