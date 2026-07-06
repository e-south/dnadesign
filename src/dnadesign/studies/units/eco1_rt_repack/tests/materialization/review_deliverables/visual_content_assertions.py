"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/visual_content_assertions.py

Shared visual-content assertions for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.plain_titles import (
    PLAIN_DELIVERABLE_TITLES,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    resolve_manifest_path,
)


def assert_mask_and_msa_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    msa_text = _read_deliverable(manifest_path, deliverables, "msa_plurality_mask_panel")
    assert "Clade 9 conservation defines the baseline mask" in msa_text
    assert "all accepted clade 9 alignment rows" in str(deliverables["msa_plurality_mask_panel"]["description"])
    assert "current conservation mask uses this clade 9 denominator" in str(
        deliverables["msa_plurality_mask_panel"]["description"]
    )
    assert deliverables["msa_plurality_mask_panel"]["evidence_summary"]["current_mask_denominator"] is True
    assert "ec86_clade9_conservation_v1__" not in msa_text
    assert "clade 9 25% WT plurality" in msa_text
    assert "clade 9 50% WT plurality" in msa_text
    assert "Baseline fixed residues (clade 9 p25 + 5 A)" in msa_text
    assert "Mask-protected" not in msa_text
    assert "Subtype II-A3/42_1 rows" not in msa_text
    assert "II-A3 subset | C9 001 fig|fixture.1.peg.1" in msa_text

    subtype_text = _read_deliverable(manifest_path, deliverables, "msa_subtype_plurality_panel")
    assert "The Eco1 subtype MSA gives a closer conservation view" in subtype_text
    assert "all accepted II-A3/42_1 subtype alignment rows" in str(
        deliverables["msa_subtype_plurality_panel"]["description"]
    )
    assert "does not replace the clade 9 denominator" in str(deliverables["msa_subtype_plurality_panel"]["description"])
    assert deliverables["msa_subtype_plurality_panel"]["evidence_summary"]["current_mask_denominator"] is False
    assert "II-A3 002 fig|fixture.2.peg.1" in subtype_text
    assert "II-A3/42_1 25% WT plurality" in subtype_text
    assert "II-A3/42_1 50% WT plurality" in subtype_text
    assert "Baseline fixed residues (clade 9 p25 + 5 A)" in subtype_text
    assert "linear_mask_tracks" not in deliverables

    design_class_mask_text = _read_deliverable(manifest_path, deliverables, "design_class_mask_overview")
    assert "Fixed residues combine motif, conservation, and substrate rules" in design_class_mask_text
    assert "WT amino acid" not in design_class_mask_text
    assert "Residue position" in design_class_mask_text
    assert "EC86 canonical residue position" not in design_class_mask_text
    assert "EC86 per-residue ruler" not in design_class_mask_text
    assert "Mask evidence and design-class policy" not in design_class_mask_text
    assert "Clade 9 25% + 5 A | 4 fixed" in design_class_mask_text
    assert "Clade 9 25% + 6 A" in design_class_mask_text
    assert "Clade 9 25% + 8 A" in design_class_mask_text
    assert "Clade 9 25% + 10 A" in design_class_mask_text
    assert "Clade 9 50% + 5 A" in design_class_mask_text
    assert "II-A3/42_1 50% + 5 A" in design_class_mask_text
    assert "Clade 9: &gt;=25% WT plurality" in design_class_mask_text
    assert "Clade 9: &gt;=50% WT plurality" in design_class_mask_text
    assert "II-A3/42_1: &gt;=50% WT plurality" in design_class_mask_text
    assert "Wang/EC86 substrate-contact priors" in design_class_mask_text
    assert "Wang/Ec86" not in design_class_mask_text
    assert "DNA/RNA within 10 A" in design_class_mask_text
    assert "Conservation threshold" not in design_class_mask_text
    assert "DNA/RNA contact threshold" not in design_class_mask_text
    assert "Fixed by design-class policy" not in design_class_mask_text
    assert "Designable by design-class policy" not in design_class_mask_text
    assert "editable" not in design_class_mask_text
    assert "#009e73" not in design_class_mask_text.lower()
    assert "Fixed-residue union" not in design_class_mask_text
    assert "Protected union" not in design_class_mask_text
    assert "current baseline only" not in design_class_mask_text.lower()

    _assert_premise_titles(deliverables)


def assert_linked_fold_and_esmc_content(manifest_path: Path, deliverables: dict[str, dict[str, object]]) -> None:
    assert "foldcheck_review_fold_metric_scatter" not in deliverables
    assert "foldcheck_review_cryoem_vs_runtime_rmsd" not in deliverables
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
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["title"] == "ESMC scores WT-context substitutions"
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["render_mode"] == "wide_visual"
    assert "LLR = log P(alternate) - log P(WT)" in str(
        deliverables["wt_esmc_substitution_llr_heatmap"]["method_summary"]
    )
    assert deliverables["wt_esmc_substitution_llr_heatmap"]["evidence_summary"]["substitution_llr_rows"] == 114

    esmc_scatter_text = _read_deliverable(manifest_path, deliverables, "msa_plurality_vs_esmc_entropy")
    assert "Clade 9 plurality tracks lower ESMC entropy" in esmc_scatter_text
    assert "Pearson r =" in esmc_scatter_text
    assert "R2 =" in esmc_scatter_text
    assert "Linear fit" in esmc_scatter_text
    assert "25% plurality threshold" not in esmc_scatter_text
    assert "model check of the WT sequence context" in str(
        deliverables["msa_plurality_vs_esmc_entropy"]["interpretation_limit"]
    )


def assert_selection_content(deliverables: dict[str, dict[str, object]]) -> None:
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


def _assert_premise_titles(deliverables: dict[str, dict[str, object]]) -> None:
    stale_titles = {
        "Baseline ProteinMPNN variants are mapped against the Ec86 WT sequence",
        "Baseline ProteinMPNN mutation density across allowed residues",
        "Design-class residue mask evidence across EC86 RT",
        "Eco1 panel-selection funnel summary",
        "RT-only handoff readiness",
        "Fold-review thresholds separate preserved folds from review-band candidates",
        "ColabFold metrics show continuous review signals",
    }
    banned_fragments = (
        "review surface",
        "current baseline only",
        "thresholds separate",
        "rank candidates",
        "best variants",
        "top six",
        "processivity improvement",
        "strand-displacement improvement",
    )
    for deliverable_id, row in deliverables.items():
        title = str(row.get("title") or "")
        display_text = " ".join(str(row.get(field) or "") for field in ("title", "alt_text", "description")).lower()
        assert title, f"{deliverable_id} must carry a title"
        assert not title.endswith("."), f"{deliverable_id} title must omit terminal periods"
        assert len(title) <= 72, f"{deliverable_id} title is too long: {title!r}"
        assert title not in stale_titles, f"{deliverable_id} title still uses stale noun-label wording"
        assert not any(fragment in title.lower() for fragment in banned_fragments), (
            f"{deliverable_id} title still has slop wording: {title!r}"
        )
        assert "ranked bar plot" not in display_text
        assert "candidate-score rank changes" not in display_text
        assert "candidate additive llr rankings" not in display_text
        expected_title = PLAIN_DELIVERABLE_TITLES.get(deliverable_id)
        if expected_title is not None:
            assert title == expected_title
