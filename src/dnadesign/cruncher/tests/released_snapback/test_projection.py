"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/released_snapback/test_projection.py

Projection and evaluator-reuse tests for released-product snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeEntry
from dnadesign.cruncher.snapback.released_models import (
    ReleasedFinalTargetGeometry,
    ReleasedSnapbackConstraintsSpec,
)
from dnadesign.cruncher.snapback.released_projection import evaluate_released_precursor


def _nick_entry(motif: str = "AACGTTG", *, top_cut_offset: int = 0) -> NickaseCatalogEntry:
    return NickaseCatalogEntry(
        id="Nx.Exact7",
        specificity_id="Nx.Exact7",
        motif_top_5to3=motif,
        top_cut_offset=top_cut_offset,
    )


def _release_entry(*, top_cut_offset: int = 0, bottom_cut_offset: int = 1) -> ReleaseEnzymeEntry:
    return ReleaseEnzymeEntry(
        variant_id="Re.Test",
        display_name="Re.Test",
        recognition_sequence="CCAA",
        top_cut_offset=top_cut_offset,
        bottom_cut_offset=bottom_cut_offset,
        class_label="other_ds_re",
        commercial_confidence="primary_vendor_current",
        source_catalog_id="local_release",
    )


def test_released_projection_supports_exact_0_3_3_with_nickase_site_larger_than_final_input_budget() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="AACGTTGTTCCAA",
        nick_entry=_nick_entry(),
        release_entry=_release_entry(),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(),
    )

    assert evaluation.status == "satisfied"
    assert evaluation.pre_nick_match is not None
    assert evaluation.candidate is not None
    assert evaluation.projection is not None
    assert evaluation.pre_nick_match.site.end - evaluation.pre_nick_match.site.start == 7
    assert evaluation.candidate.input_length_nt == 6
    assert evaluation.candidate.retained_product_length_nt == 9
    assert evaluation.candidate.nick_boundary_from_left == 0
    assert evaluation.projection.release_top_cut_precursor == 9
    assert evaluation.projection.release_site_survives_post_release is False


def test_released_projection_rejects_when_release_does_not_fully_separate_downstream_fragment() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="AACGTTGTTCCAA",
        nick_entry=_nick_entry(),
        release_entry=_release_entry(top_cut_offset=0, bottom_cut_offset=4),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(),
    )

    assert evaluation.status == "no_release_path"
    assert any(issue.code == "RELEASE_DOES_NOT_SEPARATE_DOWNSTREAM_FRAGMENT" for issue in evaluation.issues)


def test_released_projection_enforces_release_site_survival_when_requested() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="AACGTTGTTCCAA",
        nick_entry=_nick_entry(),
        release_entry=_release_entry(),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(allow_post_release_loss_of_release_site=False),
    )

    assert evaluation.status == "no_release_path"
    assert any(issue.code == "POST_RELEASE_RELEASE_SITE_LOST" for issue in evaluation.issues)


def test_released_projection_reuses_explicit_geometry_checks_for_foldback_mismatches() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="AACGTTATTCCAA",
        nick_entry=_nick_entry(motif="AACGTTA"),
        release_entry=_release_entry(),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(),
    )

    assert evaluation.status == "unsatisfied"
    assert any(issue.code == "HOMOLOGY_MISMATCH_LIMIT_EXCEEDED" for issue in evaluation.issues)
