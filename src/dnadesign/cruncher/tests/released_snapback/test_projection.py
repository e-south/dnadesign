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
from dnadesign.cruncher.snapback.released_projection import (
    _released_duplex_overlap_pairing_issues,
    evaluate_released_precursor,
)


def _nick_entry(motif: str = "AACGTTG", *, top_cut_offset: int = 0) -> NickaseCatalogEntry:
    return NickaseCatalogEntry(
        id="Nx.Exact7",
        specificity_id="Nx.Exact7",
        motif_top_5to3=motif,
        top_cut_offset=top_cut_offset,
    )


def _release_entry(*, top_cut_offset: int = 1, bottom_cut_offset: int = 0) -> ReleaseEnzymeEntry:
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
    assert evaluation.candidate.active_bottom_input_length_nt == 6
    assert evaluation.candidate.active_bottom_length_nt == 9
    assert evaluation.candidate.nick_boundary_from_left == 0
    assert evaluation.projection.release_top_cut_precursor == 10
    assert evaluation.projection.release_bottom_cut_precursor == 9
    assert evaluation.projection.retained_top_strand == ""
    assert evaluation.projection.retained_top_length_nt == 0
    assert evaluation.projection.active_bottom_strand == "TTGCAACAA"
    assert evaluation.projection.release_site_survives_post_release is False


def test_released_projection_retains_only_top_prefix_left_of_nick_for_nonzero_boundary() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="CCTAAGCATTAAGAGACC",
        nick_entry=NickaseCatalogEntry(
            id="Nt.Bpu10I",
            specificity_id="Bpu10I",
            motif_top_5to3="CCTNAGC",
            top_cut_offset=2,
        ),
        release_entry=ReleaseEnzymeEntry(
            variant_id="BsaI-HFv2",
            display_name="BsaI-HFv2",
            recognition_sequence="GGTCTC",
            top_cut_offset=7,
            bottom_cut_offset=11,
            class_label="type_iis",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="type_iis_release_v1",
        ),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=2, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(require_nick_survives_in_retained_product=False),
    )

    assert evaluation.status == "satisfied"
    assert evaluation.projection is not None
    assert evaluation.projection.retained_top_strand == "CC"
    assert evaluation.projection.retained_top_length_nt == 2
    assert evaluation.projection.active_bottom_strand == "GGATTCGTAAT"


def test_released_projection_rejects_exact_0_3_3_with_left_of_origin_nickase_site() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="GTCTCAAACGTTGTTCCAA",
        nick_entry=_nick_entry(motif="GTCTC", top_cut_offset=6),
        release_entry=_release_entry(),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(),
        precursor_coordinate_offset=6,
    )

    assert evaluation.status == "invalid_precursor"
    assert any(issue.code == "PRE_NICK_SITE_LEFT_OF_ORIGIN" for issue in evaluation.issues)


def test_released_projection_rejects_internal_cut_nickase_sites_that_overlap_the_active_strand() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="CCTAAGCATTAAGAGACC",
        nick_entry=NickaseCatalogEntry(
            id="Nt.Bpu10I",
            specificity_id="Bpu10I",
            motif_top_5to3="CCTNAGC",
            top_cut_offset=2,
        ),
        release_entry=ReleaseEnzymeEntry(
            variant_id="BsaI-HFv2",
            display_name="BsaI-HFv2",
            recognition_sequence="GGTCTC",
            top_cut_offset=7,
            bottom_cut_offset=11,
            class_label="type_iis",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="type_iis_release_v1",
        ),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(),
        precursor_coordinate_offset=2,
    )

    assert evaluation.status == "invalid_precursor"
    assert any(issue.code == "PRE_NICK_SITE_OVERLAPS_ACTIVE_STRAND" for issue in evaluation.issues)


def test_released_projection_rejects_pre_nick_sites_left_of_origin() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="GTCTCAAAAAAATTTAGAAGAGC",
        nick_entry=NickaseCatalogEntry(
            id="Nt.BsmAI",
            specificity_id="BsmAI",
            motif_top_5to3="GTCTC",
            top_cut_offset=6,
        ),
        release_entry=ReleaseEnzymeEntry(
            variant_id="BspQI",
            display_name="BspQI",
            recognition_sequence="GCTCTTC",
            top_cut_offset=8,
            bottom_cut_offset=11,
            class_label="type_iis",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="type_iis_release_v1",
        ),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(),
        precursor_coordinate_offset=6,
    )

    assert evaluation.status == "invalid_precursor"
    assert any(issue.code == "PRE_NICK_SITE_LEFT_OF_ORIGIN" for issue in evaluation.issues)


def test_released_projection_rejects_when_release_does_not_fully_separate_downstream_fragment() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="AACGTTGTTCCAA",
        nick_entry=_nick_entry(),
        release_entry=_release_entry(top_cut_offset=4, bottom_cut_offset=0),
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


def test_released_projection_does_not_treat_another_same_orientation_site_as_survival() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="CCAATTTGGCCAA",
        nick_entry=_nick_entry(motif="CCAATTT"),
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


def test_released_projection_rejects_legacy_retained_top_nick_survival_constraint() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="AACGTTGTTCCAA",
        nick_entry=_nick_entry(),
        release_entry=_release_entry(),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(require_nick_survives_in_retained_product=True),
    )

    assert evaluation.status == "no_release_path"
    assert any(issue.code == "LEGACY_RETAINED_TOP_NICK_SURVIVAL_UNSUPPORTED" for issue in evaluation.issues)


def test_released_projection_preserves_real_precursor_extra_nick_evidence() -> None:
    evaluation = evaluate_released_precursor(
        precursor_top_strand="AACGTTGTTCCAAGAACGTTG",
        nick_entry=_nick_entry(),
        release_entry=_release_entry(),
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        constraints=ReleasedSnapbackConstraintsSpec(),
    )

    assert evaluation.status == "satisfied"
    assert evaluation.candidate is not None
    assert evaluation.candidate.extra_nick_event_count == 1
    assert evaluation.candidate.extra_target_strand_nick_count == 1
    assert [event.boundary_context for event in evaluation.candidate.extra_nick_events] == [14]


def test_released_projection_rejects_non_watson_crick_post_release_duplex_overlap() -> None:
    issues = _released_duplex_overlap_pairing_issues(
        retained_top_strand="AACCAA",
        active_bottom_strand="TTTG",
        coordinate_offset=2,
        release_top_cut_precursor=6,
        release_bottom_cut_precursor=4,
    )

    assert len(issues) == 1
    assert issues[0].code == "POST_RELEASE_DUPLEX_PAIRING_INVALID"
    assert issues[0].details["retained_top_overlap"] == "CC"
    assert issues[0].details["active_bottom_overlap"] == "TT"


def test_released_projection_allows_zero_length_top_prefix_after_origin_nick() -> None:
    issues = _released_duplex_overlap_pairing_issues(
        retained_top_strand="",
        active_bottom_strand="TTGCAACAA",
        coordinate_offset=0,
        release_top_cut_precursor=0,
        release_bottom_cut_precursor=9,
    )

    assert issues == []
