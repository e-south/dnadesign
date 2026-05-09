from dnadesign.latentdna.src.scalars.preassay import PREASSAY_BUILDER_KINDS


def test_preassay_builder_registry_tracks_active_summary_kinds() -> None:
    assert PREASSAY_BUILDER_KINDS == {
        "candidate_decision_frontier",
        "cohort_structure_summary",
        "context_pair_summary",
        "representation_health_summary",
        "design_structure_summary",
        "ordinal_axis_audit",
        "axis_centroid_distance",
        "candidate_x_selection_scorecard",
        "collection_strength_ordinal_audit",
        "context_robustness_summary",
        "reference_alignment_summary",
        "reference_to_centroid_similarity",
    }
