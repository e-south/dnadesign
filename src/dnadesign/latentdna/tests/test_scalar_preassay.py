from dnadesign.latentdna.src.scalars.preassay import PREASSAY_BUILDER_KINDS


def test_preassay_builder_registry_tracks_active_summary_kinds() -> None:
    assert PREASSAY_BUILDER_KINDS == {
        "representation_health_summary",
        "design_structure_summary",
        "sigma35_ordinal_audit",
        "context_robustness_summary",
        "reference_alignment_summary",
    }
