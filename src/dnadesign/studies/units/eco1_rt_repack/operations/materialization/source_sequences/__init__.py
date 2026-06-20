"""Conservation source-sequence bundle primitive for Eco1 RT repack."""

__all__ = [
    "MaterializedSourceSequenceBundles",
    "MaterializedConservationRosterCache",
    "MaterializedProviderSourceFastas",
    "SourceSequenceBundleSufficiencyReport",
    "materialize_conservation_roster_cache",
    "materialize_provider_source_fastas",
    "materialize_source_sequence_bundles",
    "validate_source_sequence_bundle_sufficiency",
]


def __getattr__(name: str) -> object:
    if name in {"MaterializedSourceSequenceBundles", "materialize_source_sequence_bundles"}:
        from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.pipeline import (
            MaterializedSourceSequenceBundles,
            materialize_source_sequence_bundles,
        )

        return {
            "MaterializedSourceSequenceBundles": MaterializedSourceSequenceBundles,
            "materialize_source_sequence_bundles": materialize_source_sequence_bundles,
        }[name]
    if name in {"MaterializedConservationRosterCache", "materialize_conservation_roster_cache"}:
        from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache import (
            MaterializedConservationRosterCache,
            materialize_conservation_roster_cache,
        )

        return {
            "MaterializedConservationRosterCache": MaterializedConservationRosterCache,
            "materialize_conservation_roster_cache": materialize_conservation_roster_cache,
        }[name]
    if name in {"MaterializedProviderSourceFastas", "materialize_provider_source_fastas"}:
        from .provider_sources import (
            MaterializedProviderSourceFastas,
            materialize_provider_source_fastas,
        )

        return {
            "MaterializedProviderSourceFastas": MaterializedProviderSourceFastas,
            "materialize_provider_source_fastas": materialize_provider_source_fastas,
        }[name]
    if name in {"SourceSequenceBundleSufficiencyReport", "validate_source_sequence_bundle_sufficiency"}:
        from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency import (
            SourceSequenceBundleSufficiencyReport,
            validate_source_sequence_bundle_sufficiency,
        )

        return {
            "SourceSequenceBundleSufficiencyReport": SourceSequenceBundleSufficiencyReport,
            "validate_source_sequence_bundle_sufficiency": validate_source_sequence_bundle_sufficiency,
        }[name]
    raise AttributeError(name)
