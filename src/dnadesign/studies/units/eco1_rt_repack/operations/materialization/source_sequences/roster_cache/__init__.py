"""Conservation roster-cache primitive for Eco1 RT repack."""

from importlib import import_module

__all__ = [
    "MaterializedConservationRosterCache",
    "materialize_conservation_roster_cache",
]


def __getattr__(name: str) -> object:
    if name in {"MaterializedConservationRosterCache", "materialize_conservation_roster_cache"}:
        module = import_module(
            "dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.pipeline"
        )
        return getattr(module, name)
    raise AttributeError(name)
