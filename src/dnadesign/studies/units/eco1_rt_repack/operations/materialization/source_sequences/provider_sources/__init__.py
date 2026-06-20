"""Provider-source acquisition for Eco1 conservation source sequences."""

from .pipeline import (
    MaterializedProviderSourceFastas,
    materialize_provider_source_fastas,
)

__all__ = [
    "MaterializedProviderSourceFastas",
    "materialize_provider_source_fastas",
]
