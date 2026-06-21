"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/__init__.py

Provider-source acquisition for Eco1 conservation source sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .pipeline import (
    MaterializedProviderSourceFastas,
    materialize_provider_source_fastas,
)

__all__ = [
    "MaterializedProviderSourceFastas",
    "materialize_provider_source_fastas",
]
