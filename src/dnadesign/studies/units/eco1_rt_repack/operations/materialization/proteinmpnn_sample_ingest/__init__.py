"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_sample_ingest/__init__.py

Eco1 ProteinMPNN sample-table materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_sample_ingest.pipeline import (
    materialize_proteinmpnn_samples,
)

__all__ = ["materialize_proteinmpnn_samples"]
