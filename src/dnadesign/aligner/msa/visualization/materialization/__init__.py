"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/materialization/__init__.py

Materialization pipeline for generic MSA visualization sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.aligner.msa.visualization.materialization.pipeline import (
    materialize_msa_visualizations,
)

__all__ = ["materialize_msa_visualizations"]
