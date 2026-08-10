"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/presentation/sequence_review/__init__.py

Public exports for Junction sequence-comparison evidence and plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .contract import JunctionSequenceChoice, JunctionSequenceDissimilarityV1
from .plot import plot_sequence_dissimilarity, render_sequence_dissimilarity_svg
from .projection import render_sequence_dissimilarity_contracts, sequence_dissimilarity_contracts

__all__ = [
    "JunctionSequenceChoice",
    "JunctionSequenceDissimilarityV1",
    "plot_sequence_dissimilarity",
    "render_sequence_dissimilarity_svg",
    "render_sequence_dissimilarity_contracts",
    "sequence_dissimilarity_contracts",
]
