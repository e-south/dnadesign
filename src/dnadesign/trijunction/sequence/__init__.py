"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/sequence/__init__.py

Public DNA sequence primitives for TriJunction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .alphabet import DNA_ALPHABET, reverse_complement, validate_dna
from .distances import (
    POSITION_WEIGHT_SCALE,
    directional_position_weighted_levenshtein,
    levenshtein_distance,
    position_weight_units,
    position_weighted_levenshtein,
    position_weighted_levenshtein_units,
    position_weighted_levenshtein_units_many,
)
from .substrings import kmer_set, kmer_set_with_reverse_complements, longest_common_substring_length

__all__ = [
    "DNA_ALPHABET",
    "POSITION_WEIGHT_SCALE",
    "directional_position_weighted_levenshtein",
    "kmer_set",
    "kmer_set_with_reverse_complements",
    "levenshtein_distance",
    "longest_common_substring_length",
    "position_weighted_levenshtein",
    "position_weighted_levenshtein_units",
    "position_weighted_levenshtein_units_many",
    "position_weight_units",
    "reverse_complement",
    "validate_dna",
]
