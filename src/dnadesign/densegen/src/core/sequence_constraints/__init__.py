"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/sequence_constraints/__init__.py

Sequence-constraint compilation, scanning, and constrained sampling helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .engine import (
    CompiledSequenceConstraints,
    SequenceValidationResult,
    compile_sequence_constraints,
    validate_sequence_constraints,
)
from .kmers import find_kmer_matches, reverse_complement
from .sampler import ConstrainedSequenceError, generate_constrained_sequence

__all__ = [
    "CompiledSequenceConstraints",
    "ConstrainedSequenceError",
    "SequenceValidationResult",
    "compile_sequence_constraints",
    "find_kmer_matches",
    "generate_constrained_sequence",
    "reverse_complement",
    "validate_sequence_constraints",
]
