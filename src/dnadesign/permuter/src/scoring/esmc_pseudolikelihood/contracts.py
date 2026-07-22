"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/scoring/esmc_pseudolikelihood/contracts.py

Contracts for ESMC leave-one-out protein pseudo-likelihood scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
ESMC_PSEUDOLIKELIHOOD_METHOD_ID = "esmc_leave_one_out_pseudolikelihood_v1"


@dataclass(frozen=True)
class EsmcPseudolikelihoodJob:
    """One leave-one-out masked context for a protein sequence."""

    sequence_id: str
    sequence: str
    sequence_hash: str
    canonical_position: int
    residue_index_zero_based: int
    residue: str
    masked_sequence: str
    masked_sequence_hash: str


@dataclass(frozen=True)
class EsmcPseudolikelihoodRows:
    """Normalized rows from one ESMC sequence-logits pseudo-likelihood query."""

    position_row: dict[str, object]


@dataclass(frozen=True)
class EsmcPseudolikelihoodArtifacts:
    """Paths emitted by one ESMC pseudo-likelihood materialization."""

    position_pll_path: Path
    sequence_pll_path: Path
    manifest_path: Path
