"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/scoring/esmc_masked_marginal/contracts.py

Contracts for ESMC masked-marginal protein mutation scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
MASKED_MARGINAL_SCORING_METHOD_ID = "esmc_masked_marginal_v1"


@dataclass(frozen=True)
class MaskedMarginalJob:
    """One masked reference-sequence context for a protein DMS query."""

    sequence_id: str
    sequence: str
    sequence_hash: str
    canonical_position: int
    residue_index_zero_based: int
    wt_aa: str
    masked_sequence: str
    masked_sequence_hash: str


@dataclass(frozen=True)
class MaskedMarginalRows:
    """Normalized rows from one masked-marginal Biohub logits response."""

    position_row: dict[str, object]
    substitution_rows: list[dict[str, object]]


@dataclass(frozen=True)
class MaskedMarginalArtifacts:
    """Paths emitted by one masked-marginal scoring materialization."""

    position_entropy_path: Path
    substitution_llr_path: Path
    manifest_path: Path
