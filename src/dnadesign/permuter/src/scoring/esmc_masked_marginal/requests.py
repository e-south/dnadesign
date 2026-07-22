"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/scoring/esmc_masked_marginal/requests.py

Build masked protein-position requests from the Permuter protein DMS contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

from dnadesign.permuter.src.api import ProteinDmsRequest
from dnadesign.permuter.src.scoring.esmc_masked_marginal.contracts import (
    CANONICAL_AMINO_ACIDS,
    MaskedMarginalJob,
)


def build_masked_marginal_jobs(
    request: ProteinDmsRequest,
    *,
    sequence_id: str | None = None,
) -> list[MaskedMarginalJob]:
    """Return one masked context per selected protein DMS position."""

    ref_name = str(request.ref_name or "").strip()
    if not ref_name:
        raise ValueError("ProteinDmsRequest.ref_name is required")
    normalized = _normalize_protein_sequence(request.sequence)
    positions = _normalize_positions(sequence_length=len(normalized), positions=request.positions)
    selected_sequence_id = sequence_id or ref_name
    source_hash = protein_sequence_hash(normalized)
    jobs: list[MaskedMarginalJob] = []
    for canonical_position in positions:
        residue_index = canonical_position - 1
        wt_aa = normalized[residue_index]
        masked_sequence = normalized[:residue_index] + "_" + normalized[residue_index + 1 :]
        jobs.append(
            MaskedMarginalJob(
                sequence_id=selected_sequence_id,
                sequence=normalized,
                sequence_hash=source_hash,
                canonical_position=canonical_position,
                residue_index_zero_based=residue_index,
                wt_aa=wt_aa,
                masked_sequence=masked_sequence,
                masked_sequence_hash=protein_sequence_hash(masked_sequence),
            )
        )
    return jobs


def protein_sequence_hash(sequence: str) -> str:
    """Hash a normalized protein sequence with the same sha256 URI shape used elsewhere."""

    normalized = "".join(str(sequence).split()).upper()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_protein_sequence(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("ProteinDmsRequest.sequence is required")
    invalid = sorted(set(normalized) - set(CANONICAL_AMINO_ACIDS))
    if invalid:
        raise ValueError(f"ProteinDmsRequest.sequence contains unsupported residue(s): {invalid}")
    return normalized


def _normalize_positions(*, sequence_length: int, positions: tuple[int, ...]) -> tuple[int, ...]:
    if not positions:
        return tuple(range(1, sequence_length + 1))
    normalized = tuple(int(position) for position in positions)
    bad = [position for position in normalized if position < 1 or position > sequence_length]
    if bad:
        raise ValueError(f"Protein position(s) out of bounds for length {sequence_length}: {bad}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate protein positions are not allowed: {normalized}")
    return normalized
