"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/scoring/esmc_pseudolikelihood/requests.py

Build leave-one-out ESMC pseudo-likelihood jobs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

from dnadesign.permuter.src.scoring.esmc_pseudolikelihood.contracts import (
    CANONICAL_AMINO_ACIDS,
    EsmcPseudolikelihoodJob,
)


def build_pseudolikelihood_jobs(
    *,
    sequence_id: str,
    sequence: str,
    positions: tuple[int, ...] | None = None,
) -> list[EsmcPseudolikelihoodJob]:
    """Return one masked context per selected protein position."""

    normalized_id = str(sequence_id or "").strip()
    if not normalized_id:
        raise ValueError("sequence_id is required")
    normalized = _normalize_protein_sequence(sequence)
    selected_positions = _normalize_positions(sequence_length=len(normalized), positions=positions)
    source_hash = _protein_sequence_hash(normalized)
    jobs: list[EsmcPseudolikelihoodJob] = []
    for canonical_position in selected_positions:
        residue_index = canonical_position - 1
        residue = normalized[residue_index]
        masked_sequence = normalized[:residue_index] + "_" + normalized[residue_index + 1 :]
        jobs.append(
            EsmcPseudolikelihoodJob(
                sequence_id=normalized_id,
                sequence=normalized,
                sequence_hash=source_hash,
                canonical_position=canonical_position,
                residue_index_zero_based=residue_index,
                residue=residue,
                masked_sequence=masked_sequence,
                masked_sequence_hash=_protein_sequence_hash(masked_sequence),
            )
        )
    return jobs


def _normalize_protein_sequence(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("sequence is required")
    invalid = sorted(set(normalized) - set(CANONICAL_AMINO_ACIDS))
    if invalid:
        raise ValueError(f"sequence contains unsupported residue(s): {invalid}")
    return normalized


def _protein_sequence_hash(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_positions(*, sequence_length: int, positions: tuple[int, ...] | None) -> tuple[int, ...]:
    if not positions:
        return tuple(range(1, sequence_length + 1))
    normalized = tuple(int(position) for position in positions)
    bad = [position for position in normalized if position < 1 or position > sequence_length]
    if bad:
        raise ValueError(f"protein position(s) out of bounds for length {sequence_length}: {bad}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate protein positions are not allowed: {normalized}")
    return normalized
