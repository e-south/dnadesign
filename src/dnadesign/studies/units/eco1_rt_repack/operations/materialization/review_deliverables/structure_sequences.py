"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/structure_sequences.py

Protein-sequence lookup for Eco1 structure-browser manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sequence_by_candidate(
    candidate_table_path: Path | None,
    *,
    foldcheck_fasta_path: Path | None,
) -> dict[str, dict[str, Any]]:
    """Return protein-sequence payloads keyed by candidate id."""

    sequences: dict[str, dict[str, Any]] = {}
    if candidate_table_path is not None and candidate_table_path.exists():
        required_columns = {"candidate_id", "sequence", "sequence_hash"}
        if required_columns.issubset(set(pq.read_schema(candidate_table_path).names)):
            table = pq.read_table(candidate_table_path, columns=["candidate_id", "sequence", "sequence_hash"])
            for row in table.to_pylist():
                sequence = str(row.get("sequence") or "").strip().upper()
                if not sequence:
                    continue
                sequences[str(row["candidate_id"])] = {
                    "protein_sequence": sequence,
                    "sequence_hash": str(row.get("sequence_hash") or ""),
                    "amino_acid_length": len(sequence),
                    "sequence_source": "candidate_table.parquet",
                }
    if foldcheck_fasta_path is not None and foldcheck_fasta_path.exists():
        for candidate_id, sequence in read_fasta_sequences(foldcheck_fasta_path).items():
            sequences[candidate_id] = {
                "protein_sequence": sequence,
                "sequence_hash": "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                "amino_acid_length": len(sequence),
                "sequence_source": "foldcheck_request/input_sequences.fasta",
            }
    return sequences


def read_fasta_sequences(path: Path) -> dict[str, str]:
    """Read FASTA records into uppercase sequence strings keyed by record id."""

    sequences: dict[str, list[str]] = {}
    current_id = ""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            current_id = stripped[1:].split()[0].strip()
            if not current_id:
                raise ValueError(f"FASTA header without sequence id at {path}:{line_number}")
            sequences.setdefault(current_id, [])
            continue
        if not current_id:
            raise ValueError(f"FASTA sequence line appears before a header at {path}:{line_number}")
        sequences[current_id].append(stripped)
    return {candidate_id: "".join(parts).upper() for candidate_id, parts in sequences.items()}


__all__ = ["read_fasta_sequences", "sequence_by_candidate"]
