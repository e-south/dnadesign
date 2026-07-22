"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_request/sequences.py

Eco1 sequence reconstruction for fold-check requests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.thread.foldcheck import FoldCheckSequenceRecord, sequence_hash

_MUTATION_RE = re.compile(r"^([A-Z])([0-9]+)([A-Z])$")


def build_foldcheck_sequence_records(
    *,
    candidate_table_path: Path,
    residue_map_path: Path,
    wt_sequence_id: str,
) -> list[FoldCheckSequenceRecord]:
    """Build WT plus full-canonical candidate sequences for fold checking."""

    wt_sequence = _wt_sequence_from_residue_map(residue_map_path)
    candidate_rows = [
        row for row in pq.read_table(candidate_table_path).to_pylist() if str(row.get("status")) == "accepted"
    ]
    if not candidate_rows:
        raise ValueError("fold-check request requires accepted candidate_table rows")
    candidate_rows.sort(key=lambda row: (int(row["rank"]), str(row["candidate_id"])))

    records = [
        FoldCheckSequenceRecord(
            sequence_id=wt_sequence_id,
            sequence=wt_sequence,
            sequence_hash=sequence_hash(wt_sequence),
            source_kind="wild_type_baseline",
        )
    ]
    for row in candidate_rows:
        candidate_id = str(row["candidate_id"])
        sequence = _apply_canonical_mutations(wt_sequence, row.get("canonical_mutations"), candidate_id)
        records.append(
            FoldCheckSequenceRecord(
                sequence_id=candidate_id,
                sequence=sequence,
                sequence_hash=sequence_hash(sequence),
                source_kind="proteinmpnn_candidate",
            )
        )
    return records


def _wt_sequence_from_residue_map(residue_map_path: Path) -> str:
    rows = pq.read_table(residue_map_path).to_pylist()
    if not rows:
        raise ValueError("residue_map.parquet must contain WT residue rows")
    rows.sort(key=lambda row: int(row["canonical_position"]))
    expected_positions = list(range(1, len(rows) + 1))
    observed_positions = [int(row["canonical_position"]) for row in rows]
    if observed_positions != expected_positions:
        raise ValueError("residue_map.parquet must contain contiguous canonical positions")
    wt = "".join(str(row["wt_aa"]) for row in rows)
    if not wt or any(len(aa) != 1 for aa in wt):
        raise ValueError("residue_map.parquet wt_aa values must be single-letter amino acids")
    return wt


def _apply_canonical_mutations(wt_sequence: str, mutations: Any, candidate_id: str) -> str:
    if not isinstance(mutations, list):
        raise ValueError(f"candidate {candidate_id} canonical_mutations must be a list")
    sequence = list(wt_sequence)
    for label in mutations:
        if not isinstance(label, str):
            raise ValueError(f"candidate {candidate_id} mutation labels must be strings")
        match = _MUTATION_RE.match(label)
        if match is None:
            raise ValueError(f"candidate {candidate_id} has invalid mutation label {label!r}")
        wt_aa, position_text, new_aa = match.groups()
        position = int(position_text)
        if position < 1 or position > len(sequence):
            raise ValueError(f"candidate {candidate_id} mutation {label!r} is outside WT sequence bounds")
        if sequence[position - 1] != wt_aa:
            raise ValueError(
                f"candidate {candidate_id} mutation {label!r} does not match WT residue {sequence[position - 1]!r}"
            )
        sequence[position - 1] = new_aa
    return "".join(sequence)
