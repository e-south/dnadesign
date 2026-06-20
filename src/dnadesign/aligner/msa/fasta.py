"""FASTA reading and writing helpers for generic MSA workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dnadesign.aligner.msa.validation import validate_fasta_records


def load_fasta_records(path: Path, *, alphabet: str = "protein", allow_gaps: bool = False) -> dict[str, str]:
    """Load FASTA records into an insertion-ordered mapping."""

    if not path.exists():
        raise FileNotFoundError(path)

    records: dict[str, str] = {}
    current_id: str | None = None
    current_chunks: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records[current_id] = "".join(current_chunks).upper()
            current_id = line[1:].split()[0]
            if not current_id:
                raise ValueError(f"FASTA record id is empty in {path}")
            if current_id in records:
                raise ValueError(f"duplicate FASTA record id {current_id!r} in {path}")
            current_chunks = []
        elif current_id is None:
            raise ValueError(f"FASTA sequence data appears before a record id in {path}")
        else:
            current_chunks.append(line)

    if current_id is not None:
        records[current_id] = "".join(current_chunks).upper()

    validate_fasta_records(records, alphabet=alphabet, allow_gaps=allow_gaps)
    return records


def write_fasta_records(path: Path, records: Mapping[str, str]) -> None:
    """Write FASTA records with one-line sequences."""

    if not records:
        raise ValueError("Cannot write empty FASTA records")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record_id, sequence in records.items():
        if not record_id:
            raise ValueError("FASTA record id must be non-empty")
        lines.append(f">{record_id}")
        lines.append(sequence.upper())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
