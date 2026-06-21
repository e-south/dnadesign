"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/fasta.py

FASTA parsing and writing for Eco1 provider-source acquisition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

_PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")


def parse_provider_fasta(text: str, *, requested_accessions: Sequence[str]) -> dict[str, str]:
    """Parse provider FASTA text and map headers back to requested accessions."""

    requested = tuple(requested_accessions)
    requested_set = set(requested)
    records: dict[str, str] = {}
    current_accession: str | None = None
    current_chunks: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_accession is not None:
                records[current_accession] = _validated_sequence("".join(current_chunks), current_accession)
            current_accession = _accession_from_header(line[1:], requested=requested, requested_set=requested_set)
            if current_accession in records:
                raise ValueError(f"duplicate FASTA record for requested accession {current_accession!r}")
            current_chunks = []
        elif current_accession is None:
            raise ValueError("FASTA sequence data appears before a record header")
        else:
            current_chunks.append(line)
    if current_accession is not None:
        records[current_accession] = _validated_sequence("".join(current_chunks), current_accession)
    return records


def write_provider_fasta(path: Path, records: Mapping[str, str]) -> None:
    """Write provider FASTA records with stable source accession headers."""

    if not records:
        raise ValueError(f"provider source FASTA would be empty: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for accession, sequence in records.items():
        lines.extend([f">{accession}", sequence.upper()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _accession_from_header(header: str, *, requested: Sequence[str], requested_set: set[str]) -> str:
    token = header.split()[0]
    if token in requested_set:
        return token
    for accession in requested:
        if token.startswith(f"{accession}|"):
            return accession
    raise ValueError(f"provider FASTA returned unexpected accession header {header!r}")


def _validated_sequence(sequence: str, accession: str) -> str:
    normalized = sequence.upper()
    if not normalized:
        raise ValueError(f"provider FASTA record {accession!r} has an empty sequence")
    invalid = sorted({character for character in normalized if character not in _PROTEIN_ALPHABET})
    if invalid:
        raise ValueError(f"invalid protein character {invalid[0]!r} in provider FASTA record {accession!r}")
    return normalized
