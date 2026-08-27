"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/design_fasta.py

Strict admission of official LigandMPNN design FASTA records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

_SEQUENCE_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX:")


def parse_official_design_fasta(
    payload: bytes,
    *,
    input_stem: str,
    expected_design_count: int,
) -> int:
    """Validate one official ``run.py`` FASTA and return its design count."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("official LigandMPNN FASTA must be UTF-8") from error
    records = _parse_fasta_records(text)
    if not records:
        raise ValueError("official LigandMPNN FASTA contains no records")

    native_header, native_sequence = records[0]
    if not native_header.startswith(f"{input_stem}, T="):
        raise ValueError("official LigandMPNN FASTA has an invalid native record header")
    _validate_sequence(native_sequence)

    design_records = records[1:]
    observed_count = len(design_records)
    if observed_count != expected_design_count:
        raise ValueError(
            f"official LigandMPNN FASTA expected {expected_design_count} designed records; observed {observed_count}"
        )
    id_pattern = re.compile(rf"^{re.escape(input_stem)}, id=([0-9]+), ")
    observed_ids: list[int] = []
    for header, sequence in design_records:
        match = id_pattern.match(header)
        if match is None:
            raise ValueError("official LigandMPNN FASTA has an invalid design record header")
        observed_ids.append(int(match.group(1)))
        _validate_sequence(sequence)
    expected_ids = list(range(1, expected_design_count + 1))
    if observed_ids != expected_ids:
        raise ValueError(
            f"official LigandMPNN FASTA design record ids must be exactly {expected_ids}; observed {observed_ids}"
        )
    return observed_count


def _parse_fasta_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_lines: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence_lines)))
            header = line[1:]
            sequence_lines = []
            if not header:
                raise ValueError(f"official LigandMPNN FASTA has an empty header at line {line_number}")
            continue
        if header is None:
            raise ValueError(f"official LigandMPNN FASTA has sequence data before a header at line {line_number}")
        sequence_lines.append(line)
    if header is not None:
        records.append((header, "".join(sequence_lines)))
    return records


def _validate_sequence(sequence: str) -> None:
    segments = sequence.split(":")
    if (
        not sequence
        or any(not segment for segment in segments)
        or any(residue not in _SEQUENCE_ALPHABET for residue in sequence)
    ):
        raise ValueError("official LigandMPNN FASTA contains an invalid amino-acid sequence")


__all__ = ["parse_official_design_fasta"]
