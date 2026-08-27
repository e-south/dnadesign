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
from dataclasses import dataclass

_SEQUENCE_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX:")


@dataclass(frozen=True)
class OfficialLigandMpnnDesignFasta:
    """Parsed native and designed sequences in upstream sorted-chain order."""

    native_segments: tuple[str, ...]
    designed_segments: tuple[tuple[str, ...], ...]

    @property
    def design_count(self) -> int:
        return len(self.designed_segments)


def parse_official_design_fasta(
    payload: bytes,
    *,
    input_stem: str,
    expected_design_count: int,
) -> int:
    """Validate one official ``run.py`` FASTA and return its design count."""

    return parse_official_design_fasta_records(
        payload,
        input_stem=input_stem,
        expected_design_count=expected_design_count,
    ).design_count


def parse_official_design_fasta_records(
    payload: bytes,
    *,
    input_stem: str,
    expected_design_count: int,
) -> OfficialLigandMpnnDesignFasta:
    """Validate and return exact native/design segments from official ``run.py`` FASTA."""

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
    native_segments = _validate_sequence(native_sequence)
    native_segment_lengths = tuple(len(segment) for segment in native_segments)

    design_records = records[1:]
    observed_count = len(design_records)
    if observed_count != expected_design_count:
        raise ValueError(
            f"official LigandMPNN FASTA expected {expected_design_count} designed records; observed {observed_count}"
        )
    id_pattern = re.compile(rf"^{re.escape(input_stem)}, id=([0-9]+), ")
    observed_ids: list[int] = []
    designed_segments: list[tuple[str, ...]] = []
    for header, sequence in design_records:
        match = id_pattern.match(header)
        if match is None:
            raise ValueError("official LigandMPNN FASTA has an invalid design record header")
        observed_ids.append(int(match.group(1)))
        segments = _validate_sequence(sequence)
        designed_segments.append(segments)
        designed_segment_lengths = tuple(len(segment) for segment in segments)
        if designed_segment_lengths != native_segment_lengths:
            raise ValueError(
                "official LigandMPNN FASTA design must preserve native ordered chain-segment lengths: "
                f"expected {native_segment_lengths}; observed {designed_segment_lengths}"
            )
    expected_ids = list(range(1, expected_design_count + 1))
    if observed_ids != expected_ids:
        raise ValueError(
            f"official LigandMPNN FASTA design record ids must be exactly {expected_ids}; observed {observed_ids}"
        )
    return OfficialLigandMpnnDesignFasta(
        native_segments=native_segments,
        designed_segments=tuple(designed_segments),
    )


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


def _validate_sequence(sequence: str) -> tuple[str, ...]:
    segments = sequence.split(":")
    if (
        not sequence
        or any(not segment for segment in segments)
        or any(residue not in _SEQUENCE_ALPHABET for residue in sequence)
    ):
        raise ValueError("official LigandMPNN FASTA contains an invalid amino-acid sequence")
    return tuple(segments)


__all__ = [
    "OfficialLigandMpnnDesignFasta",
    "parse_official_design_fasta",
    "parse_official_design_fasta_records",
]
