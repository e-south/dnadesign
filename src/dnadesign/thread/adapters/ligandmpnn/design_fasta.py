"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/design_fasta.py

Strict admission of official LigandMPNN design FASTA records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
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
    expected_seed: int,
    expected_temperature: float,
    expected_batch_size: int,
    expected_number_of_batches: int,
) -> int:
    """Validate one official ``run.py`` FASTA and return its design count."""

    return parse_official_design_fasta_records(
        payload,
        input_stem=input_stem,
        expected_design_count=expected_design_count,
        expected_seed=expected_seed,
        expected_temperature=expected_temperature,
        expected_batch_size=expected_batch_size,
        expected_number_of_batches=expected_number_of_batches,
    ).design_count


def parse_official_design_fasta_records(
    payload: bytes,
    *,
    input_stem: str,
    expected_design_count: int,
    expected_seed: int,
    expected_temperature: float,
    expected_batch_size: int,
    expected_number_of_batches: int,
) -> OfficialLigandMpnnDesignFasta:
    """Validate and return exact native/design segments from official ``run.py`` FASTA."""

    if expected_design_count != expected_batch_size * expected_number_of_batches:
        raise ValueError("official LigandMPNN FASTA expected design count does not match requested batch shape")

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("official LigandMPNN FASTA must be UTF-8") from error
    records = _parse_fasta_records(text)
    if not records:
        raise ValueError("official LigandMPNN FASTA contains no records")

    native_header, native_sequence = records[0]
    native_metadata = _parse_header_metadata(native_header, input_stem=input_stem, require_design_id=False)
    _validate_execution_metadata(
        native_metadata,
        expected_seed=expected_seed,
        expected_temperature=expected_temperature,
        expected_batch_size=expected_batch_size,
        expected_number_of_batches=expected_number_of_batches,
    )
    native_segments = _validate_sequence(native_sequence)
    native_segment_lengths = tuple(len(segment) for segment in native_segments)

    design_records = records[1:]
    observed_count = len(design_records)
    if observed_count != expected_design_count:
        raise ValueError(
            f"official LigandMPNN FASTA expected {expected_design_count} designed records; observed {observed_count}"
        )
    observed_ids: list[int] = []
    designed_segments: list[tuple[str, ...]] = []
    for header, sequence in design_records:
        metadata = _parse_header_metadata(header, input_stem=input_stem, require_design_id=True)
        design_id = int(metadata.pop("id"))
        _validate_execution_metadata(
            metadata,
            expected_seed=expected_seed,
            expected_temperature=expected_temperature,
            expected_batch_size=None,
            expected_number_of_batches=None,
        )
        observed_ids.append(design_id)
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


def _parse_header_metadata(
    header: str,
    *,
    input_stem: str,
    require_design_id: bool,
) -> dict[str, str]:
    parts = header.split(", ")
    if not parts or parts[0] != input_stem:
        raise ValueError("official LigandMPNN FASTA has an invalid record header")
    metadata: dict[str, str] = {}
    for item in parts[1:]:
        if "=" not in item:
            raise ValueError("official LigandMPNN FASTA header metadata must use key=value fields")
        key, value = item.split("=", 1)
        if not key or not value or key in metadata:
            raise ValueError("official LigandMPNN FASTA header metadata is incomplete or duplicated")
        metadata[key] = value
    if not require_design_id and "id" in metadata:
        raise ValueError("official LigandMPNN FASTA native record must not declare a design id")
    if require_design_id and re.fullmatch(r"[0-9]+", metadata.get("id", "")) is None:
        raise ValueError("official LigandMPNN FASTA has an invalid design record header")
    return metadata


def _validate_execution_metadata(
    metadata: dict[str, str],
    *,
    expected_seed: int,
    expected_temperature: float,
    expected_batch_size: int | None,
    expected_number_of_batches: int | None,
) -> None:
    try:
        observed_seed = int(metadata["seed"])
    except (KeyError, ValueError) as error:
        raise ValueError("official LigandMPNN FASTA header is missing a valid seed") from error
    try:
        observed_temperature = float(metadata["T"])
    except (KeyError, ValueError) as error:
        raise ValueError("official LigandMPNN FASTA header is missing a valid temperature") from error
    if observed_seed != expected_seed:
        raise ValueError(
            f"official LigandMPNN FASTA seed {observed_seed} does not match requested seed {expected_seed}"
        )
    if not math.isfinite(observed_temperature) or observed_temperature != float(expected_temperature):
        raise ValueError(
            "official LigandMPNN FASTA temperature "
            f"{observed_temperature!r} does not match requested temperature {expected_temperature}"
        )
    if expected_batch_size is not None:
        observed_batch_size = _parse_positive_header_integer(metadata, "batch_size")
        if observed_batch_size != expected_batch_size:
            raise ValueError(
                "official LigandMPNN FASTA batch_size "
                f"{observed_batch_size} does not match requested batch_size {expected_batch_size}"
            )
    if expected_number_of_batches is not None:
        observed_number_of_batches = _parse_positive_header_integer(metadata, "number_of_batches")
        if observed_number_of_batches != expected_number_of_batches:
            raise ValueError(
                "official LigandMPNN FASTA number_of_batches "
                f"{observed_number_of_batches} does not match requested number_of_batches {expected_number_of_batches}"
            )


def _parse_positive_header_integer(metadata: dict[str, str], field_name: str) -> int:
    value = metadata.get(field_name, "")
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError(f"official LigandMPNN FASTA header is missing a valid {field_name}")
    return int(value)


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
