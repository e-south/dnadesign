"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/ingress/sources.py

Strict raw, text, and FASTA sequence ingress for Junction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dnadesign.junction.contracts.request import MAX_REQUEST_BYTES
from dnadesign.junction.contracts.request.validation import require_dna, require_identifier
from dnadesign.junction.errors import JunctionConfigError

SequenceSourceFormat = Literal["auto", "text", "fasta"]
_FASTA_SUFFIXES = frozenset({".fa", ".fasta", ".ffn", ".fna"})


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    """One named exact DNA sequence before request-level policy is applied."""

    id: str
    sequence: str

    def __post_init__(self) -> None:
        require_identifier(self.id, context="sequence record id")
        require_dna(self.sequence, context=f"sequence record {self.id!r}")


def _normalized_dna(value: str, *, context: str) -> str:
    if not isinstance(value, str):
        raise JunctionConfigError(f"{context} must be a DNA string")
    normalized = "".join(value.split()).upper()
    return require_dna(normalized, context=context)


def sequence_record(sequence: str, *, target_id: str = "target-01") -> SequenceRecord:
    """Normalize one in-memory DNA string into a named ingress record."""

    return SequenceRecord(id=target_id, sequence=_normalized_dna(sequence, context=f"sequence record {target_id!r}"))


def _descriptor_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_source(path: str | Path) -> tuple[Path, str]:
    source_path = Path(path).expanduser()
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(source_path, flags)
    except OSError as exc:
        raise JunctionConfigError(f"Unable to open Junction sequence input: {source_path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise JunctionConfigError(f"Junction sequence input is not a regular file: {source_path}")
        if opened.st_size > MAX_REQUEST_BYTES:
            raise JunctionConfigError(f"Junction sequence input exceeds the {MAX_REQUEST_BYTES}-byte limit")
        chunks: list[bytes] = []
        remaining = MAX_REQUEST_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        observed = os.fstat(descriptor)
        if len(content) > MAX_REQUEST_BYTES or observed.st_size > MAX_REQUEST_BYTES:
            raise JunctionConfigError(f"Junction sequence input exceeds the {MAX_REQUEST_BYTES}-byte limit")
        if _descriptor_identity(opened) != _descriptor_identity(observed):
            raise JunctionConfigError(f"Junction sequence input changed while it was being read: {source_path}")
    except OSError as exc:
        raise JunctionConfigError(f"Unable to read Junction sequence input: {source_path}") from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    try:
        return source_path, content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JunctionConfigError(f"Junction sequence input is not valid UTF-8: {source_path}") from exc


def _parse_fasta(source: str) -> tuple[SequenceRecord, ...]:
    records: list[SequenceRecord] = []
    identifier: str | None = None
    sequence_lines: list[str] = []

    def finish_record() -> None:
        if identifier is None:
            return
        records.append(
            SequenceRecord(
                id=identifier,
                sequence=_normalized_dna("".join(sequence_lines), context=f"FASTA record {identifier!r}"),
            )
        )

    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish_record()
            header = line[1:].strip()
            if not header:
                raise JunctionConfigError(f"FASTA header on line {line_number} is empty")
            identifier = header.split(maxsplit=1)[0]
            sequence_lines = []
            continue
        if identifier is None:
            raise JunctionConfigError(f"FASTA sequence data appears before the first header on line {line_number}")
        sequence_lines.append(line)
    finish_record()
    if not records:
        raise JunctionConfigError("FASTA input must contain at least one record")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise JunctionConfigError("FASTA input must not contain duplicate record identifiers")
    return tuple(records)


def load_sequence_records(
    path: str | Path,
    *,
    source_format: SequenceSourceFormat = "auto",
    target_id: str | None = None,
) -> tuple[SequenceRecord, ...]:
    """Load one text sequence or one or more FASTA records without following symlinks."""

    source_path, source = _read_source(path)
    if source_format not in {"auto", "text", "fasta"}:
        raise JunctionConfigError("Junction sequence format must be auto, text, or fasta")
    resolved_format = source_format
    if resolved_format == "auto":
        resolved_format = "fasta" if source_path.suffix.lower() in _FASTA_SUFFIXES else "text"
    if resolved_format == "fasta":
        if target_id is not None:
            raise JunctionConfigError("--target-id is only valid for raw or text sequence input")
        return _parse_fasta(source)
    return (sequence_record(source, target_id=target_id or "target-01"),)


__all__ = ["SequenceRecord", "SequenceSourceFormat", "load_sequence_records", "sequence_record"]
