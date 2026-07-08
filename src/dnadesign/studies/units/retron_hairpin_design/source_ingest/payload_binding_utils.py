"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_binding_utils.py

Small payload binding-site validation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .models import MsdRegionIngestError

DNA_ALPHABET = frozenset("ACGT")


def span_dict(raw: object, *, default_end: int) -> dict[str, int]:
    if raw is None:
        return {"start": 0, "end": default_end}
    if not isinstance(raw, Mapping):
        raise MsdRegionIngestError("retained_parent_span_0 must be a mapping.")
    start = int(raw.get("start", 0))
    end = int(raw.get("end", default_end))
    if not (0 <= start < end <= default_end):
        raise MsdRegionIngestError(f"Invalid retained_parent_span_0: start={start}, end={end}, parent={default_end}.")
    return {"start": start, "end": end}


def resolve_catalog_path(value: str, *, catalog_path: Path) -> Path:
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates = [catalog_path.parent / path, Path.cwd() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise MsdRegionIngestError(f"Payload binding catalog path does not exist: {value}")


def require_mapping(raw: object, label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise MsdRegionIngestError(f"{label} must be a YAML mapping.")
    return raw


def require_sequence(raw: object, label: str) -> Sequence[Mapping[str, object]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise MsdRegionIngestError(f"{label} must be a YAML sequence.")
    for item in raw:
        if not isinstance(item, Mapping):
            raise MsdRegionIngestError(f"{label} entries must be YAML mappings.")
    return raw


def optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_dna(sequence: str) -> str:
    normalized = "".join(str(sequence).upper().split())
    if not normalized or any(base not in DNA_ALPHABET for base in normalized):
        raise MsdRegionIngestError(f"Expected non-empty DNA sequence, found: {sequence!r}")
    return normalized


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1].upper()


def identity_count(left: str, right: str) -> int:
    return sum(left_base == right_base for left_base, right_base in zip(left, right))


__all__ = [
    "DNA_ALPHABET",
    "identity_count",
    "normalize_dna",
    "optional_text",
    "require_mapping",
    "require_sequence",
    "resolve_catalog_path",
    "reverse_complement",
    "span_dict",
]
