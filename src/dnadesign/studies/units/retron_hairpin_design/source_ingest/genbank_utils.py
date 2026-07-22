"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/genbank_utils.py

Small utilities for MSD-region GenBank source ingest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

import yaml
from Bio.SeqRecord import SeqRecord

from .models import MsdRegionIngestError

VARIANT_RE = re.compile(r"(?:pES-)?retron[-_ ]?(\d+)|msd[-_ ]?retron[-_ ]?(\d+)", re.IGNORECASE)
DISPLAY_ID_RE = re.compile(r"pES-retron-(\d+)", re.IGNORECASE)


def variant_id(record: SeqRecord) -> str | None:
    match = VARIANT_RE.search(record_text(record))
    if match is None:
        return None
    return f"retron{match.group(1) or match.group(2)}"


def variant_id_for_existing_row(row: Mapping[str, str], display_by_variant_key: Mapping[str, str]) -> str | None:
    construct_id = str(row.get("construct_id") or "")
    trim_match = re.fullmatch(r"pES-tetr-(.+)", construct_id)
    if trim_match is not None:
        display_id = display_by_variant_key.get(trim_match.group(1))
        if display_id:
            display_match = DISPLAY_ID_RE.fullmatch(display_id)
            if display_match is not None:
                return f"retron{display_match.group(1)}"
    display_match = DISPLAY_ID_RE.search(construct_id)
    if display_match is not None:
        return f"retron{display_match.group(1)}"
    return None


def record_text(record: SeqRecord) -> str:
    parts = [record.id, record.name, record.description]
    for feature in record.features:
        for values in feature.qualifiers.values():
            parts.extend(str(value) for value in values)
    return "\n".join(parts)


def simple_span(feature: Any) -> tuple[int, int]:
    if len(getattr(feature.location, "parts", ())) > 1:
        raise MsdRegionIngestError("Compound GenBank features are not supported for MSD-region ingest.")
    start = int(feature.location.start)
    end = int(feature.location.end)
    if end < start:
        raise MsdRegionIngestError(f"Invalid feature span {start}:{end}.")
    return start, end


def qualifier_values(feature: Any, key: str) -> list[str]:
    return [str(value) for value in feature.qualifiers.get(key, [])]


def variant_number(variant_id: str) -> str:
    match = re.fullmatch(r"retron(\d+)", variant_id)
    if match is None:
        raise MsdRegionIngestError(f"Invalid retron variant id: {variant_id}")
    return match.group(1)


def variant_sort_key(variant_id: str) -> int:
    return int(variant_number(variant_id))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1].upper()


def write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    path.write_text(allowlist_checksum_lines(text), encoding="utf-8")


def allowlist_checksum_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        key = line.split(":", 1)[0].strip()
        if key.endswith("sha256") and "pragma: allowlist secret" not in line:
            line = f"{line}  # pragma: allowlist secret"
        lines.append(line)
    return "\n".join(lines) + "\n"


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


__all__ = [
    "DISPLAY_ID_RE",
    "VARIANT_RE",
    "allowlist_checksum_lines",
    "qualifier_values",
    "record_text",
    "relative_to",
    "reverse_complement",
    "sha256_file",
    "sha256_text",
    "simple_span",
    "variant_id",
    "variant_id_for_existing_row",
    "variant_number",
    "variant_sort_key",
    "write_yaml",
]
