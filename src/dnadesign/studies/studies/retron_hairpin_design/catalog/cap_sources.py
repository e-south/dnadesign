"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/catalog/cap_sources.py

Retron MSD cap source lookup validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CAP_SOURCE_LOOKUP_RELATIVE_PATH = Path("compiler") / "catalog" / "msd_cap_sources.yaml"
_CAP_SOURCE_LABEL_RE = re.compile(
    r"^(?P<construct_id>[A-Za-z0-9_.-]+)-msd\[(?P<payload_id>[A-Za-z0-9_.-]+)\];\s*"
    r"(?P<source_family>[A-Za-z0-9_.]+)-(?P<sequence_chunks>[ACGTacgt]+(?:-[ACGTacgt]+)*)$"
)
_CAP_ID_RE = re.compile(r"^C[A-Za-z0-9_.-]+$")


class RetronMsdCapSourceError(ValueError):
    """Raised when a Retron MSD cap source entry is malformed."""


@dataclass(frozen=True)
class ParsedCapSourceLabel:
    construct_id: str
    payload_id: str
    source_family: str
    sequence_5to3: str


@dataclass(frozen=True)
class RetronMsdCapSource:
    cap_id: str
    source_label: str
    source_construct: str
    payload_id: str
    source_family: str
    sequence_5to3: str


@dataclass(frozen=True)
class RetronMsdCapSourceLookup:
    path: Path
    sources: dict[str, RetronMsdCapSource]


def parse_cap_source_label(label: str) -> ParsedCapSourceLabel:
    text = str(label or "").strip()
    match = _CAP_SOURCE_LABEL_RE.fullmatch(text)
    if match is None:
        raise RetronMsdCapSourceError(
            "Cap source label must match '<construct_id>-msd[<payload>]; <source-family>-<5to3 DNA chunks>'."
        )
    sequence = match.group("sequence_chunks").replace("-", "").upper()
    return ParsedCapSourceLabel(
        construct_id=match.group("construct_id"),
        payload_id=match.group("payload_id"),
        source_family=match.group("source_family"),
        sequence_5to3=sequence,
    )


def load_msd_cap_source_lookup(study_dir: str | Path) -> RetronMsdCapSourceLookup:
    study_path = Path(study_dir).expanduser().resolve()
    lookup_path = study_path / _CAP_SOURCE_LOOKUP_RELATIVE_PATH
    if not lookup_path.is_file():
        raise RetronMsdCapSourceError(f"Retron MSD cap source lookup not found: {lookup_path}")
    try:
        payload = yaml.safe_load(lookup_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RetronMsdCapSourceError(f"Retron MSD cap source lookup is invalid YAML: {lookup_path}") from exc
    if not isinstance(payload, dict):
        raise RetronMsdCapSourceError(f"Retron MSD cap source lookup must be a mapping: {lookup_path}")
    if payload.get("contract") != "retron_msd_cap_source_lookup_v1":
        raise RetronMsdCapSourceError("Retron MSD cap source lookup contract must be retron_msd_cap_source_lookup_v1.")
    sources = _parse_sources(payload.get("sources", {}), lookup_path=lookup_path)
    return RetronMsdCapSourceLookup(path=lookup_path, sources=sources)


def _parse_sources(raw: Any, *, lookup_path: Path) -> dict[str, RetronMsdCapSource]:
    if not isinstance(raw, dict):
        raise RetronMsdCapSourceError(f"Retron MSD cap source lookup field sources must be a mapping: {lookup_path}")
    sources: dict[str, RetronMsdCapSource] = {}
    for raw_cap_id, raw_entry in raw.items():
        cap_id = _cap_id(raw_cap_id)
        if not isinstance(raw_entry, dict):
            raise RetronMsdCapSourceError(f"Cap source entry must be a mapping: sources.{cap_id}")
        source_label = _not_blank(raw_entry.get("source_label"), label=f"sources.{cap_id}.source_label")
        parsed = parse_cap_source_label(source_label)
        source_construct = _not_blank(raw_entry.get("source_construct"), label=f"sources.{cap_id}.source_construct")
        payload_id = _not_blank(raw_entry.get("payload_id"), label=f"sources.{cap_id}.payload_id")
        source_family = _not_blank(raw_entry.get("source_family"), label=f"sources.{cap_id}.source_family")
        sequence_5to3 = _dna(raw_entry.get("sequence_5to3"), label=f"sources.{cap_id}.sequence_5to3")
        if source_construct != parsed.construct_id:
            raise RetronMsdCapSourceError(
                f"sources.{cap_id}.source_construct does not match source_label construct {parsed.construct_id}."
            )
        if payload_id != parsed.payload_id:
            raise RetronMsdCapSourceError(
                f"sources.{cap_id}.payload_id does not match source_label payload {parsed.payload_id}."
            )
        if source_family != parsed.source_family:
            raise RetronMsdCapSourceError(
                f"sources.{cap_id}.source_family does not match source_label family {parsed.source_family}."
            )
        if sequence_5to3 != parsed.sequence_5to3:
            raise RetronMsdCapSourceError(
                f"sources.{cap_id}.sequence_5to3 does not match source_label sequence {parsed.sequence_5to3}."
            )
        sources[cap_id] = RetronMsdCapSource(
            cap_id=cap_id,
            source_label=source_label,
            source_construct=source_construct,
            payload_id=payload_id,
            source_family=source_family,
            sequence_5to3=sequence_5to3,
        )
    return sources


def _cap_id(value: object) -> str:
    text = _not_blank(value, label="cap id")
    if _CAP_ID_RE.fullmatch(text) is None:
        raise RetronMsdCapSourceError(f"Cap id must start with C: {text}")
    return text


def _not_blank(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RetronMsdCapSourceError(f"{label} cannot be empty.")
    return text


def _dna(value: object, *, label: str) -> str:
    text = _not_blank(value, label=label).upper()
    invalid = sorted(set(text) - {"A", "C", "G", "T"})
    if invalid:
        raise RetronMsdCapSourceError(f"{label} contains non-DNA bases: {''.join(invalid)}.")
    return text


__all__ = [
    "ParsedCapSourceLabel",
    "RetronMsdCapSource",
    "RetronMsdCapSourceError",
    "RetronMsdCapSourceLookup",
    "load_msd_cap_source_lookup",
    "parse_cap_source_label",
]
