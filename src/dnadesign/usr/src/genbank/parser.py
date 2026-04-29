"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/genbank/parser.py

Biopython-backed GenBank parser with fidelity-preserving location capture.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from Bio import SeqIO
from Bio.SeqFeature import CompoundLocation

from .models import ParsedFeatureInterval, ParsedGenBankFeature, ParsedGenBankRecord, ParsedQualifier, RoleHintRule


class GenBankParser(Protocol):
    def parse_file(
        self,
        path: Path,
        *,
        role_hint_rules: list[RoleHintRule] | None = None,
    ) -> list[ParsedGenBankRecord]: ...


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _preferred_label(feature) -> str | None:
    for key in ("label", "gene", "locus_tag", "standard_name", "note", "product"):
        raw_values = feature.qualifiers.get(key)
        if not raw_values:
            continue
        first = str(raw_values[0]).strip()
        if first:
            return first
    return None


def _is_exact_position(position) -> bool:
    return type(position).__name__ == "ExactPosition"


def _location_parts(location) -> list:
    if isinstance(location, CompoundLocation):
        return list(location.parts)
    return [location]


def _parsed_intervals(location) -> tuple[list[ParsedFeatureInterval], bool]:
    intervals: list[ParsedFeatureInterval] = []
    is_fuzzy = False
    for part in _location_parts(location):
        start = int(part.start)
        end = int(part.end)
        partial = not (_is_exact_position(part.start) and _is_exact_position(part.end))
        is_fuzzy = is_fuzzy or partial
        intervals.append(
            ParsedFeatureInterval(
                start_0=start,
                end_0=end,
                strand=part.strand,
                partial=partial,
            )
        )
    return intervals, is_fuzzy


def _location_kind(location) -> str:
    if isinstance(location, CompoundLocation):
        operator = str(getattr(location, "operator", "") or "").strip() or "compound"
        return f"compound_{operator}"
    return "interval"


def _role_hint(label: str | None, *, rules: list[RoleHintRule]) -> str | None:
    for rule in rules:
        if rule.matches(label):
            return rule.role_hint
    return None


class BiopythonGenBankParser:
    parser_name = "biopython"

    def parse_file(self, path: Path, *, role_hint_rules: list[RoleHintRule] | None = None) -> list[ParsedGenBankRecord]:
        rules = list(role_hint_rules or [])
        source_path = Path(path)
        source_sha256 = _sha256_file(source_path)
        parsed_records: list[ParsedGenBankRecord] = []
        for record in SeqIO.parse(str(source_path), "genbank"):
            sequence = str(record.seq).upper()
            features: list[ParsedGenBankFeature] = []
            for feature_order, feature in enumerate(record.features):
                label = _preferred_label(feature)
                intervals, is_fuzzy = _parsed_intervals(feature.location)
                start_0 = min((interval.start_0 for interval in intervals), default=None)
                end_0 = max((interval.end_0 for interval in intervals), default=None)
                confidence = "unknown" if not intervals else ("low" if is_fuzzy else "high")
                qualifiers = [
                    ParsedQualifier(key=str(key), value=str(value))
                    for key, values in feature.qualifiers.items()
                    for value in values
                ]
                feature_id_source = "|".join(
                    [
                        str(record.id or ""),
                        str(feature_order),
                        str(feature.type or ""),
                        str(feature.location),
                        str(label or ""),
                    ]
                )
                features.append(
                    ParsedGenBankFeature(
                        feature_id=f"gbf_{hashlib.sha1(feature_id_source.encode('utf-8')).hexdigest()[:12]}",
                        feature_order=feature_order,
                        feature_type=str(feature.type or "unknown"),
                        label=label,
                        role_hint=_role_hint(label, rules=rules),
                        location_raw=str(feature.location),
                        location_kind=_location_kind(feature.location),
                        start_0=start_0,
                        end_0=end_0,
                        strand=getattr(feature.location, "strand", None),
                        intervals_0=intervals,
                        is_fuzzy=is_fuzzy,
                        is_compound=isinstance(feature.location, CompoundLocation),
                        qualifiers=qualifiers,
                        confidence=confidence,
                        source=self.parser_name,
                    )
                )
            parsed_records.append(
                ParsedGenBankRecord(
                    source_file=str(source_path),
                    source_sha256=source_sha256,
                    record_id=str(record.id or "").strip() or None,
                    record_name=str(record.name or "").strip() or None,
                    description=str(record.description or "").strip() or None,
                    topology=str(record.annotations.get("topology") or "").strip() or None,
                    molecule_type=str(record.annotations.get("molecule_type") or "").strip() or None,
                    sequence_region_start_0=0,
                    sequence_region_end_0=len(sequence),
                    sequence=sequence,
                    features=features,
                )
            )
        return parsed_records


__all__ = ["BiopythonGenBankParser", "GenBankParser"]
