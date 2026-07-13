"""Protein and provenance validation for the Eco1 Twist handoff."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    EXPECTED_SELECTED_POLICY_COUNTS,
    SELECTED_PANEL_SIZE,
)

MUTATION_RE = re.compile(r"^([A-Z])([1-9][0-9]*)([A-Z])$")
FORBIDDEN_SITES = ("GGTCTC", "GAGACC", "CGTCTC", "GAGACG")


def read_unique_fasta(path: Path) -> dict[str, str]:
    """Read one uppercase protein sequence per unique FASTA identifier."""

    records: dict[str, str] = {}
    for record in SeqIO.parse(path, "fasta"):
        if record.id in records:
            raise ValueError(f"foldcheck FASTA contains duplicate ID {record.id!r}")
        records[record.id] = str(record.seq).upper()
    if not records:
        raise ValueError("foldcheck FASTA is empty")
    return records


def validate_wild_type(record: SeqRecord) -> tuple[str, str]:
    """Return the authoritative WT DNA and protein after strict validation."""

    dna = str(record.seq).upper()
    if len(dna) != 963 or set(dna) - set("ACGT"):
        raise ValueError("authoritative WT CDS GenBank must contain one unambiguous 963-bp sequence")
    translation = str(record.seq.translate())
    if len(translation) != 321 or not translation.endswith("*") or "*" in translation[:-1]:
        raise ValueError("authoritative WT CDS must encode 320 amino acids followed by one stop codon")
    cds_features = [feature for feature in record.features if feature.type == "CDS"]
    if len(cds_features) != 1 or int(cds_features[0].location.start) != 0 or int(cds_features[0].location.end) != 963:
        raise ValueError("authoritative WT GenBank must contain one full-length CDS feature")
    annotated = cds_features[0].qualifiers.get("translation", [None])[0]
    if annotated is None or str(annotated).rstrip("*") != translation[:-1]:
        raise ValueError("authoritative WT CDS feature translation does not match its sequence")
    return dna, translation[:-1]


def validate_panel_shape(rows: list[dict[str, Any]]) -> None:
    """Require the declared eight-sequence selected panel."""

    if len(rows) != SELECTED_PANEL_SIZE:
        raise ValueError(f"candidate selection panel must contain exactly {SELECTED_PANEL_SIZE} rows")
    if any(row.get("eligible_for_handoff") is not True for row in rows):
        raise ValueError("every panel row must be selected and eligible for handoff")
    policy_counts = Counter(str(row.get("policy_id")) for row in rows)
    if dict(policy_counts) != EXPECTED_SELECTED_POLICY_COUNTS:
        raise ValueError(f"candidate selection panel has invalid policy counts: {dict(policy_counts)}")
    for field in ("candidate_id", "selection_slot", "sequence_hash"):
        values = [str(row.get(field)) for row in rows]
        if len(set(values)) != SELECTED_PANEL_SIZE or "None" in values:
            raise ValueError(f"candidate selection panel requires {SELECTED_PANEL_SIZE} unique {field} values")
    selected_rows(rows)


def selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all selected sequences in their declared order."""

    ranks = [row.get("selection_rank") for row in rows]
    if any(rank is None for rank in ranks) or sorted(int(rank) for rank in ranks) != list(
        range(1, SELECTED_PANEL_SIZE + 1)
    ):
        raise ValueError("selected rows must have unique contiguous ranks")
    return sorted(rows, key=lambda row: int(row["selection_rank"]))


def unique_rows(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    """Index rows by a required unique field."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key))
        if value in result:
            raise ValueError(f"{label} contains duplicate {key} {value!r}")
        result[value] = row
    return result


def validate_mutations(candidate_id: str, raw_tokens: Any, wt: str, candidate: str) -> list[str]:
    """Validate compact amino-acid substitutions against WT and candidate proteins."""

    if not isinstance(raw_tokens, list) or not all(isinstance(token, str) for token in raw_tokens):
        raise ValueError(f"candidate {candidate_id} canonical_mutations must be a list of strings")
    expected: list[str] = []
    seen_positions: set[int] = set()
    for token in raw_tokens:
        match = MUTATION_RE.fullmatch(token)
        if match is None:
            raise ValueError(f"candidate {candidate_id} has non-canonical mutation token {token!r}")
        ref, position_text, alt = match.groups()
        position = int(position_text)
        if position > len(wt) or position in seen_positions:
            raise ValueError(f"candidate {candidate_id} has invalid or duplicate mutation position {position}")
        seen_positions.add(position)
        if wt[position - 1] != ref or candidate[position - 1] != alt or ref == alt:
            raise ValueError(f"candidate {candidate_id} mutation token {token!r} disagrees with WT/candidate")
        expected.append(token)
    observed = [
        f"{ref}{index}{alt}" for index, (ref, alt) in enumerate(zip(wt, candidate, strict=True), 1) if ref != alt
    ]
    if expected != observed:
        raise ValueError(f"candidate {candidate_id} canonical mutation tokens do not exactly describe WT/candidate")
    return expected


__all__ = [
    "FORBIDDEN_SITES",
    "MUTATION_RE",
    "read_unique_fasta",
    "selected_rows",
    "unique_rows",
    "validate_mutations",
    "validate_panel_shape",
    "validate_wild_type",
]
