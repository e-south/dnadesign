"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/api/contracts.py

Public API request/result contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

Metadata = Mapping[str, object]
_RESERVED_METADATA_KEYS = {"permuter"}


@dataclass(frozen=True)
class NucleotideDmsRequest:
    ref_name: str
    sequence: str
    regions: tuple[tuple[int, int], ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class CodingDnaDmsRequest:
    ref_name: str
    sequence: str
    codon_table: str | Path
    positions: tuple[int, ...] = ()
    alternate_amino_acids: tuple[str, ...] = ()
    codon_policy: Literal["top"] = "top"
    max_variants: int | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class ProteinDmsRequest:
    ref_name: str
    sequence: str
    positions: tuple[int, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class VariantRecord:
    id: str
    ref_name: str
    bio_type: Literal["dna", "protein"]
    sequence: str
    modifications: tuple[str, ...]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSpec:
    id: str
    evaluator: str
    metric: str
    params: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorPlan:
    metrics: tuple[MetricSpec, ...]
    ref_sequence: str | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class DatasetRef:
    dataset_dir: Path
    records_path: Path
    row_count: int
    ref_path: Path | None = None
    ref_aa_path: Path | None = None
    record_path: Path | None = None


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    records_path: Path
    row_count: int
    strict: bool
    metric_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodingDnaDmsVariantMetadata:
    protocol: Literal["coding_dna_dms"]
    aa_pos: int
    aa_wt: str
    aa_alt: str
    codon_index: int
    codon_wt: str
    codon_new: str
    codon_policy: str
    codon_table: str

    @classmethod
    def from_record(cls, record: VariantRecord) -> "CodingDnaDmsVariantMetadata":
        payload = _metadata_mapping(record.metadata.get("permuter"), label=f"{record.id}.metadata.permuter")
        protocol = str(_required(payload, "protocol"))
        if protocol != "coding_dna_dms":
            raise ValueError(f"{record.id}: expected coding_dna_dms metadata, found {protocol!r}")
        aa_pos = _positive_int(payload, "aa_pos", record_id=record.id)
        codon_index = _nonnegative_int(payload, "codon_index", record_id=record.id)
        aa_wt = _amino_acid(payload, "aa_wt", record_id=record.id)
        aa_alt = _amino_acid(payload, "aa_alt", record_id=record.id)
        codon_wt = _codon(payload, "codon_wt", record_id=record.id)
        codon_new = _codon(payload, "codon_new", record_id=record.id)
        codon_policy = str(_required(payload, "codon_policy")).strip()
        codon_table = str(_required(payload, "codon_table")).strip()
        if not codon_policy:
            raise ValueError(f"{record.id}: codon_policy is required")
        if not codon_table:
            raise ValueError(f"{record.id}: codon_table is required")
        return cls(
            protocol="coding_dna_dms",
            aa_pos=aa_pos,
            aa_wt=aa_wt,
            aa_alt=aa_alt,
            codon_index=codon_index,
            codon_wt=codon_wt,
            codon_new=codon_new,
            codon_policy=codon_policy,
            codon_table=codon_table,
        )


@dataclass(frozen=True)
class PermuterResult:
    request_id: str
    ref_name: str
    bio_type: Literal["dna", "protein"]
    reference_sequence: str
    records: tuple[VariantRecord, ...]
    metadata: Metadata = field(default_factory=dict)


def with_permuter_metadata(metadata: Metadata, payload: Mapping[str, object]) -> dict[str, object]:
    """Attach Permuter-owned provenance without silently overwriting caller metadata."""

    out = dict(metadata)
    reserved = sorted(key for key in _RESERVED_METADATA_KEYS if key in out)
    if reserved:
        raise ValueError(f"metadata key(s) are reserved for Permuter provenance: {reserved}")
    out["permuter"] = dict(payload)
    return out


def _metadata_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required(payload: Mapping[str, object], key: str) -> object:
    if key not in payload:
        raise ValueError(f"coding_dna_dms metadata missing required field: {key}")
    return payload[key]


def _positive_int(payload: Mapping[str, object], key: str, *, record_id: str) -> int:
    value = int(_required(payload, key))
    if value < 1:
        raise ValueError(f"{record_id}: {key} must be >= 1")
    return value


def _nonnegative_int(payload: Mapping[str, object], key: str, *, record_id: str) -> int:
    value = int(_required(payload, key))
    if value < 0:
        raise ValueError(f"{record_id}: {key} must be >= 0")
    return value


def _amino_acid(payload: Mapping[str, object], key: str, *, record_id: str) -> str:
    value = str(_required(payload, key)).strip().upper()
    if len(value) != 1 or value not in "ACDEFGHIKLMNPQRSTVWY":
        raise ValueError(f"{record_id}: {key} must be one canonical amino-acid code")
    return value


def _codon(payload: Mapping[str, object], key: str, *, record_id: str) -> str:
    value = str(_required(payload, key)).strip().upper()
    if len(value) != 3 or any(base not in "ACGT" for base in value):
        raise ValueError(f"{record_id}: {key} must be a DNA codon")
    return value
