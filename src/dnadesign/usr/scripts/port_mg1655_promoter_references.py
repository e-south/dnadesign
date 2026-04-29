"""
Curated projected GenBank port for promoter reference controls.

This one-off USR migration helper builds ``usr_promoter_references`` from
source-backed GenBank inputs. It strips cloning/primer flanks from archived
MG1655 noncoding records, imports synthetic promoter standards as promoter
inserts, and writes Construct-facing USR rows with source annotations, strength
metadata, and sequence views.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src.contracts import SchemaError, compute_id
from dnadesign.usr.src.genbank.importer import _copy_source_artifact, _source_artifact_uri
from dnadesign.usr.src.genbank.models import ParsedGenBankFeature, ParsedGenBankRecord, RoleHintRule
from dnadesign.usr.src.genbank.parser import BiopythonGenBankParser
from dnadesign.usr.src.registry import (
    arrow_type_from_str,
    ensure_registry_entries,
    ensure_sequence_contract_namespaces,
    load_registry,
    promoter_standard_entry,
    registry_entry,
)
from dnadesign.usr.src.sequence_views import SequenceViewRecord, write_sequence_views
from dnadesign.usr.src.storage.parquet import now_utc

DEFAULT_OUTPUT_DATASET = "usr_promoter_references"
DEFAULT_LEGACY_DATASET = "usr_mg1655_promoter_controls"
PORT_RUN_ID = "promoter_reference_projected_genbank_port"
MG1655_GENBANK_ARTIFACT_SOURCE = "archived/MG1655_noncoding_set"
SYNTHETIC_STANDARDS_ARTIFACT_SOURCE = "archived/synthetic_promoter_standards"
DEFAULT_EXPECTED_GENBANK_COUNT = 48
PROMOTER_STANDARD_NAMESPACE = "promoter_standard"
J23105_LABEL = "J23105"
J23105_SEQUENCE = "TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC"
_STRICT_DNA = set("ACGT")

_REGULATOR_LABELS = (
    "AlaS-",
    "AraC+",
    "AraC-",
    "ArcA-",
    "ArgR-",
    "BaeR+",
    "CpxR+",
    "CpxR-",
    "Cra+",
    "Cra-",
    "CRP+",
    "FadR-",
    "Fis+",
    "FNR+",
    "IclR-",
    "IHF+",
    "IHF-",
    "LexA-",
    "MarA+",
    "MarR-",
    "Nac-",
    "OmpR+",
    "OxyR+",
    "PdhR+",
    "PhoB+",
    "PspF+",
    "Rob+",
    "SoxR+",
    "SoxS+",
    "SoxS-",
)

ROLE_HINT_RULES = [
    RoleHintRule(match_label="-35", role_hint="sigma70_minus35"),
    RoleHintRule(match_label="-10", role_hint="sigma70_minus10"),
    RoleHintRule(match_any_label=list(_REGULATOR_LABELS), role_hint="TFBS"),
]

_FILENAME_LABEL_ALIASES = {
    "spyp-upstream-of-spy.gb": ("spyp", ("spyP", "spyP_MG1655")),
    "soxsp-upstream-soxs.gb": ("soxSp", ("soxS",)),
    "sulap-upstream-sula.gb": ("sulAp", ("sulA",)),
}


@dataclass(frozen=True)
class PromoterReference:
    label: str
    aliases: tuple[str, ...]
    sequence: str
    source_file: str
    base_source: str
    source_ref: str
    source_sha256: str
    record_id: str | None
    record_name: str | None
    description: str | None
    topology: str | None
    molecule_type: str | None
    source_interval_start_0: int
    source_interval_end_0: int
    source_intervals_0: tuple[dict[str, object], ...]
    source_feature_id: str
    source_feature_label: str | None
    focal_confidence: str
    seq_annot_features: tuple[dict[str, object], ...]
    features_retained: tuple[dict[str, object], ...]
    features_clipped: tuple[dict[str, object], ...]
    features_lost: tuple[dict[str, object], ...]
    derived_operation: str
    derived_focal_rule: str
    derived_created_by: str
    derivation_spec_id: str
    construct_seed_role: str
    construct_seed_manifest_id: str
    standard_metadata: PromoterStandardMetadata | None = None

    @property
    def id(self) -> str:
        return compute_id("dna", self.sequence)


@dataclass(frozen=True)
class LegacyReference:
    label: str
    aliases: tuple[str, ...]
    sequence: str
    source_dataset: str

    @property
    def id(self) -> str:
        return compute_id("dna", self.sequence)


@dataclass(frozen=True)
class PromoterStandardMetadata:
    collection_id: str
    promoter_id: str
    display_name: str
    role: str
    strength_metric: str
    strength_value: str
    strength_value_numeric: float | None
    strength_reference: str
    source_record: str
    notes: str


@dataclass(frozen=True)
class PromoterReferencePlan:
    promoters: tuple[PromoterReference, ...]
    legacy_references: tuple[LegacyReference, ...]
    archive_dir: str
    synthetic_standards_dir: str | None
    legacy_dataset: str | None

    def summary(self) -> dict[str, object]:
        synthetic_rows = [row for row in self.promoters if row.standard_metadata is not None]
        return {
            "archive_dir": self.archive_dir,
            "synthetic_standards_dir": self.synthetic_standards_dir,
            "legacy_dataset": self.legacy_dataset,
            "genbank_records": len(self.promoters),
            "synthetic_standard_records": len(synthetic_rows),
            "legacy_references": len(self.legacy_references),
            "total_rows": len(self.promoters) + len(self.legacy_references),
            "labels": [row.label for row in [*self.promoters, *self.legacy_references]],
        }


@dataclass(frozen=True)
class WriteResult:
    dataset: str
    dataset_dir: str
    rows_written: int
    genbank_rows_written: int
    legacy_rows_written: int
    seq_annot_overlay_rows: int
    derived_overlay_rows: int
    label_overlay_rows: int
    construct_seed_overlay_rows: int
    promoter_standard_overlay_rows: int
    sequence_views_written: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_archive_dir() -> Path:
    return _repo_root().parent / "archived" / "MG1655_noncoding_set"


def _default_synthetic_standards_dir() -> Path:
    return _repo_root().parent / "archived" / "synthetic_promoter_standards"


def _default_usr_root() -> Path:
    return _repo_root() / "src" / "dnadesign" / "usr" / "datasets"


def _ensure_promoter_reference_namespaces(root: Path) -> None:
    ensure_sequence_contract_namespaces(root)
    ensure_registry_entries(root, entries=(promoter_standard_entry(),))


def _normalize_sequence(sequence: str) -> str:
    normalized = "".join(ch for ch in str(sequence).upper().replace("U", "T") if ch.isalpha())
    if not normalized or set(normalized) - _STRICT_DNA:
        raise SchemaError("Promoter reference sequences must be strict A/C/G/T DNA.")
    return normalized.lower()


def _dedupe_aliases(values: Iterable[str | None], *, primary: str) -> tuple[str, ...]:
    seen = {primary}
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _label_from_source_feature(path: Path, feature: ParsedGenBankFeature) -> tuple[str, tuple[str, ...]]:
    override = _FILENAME_LABEL_ALIASES.get(path.name.lower())
    if override is not None:
        return override
    raw_label = str(feature.label or path.stem).strip()
    label = raw_label.split("(", 1)[0].strip()
    if label.casefold().startswith("pred. "):
        label = label[6:].strip()
    if not label:
        raise SchemaError(f"Cannot derive a promoter label from '{path}'.")
    return label, ()


def _is_primer_feature(feature: ParsedGenBankFeature) -> bool:
    label = str(feature.label or "").casefold()
    feature_type = str(feature.feature_type or "").casefold()
    return "primer" in label or "primer" in feature_type


def _is_precise_single_interval(feature: ParsedGenBankFeature) -> bool:
    return (
        feature.start_0 is not None
        and feature.end_0 is not None
        and not feature.is_fuzzy
        and len(feature.intervals_0) == 1
    )


def _select_upstream_feature(record: ParsedGenBankRecord, *, path: Path) -> ParsedGenBankFeature:
    candidates = [
        feature
        for feature in record.features
        if feature.feature_type == "misc_feature"
        and "upstream" in str(feature.label or "").casefold()
        and not _is_primer_feature(feature)
        and _is_precise_single_interval(feature)
    ]
    if not candidates:
        raise SchemaError(f"GenBank source '{path}' has no precise non-primer upstream misc_feature to project.")
    candidates.sort(key=lambda feature: int(feature.end_0 or 0) - int(feature.start_0 or 0), reverse=True)
    if len(candidates) > 1:
        longest = int(candidates[0].end_0 or 0) - int(candidates[0].start_0 or 0)
        tied = [feature for feature in candidates if int(feature.end_0 or 0) - int(feature.start_0 or 0) == longest]
        if len(tied) > 1:
            labels = ", ".join(str(feature.label or feature.feature_id) for feature in tied[:5])
            raise SchemaError(f"GenBank source '{path}' has ambiguous upstream features: {labels}.")
    return candidates[0]


def _select_full_span_promoter_feature(record: ParsedGenBankRecord, *, path: Path) -> ParsedGenBankFeature:
    candidates = [
        feature
        for feature in record.features
        if feature.feature_type == "promoter"
        and _is_precise_single_interval(feature)
        and feature.start_0 == 0
        and feature.end_0 == len(record.sequence)
    ]
    if len(candidates) != 1:
        labels = ", ".join(str(feature.label or feature.feature_id) for feature in candidates[:5])
        raise SchemaError(
            f"Synthetic standard '{path}' must have exactly one full-span promoter feature; found {labels}."
        )
    return candidates[0]


def _qualifier_note_map(feature: ParsedGenBankFeature) -> dict[str, str]:
    notes: dict[str, str] = {}
    for qualifier in feature.qualifiers:
        if qualifier.key != "note":
            continue
        key, separator, value = qualifier.value.partition("=")
        if not separator:
            continue
        notes[key] = value
    return notes


def _strength_value_numeric(value: str) -> float | None:
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise SchemaError(f"Promoter standard strength_value must be numeric or NA, got '{value}'.") from exc


def _read_csv_rows(path: Path, *, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required promoter standard table does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        observed = set(reader.fieldnames or ())
        missing = sorted(required_columns - observed)
        if missing:
            raise SchemaError(f"Promoter standard table '{path}' is missing required columns: {missing}")
        return [{key: str(value or "").strip() for key, value in row.items()} for row in reader]


def _synthetic_export_policy(standards_dir: Path) -> dict[tuple[str, str], bool]:
    rows = _read_csv_rows(
        standards_dir / "data" / "promoter_export_policy.csv",
        required_columns={"collection_id", "promoter_id", "export_to_genbank", "exclusion_reason"},
    )
    policy: dict[tuple[str, str], bool] = {}
    for row in rows:
        key = (row["collection_id"], row["promoter_id"])
        if key in policy:
            raise SchemaError(f"Duplicate promoter export policy key: {key}")
        raw = row["export_to_genbank"].casefold()
        if raw not in {"true", "false"}:
            raise SchemaError(f"Invalid export_to_genbank value for {key}: {row['export_to_genbank']}")
        policy[key] = raw == "true"
    return policy


def _synthetic_promoter_rows(standards_dir: Path) -> list[dict[str, str]]:
    rows = _read_csv_rows(
        standards_dir / "data" / "promoters.csv",
        required_columns={
            "collection_id",
            "promoter_id",
            "display_name",
            "role",
            "sequence",
            "strength_metric",
            "strength_value",
            "strength_reference",
            "source_record",
            "notes",
        },
    )
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["collection_id"], row["promoter_id"])
        if key in seen:
            raise SchemaError(f"Duplicate synthetic promoter key: {key}")
        seen.add(key)
    return rows


def _standard_metadata(row: dict[str, str]) -> PromoterStandardMetadata:
    return PromoterStandardMetadata(
        collection_id=row["collection_id"],
        promoter_id=row["promoter_id"],
        display_name=row["display_name"],
        role=row["role"],
        strength_metric=row["strength_metric"],
        strength_value=row["strength_value"],
        strength_value_numeric=_strength_value_numeric(row["strength_value"]),
        strength_reference=row["strength_reference"],
        source_record=row["source_record"],
        notes=row["notes"],
    )


def _interval_dict(start_0: int, end_0: int, strand: int | None, partial: bool = False) -> dict[str, object]:
    return {"start_0": start_0, "end_0": end_0, "strand": strand, "partial": partial}


def _project_feature(
    feature: ParsedGenBankFeature,
    *,
    insert_start_0: int,
    insert_end_0: int,
) -> dict[str, object] | None:
    projected_intervals: list[dict[str, object]] = []
    for interval in feature.intervals_0:
        start = max(interval.start_0, insert_start_0)
        end = min(interval.end_0, insert_end_0)
        if end <= start:
            continue
        projected_intervals.append(
            _interval_dict(
                start - insert_start_0,
                end - insert_start_0,
                interval.strand,
                interval.partial,
            )
        )
    if not projected_intervals:
        return None
    return {
        "feature_id": feature.feature_id,
        "feature_order": feature.feature_order,
        "feature_type": feature.feature_type,
        "label": feature.label,
        "role_hint": feature.role_hint,
        "location_raw": feature.location_raw,
        "location_kind": feature.location_kind,
        "start_0": min(int(interval["start_0"]) for interval in projected_intervals),
        "end_0": max(int(interval["end_0"]) for interval in projected_intervals),
        "strand": feature.strand,
        "intervals_0": projected_intervals,
        "is_fuzzy": feature.is_fuzzy,
        "is_compound": feature.is_compound,
        "qualifiers": [qualifier.model_dump() for qualifier in feature.qualifiers],
        "confidence": feature.confidence,
        "source": f"{feature.source}:projected_insert",
    }


def _retention_row(
    feature: ParsedGenBankFeature,
    *,
    status: str,
    insert_start_0: int,
    insert_end_0: int,
    reason: str | None = None,
) -> dict[str, object]:
    projected = _project_feature(feature, insert_start_0=insert_start_0, insert_end_0=insert_end_0)
    original_intervals = [interval.model_dump() for interval in feature.intervals_0]
    original_bp = sum(interval.end_0 - interval.start_0 for interval in feature.intervals_0)
    projected_bp = 0
    if projected is not None:
        projected_bp = sum(int(interval["end_0"]) - int(interval["start_0"]) for interval in projected["intervals_0"])
    return {
        "feature_id": feature.feature_id,
        "label": feature.label,
        "role_hint": feature.role_hint,
        "feature_type": feature.feature_type,
        "status": status,
        "original_intervals_0": original_intervals,
        "derived_intervals_0": None if projected is None else projected["intervals_0"],
        "clipped_bp": max(original_bp - projected_bp, 0),
        "reason": reason,
    }


def _classify_feature_retention(
    features: Iterable[ParsedGenBankFeature],
    *,
    insert_start_0: int,
    insert_end_0: int,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    retained: list[dict[str, object]] = []
    clipped: list[dict[str, object]] = []
    lost: list[dict[str, object]] = []
    for feature in features:
        if _is_primer_feature(feature):
            lost.append(
                _retention_row(
                    feature,
                    status="lost",
                    insert_start_0=insert_start_0,
                    insert_end_0=insert_end_0,
                    reason="primer_flank_removed",
                )
            )
            continue
        if not feature.intervals_0:
            lost.append(
                _retention_row(
                    feature,
                    status="lost",
                    insert_start_0=insert_start_0,
                    insert_end_0=insert_end_0,
                    reason="no_precise_interval",
                )
            )
            continue
        projected = _project_feature(feature, insert_start_0=insert_start_0, insert_end_0=insert_end_0)
        if projected is None:
            lost.append(
                _retention_row(
                    feature,
                    status="lost",
                    insert_start_0=insert_start_0,
                    insert_end_0=insert_end_0,
                    reason="outside_projected_insert",
                )
            )
            continue
        fully_inside = all(
            interval.start_0 >= insert_start_0 and interval.end_0 <= insert_end_0 for interval in feature.intervals_0
        )
        if fully_inside:
            retained.append(
                _retention_row(
                    feature,
                    status="retained",
                    insert_start_0=insert_start_0,
                    insert_end_0=insert_end_0,
                )
            )
        else:
            clipped.append(
                _retention_row(
                    feature,
                    status="clipped",
                    insert_start_0=insert_start_0,
                    insert_end_0=insert_end_0,
                    reason="partially_outside_projected_insert",
                )
            )
    return tuple(retained), tuple(clipped), tuple(lost)


def _project_promoter(path: Path, record: ParsedGenBankRecord) -> PromoterReference:
    source_feature = _select_upstream_feature(record, path=path)
    insert_start_0 = int(source_feature.start_0)
    insert_end_0 = int(source_feature.end_0)
    if insert_end_0 > len(record.sequence):
        raise SchemaError(f"Upstream feature in '{path}' exceeds source sequence length.")
    sequence = _normalize_sequence(record.sequence[insert_start_0:insert_end_0])
    label, override_aliases = _label_from_source_feature(path, source_feature)
    seq_annot_features = tuple(
        projected
        for feature in record.features
        if not _is_primer_feature(feature)
        for projected in [_project_feature(feature, insert_start_0=insert_start_0, insert_end_0=insert_end_0)]
        if projected is not None
    )
    if not seq_annot_features:
        raise SchemaError(f"Projected promoter '{label}' from '{path}' has no retained annotations.")
    retained, clipped, lost = _classify_feature_retention(
        record.features,
        insert_start_0=insert_start_0,
        insert_end_0=insert_end_0,
    )
    return PromoterReference(
        label=label,
        aliases=_dedupe_aliases((*override_aliases, source_feature.label, path.stem), primary=label),
        sequence=sequence,
        source_file=str(path),
        base_source=f"{MG1655_GENBANK_ARTIFACT_SOURCE}:{path.name}:projected_insert",
        source_ref=f"{MG1655_GENBANK_ARTIFACT_SOURCE}:{path.name}",
        source_sha256=record.source_sha256,
        record_id=record.record_id,
        record_name=record.record_name,
        description=record.description,
        topology=record.topology,
        molecule_type=record.molecule_type,
        source_interval_start_0=insert_start_0,
        source_interval_end_0=insert_end_0,
        source_intervals_0=tuple(interval.model_dump() for interval in source_feature.intervals_0),
        source_feature_id=source_feature.feature_id,
        source_feature_label=source_feature.label,
        focal_confidence=source_feature.confidence,
        seq_annot_features=seq_annot_features,
        features_retained=retained,
        features_clipped=clipped,
        features_lost=lost,
        derived_operation="project_genbank_upstream_feature",
        derived_focal_rule="genbank_misc_feature_label_contains_upstream",
        derived_created_by="usr.port_mg1655_promoter_references",
        derivation_spec_id=f"project_genbank_upstream:{path.name}:{insert_start_0}-{insert_end_0}",
        construct_seed_role="anchor",
        construct_seed_manifest_id=PORT_RUN_ID,
    )


def _project_synthetic_standard(path: Path, record: ParsedGenBankRecord, row: dict[str, str]) -> PromoterReference:
    source_feature = _select_full_span_promoter_feature(record, path=path)
    note_map = _qualifier_note_map(source_feature)
    for key in ("collection_id", "promoter_id", "source_record", "role", "strength_metric", "strength_value"):
        if note_map.get(key) != row[key]:
            raise SchemaError(
                f"Synthetic standard '{path}' promoter note '{key}'={note_map.get(key)!r} "
                f"does not match canonical table value {row[key]!r}."
            )
    if note_map.get("strength_reference") != row["strength_reference"]:
        raise SchemaError(
            f"Synthetic standard '{path}' strength_reference={note_map.get('strength_reference')!r} "
            f"does not match canonical table value {row['strength_reference']!r}."
        )
    if source_feature.label != row["display_name"]:
        raise SchemaError(
            f"Synthetic standard '{path}' promoter label {source_feature.label!r} "
            f"does not match display_name {row['display_name']!r}."
        )
    sequence = _normalize_sequence(record.sequence)
    if sequence != _normalize_sequence(row["sequence"]):
        raise SchemaError(f"Synthetic standard '{path}' sequence does not match promoters.csv.")
    seq_annot_features = tuple(
        projected
        for feature in record.features
        if feature.feature_type != "source"
        for projected in [_project_feature(feature, insert_start_0=0, insert_end_0=len(record.sequence))]
        if projected is not None
    )
    retained, clipped, lost = _classify_feature_retention(
        [feature for feature in record.features if feature.feature_type != "source"],
        insert_start_0=0,
        insert_end_0=len(record.sequence),
    )
    label = row["display_name"]
    aliases = [row["promoter_id"], f"{row['collection_id']}:{row['promoter_id']}"]
    if row["promoter_id"] == "BBa_J23105":
        aliases.append("Anderson_J23105")
    source_ref = f"{SYNTHETIC_STANDARDS_ARTIFACT_SOURCE}:{row['collection_id']}:{path.name}"
    return PromoterReference(
        label=label,
        aliases=_dedupe_aliases(aliases, primary=label),
        sequence=sequence,
        source_file=str(path),
        base_source=f"{source_ref}:selected_region",
        source_ref=source_ref,
        source_sha256=record.source_sha256,
        record_id=record.record_id,
        record_name=record.record_name,
        description=record.description,
        topology=record.topology,
        molecule_type=record.molecule_type,
        source_interval_start_0=0,
        source_interval_end_0=len(record.sequence),
        source_intervals_0=tuple(interval.model_dump() for interval in source_feature.intervals_0),
        source_feature_id=source_feature.feature_id,
        source_feature_label=source_feature.label,
        focal_confidence=source_feature.confidence,
        seq_annot_features=seq_annot_features,
        features_retained=retained,
        features_clipped=clipped,
        features_lost=lost,
        derived_operation="import_synthetic_promoter_standard",
        derived_focal_rule="genbank_promoter_full_span",
        derived_created_by="usr.port_mg1655_promoter_references",
        derivation_spec_id=f"synthetic_promoter_standard:{row['collection_id']}:{row['promoter_id']}",
        construct_seed_role="promoter_standard",
        construct_seed_manifest_id=PORT_RUN_ID,
        standard_metadata=_standard_metadata(row),
    )


def _load_legacy_j23105(usr_root: Path, *, legacy_dataset: str) -> tuple[LegacyReference, ...]:
    dataset = Dataset(usr_root, legacy_dataset)
    if not dataset.records_path.exists():
        return ()
    table = pq.read_table(dataset.records_path)
    rows = table.to_pylist()
    expected_id = compute_id("dna", J23105_SEQUENCE)
    for row in rows:
        label = str(row.get("usr_label__primary") or "")
        aliases = [str(alias) for alias in row.get("usr_label__aliases") or []]
        sequence = _normalize_sequence(str(row["sequence"]))
        if (
            label == J23105_LABEL
            or str(row.get("id") or "") == expected_id
            or sequence == J23105_SEQUENCE.lower()
            or any(alias == "Anderson_J23105" for alias in aliases)
        ):
            return (
                LegacyReference(
                    label=J23105_LABEL,
                    aliases=_dedupe_aliases([*aliases, "Anderson_J23105"], primary=J23105_LABEL),
                    sequence=sequence,
                    source_dataset=legacy_dataset,
                ),
            )
    return ()


def _load_synthetic_standards(standards_dir: Path) -> tuple[PromoterReference, ...]:
    source_dir = Path(standards_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Synthetic promoter standards directory does not exist: {source_dir}")
    policy = _synthetic_export_policy(source_dir)
    rows = _synthetic_promoter_rows(source_dir)
    parser = BiopythonGenBankParser()
    promoters: list[PromoterReference] = []
    seen_ids: dict[str, str] = {}
    for row in rows:
        key = (row["collection_id"], row["promoter_id"])
        if not policy.get(key, False):
            continue
        path = source_dir / "genbank" / row["collection_id"] / f"{row['promoter_id']}.gb"
        records = parser.parse_file(path, role_hint_rules=ROLE_HINT_RULES)
        if len(records) != 1:
            raise SchemaError(f"Synthetic standard '{path}' produced {len(records)} records; expected one.")
        promoter = _project_synthetic_standard(path, records[0], row)
        existing_label = seen_ids.get(promoter.id)
        if existing_label is not None:
            raise SchemaError(
                f"Synthetic promoter standard '{promoter.label}' has the same sequence id as '{existing_label}'. "
                "Update promoter_export_policy.csv before porting into usr_promoter_references."
            )
        seen_ids[promoter.id] = promoter.label
        promoters.append(promoter)
    return tuple(promoters)


def build_promoter_reference_plan(
    *,
    archive_dir: Path,
    synthetic_standards_dir: Path | None = None,
    legacy_usr_root: Path | None = None,
    legacy_dataset: str = DEFAULT_LEGACY_DATASET,
    include_legacy_j23105: bool = True,
) -> PromoterReferencePlan:
    source_dir = Path(archive_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Archived GenBank directory does not exist: {source_dir}")
    paths = tuple(sorted(source_dir.glob("*.gb")))
    if not paths:
        raise SchemaError(f"No GenBank files found under '{source_dir}'.")
    parser = BiopythonGenBankParser()
    promoters: list[PromoterReference] = []
    seen_ids: dict[str, str] = {}
    for path in paths:
        records = parser.parse_file(path, role_hint_rules=ROLE_HINT_RULES)
        if len(records) != 1:
            raise SchemaError(f"GenBank source '{path}' produced {len(records)} records; expected one.")
        promoter = _project_promoter(path, records[0])
        existing_label = seen_ids.get(promoter.id)
        if existing_label is not None:
            raise SchemaError(
                f"Projected promoter '{promoter.label}' has the same sequence id as '{existing_label}'. "
                "Resolve the semantic duplicate before creating usr_promoter_references."
            )
        seen_ids[promoter.id] = promoter.label
        promoters.append(promoter)
    if synthetic_standards_dir is not None:
        for promoter in _load_synthetic_standards(Path(synthetic_standards_dir)):
            existing_label = seen_ids.get(promoter.id)
            if existing_label is not None:
                raise SchemaError(
                    f"Synthetic promoter standard '{promoter.label}' has the same sequence id as '{existing_label}'. "
                    "Resolve the semantic duplicate before creating usr_promoter_references."
                )
            seen_ids[promoter.id] = promoter.label
            promoters.append(promoter)
    legacy_references: tuple[LegacyReference, ...] = ()
    if include_legacy_j23105 and legacy_usr_root is not None:
        retained_legacy: list[LegacyReference] = []
        for reference in _load_legacy_j23105(Path(legacy_usr_root), legacy_dataset=legacy_dataset):
            matched_index = next((idx for idx, promoter in enumerate(promoters) if promoter.id == reference.id), None)
            if matched_index is None:
                retained_legacy.append(reference)
                continue
            matched = promoters[matched_index]
            promoters[matched_index] = replace(
                matched,
                aliases=_dedupe_aliases([*matched.aliases, *reference.aliases], primary=matched.label),
            )
        legacy_references = tuple(retained_legacy)
    return PromoterReferencePlan(
        promoters=tuple(sorted(promoters, key=lambda row: row.label.casefold())),
        legacy_references=legacy_references,
        archive_dir=str(source_dir),
        synthetic_standards_dir=None if synthetic_standards_dir is None else str(Path(synthetic_standards_dir)),
        legacy_dataset=legacy_dataset if legacy_references else None,
    )


def _namespace_schema(usr_root: Path, namespace: str, rows: Iterable[dict[str, object]]) -> pa.Schema:
    row_list = list(rows)
    column_names = {key for row in row_list for key in row}
    entry = registry_entry(load_registry(usr_root, required=True), namespace)
    allowed = {column.name: arrow_type_from_str(column.type) for column in entry.columns}
    fields = [pa.field("id", pa.string())]
    for column in entry.columns:
        if column.name in column_names:
            fields.append(pa.field(column.name, allowed[column.name]))
    return pa.schema(fields)


def _table_from_rows(schema: pa.Schema, rows: Iterable[dict[str, object]]) -> pa.Table:
    normalized_rows = [{field.name: row.get(field.name) for field in schema} for row in rows]
    return pa.Table.from_pylist(normalized_rows, schema=schema)


def _sha256_sequence(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def _base_rows(plan: PromoterReferencePlan, *, created_at: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for promoter in plan.promoters:
        rows.append(
            {
                "id": promoter.id,
                "bio_type": "dna",
                "sequence": promoter.sequence.lower(),
                "alphabet": "dna_4",
                "source": promoter.base_source,
                "created_at": created_at,
            }
        )
    for reference in plan.legacy_references:
        rows.append(
            {
                "id": reference.id,
                "bio_type": "dna",
                "sequence": reference.sequence.lower(),
                "alphabet": "dna_4",
                "source": f"legacy_usr:{reference.source_dataset}",
                "created_at": created_at,
            }
        )
    return rows


def _label_rows(plan: PromoterReferencePlan) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for promoter in plan.promoters:
        rows.append(
            {
                "id": promoter.id,
                "usr_label__primary": promoter.label,
                "usr_label__aliases": list(promoter.aliases),
            }
        )
    for reference in plan.legacy_references:
        rows.append(
            {
                "id": reference.id,
                "usr_label__primary": reference.label,
                "usr_label__aliases": list(reference.aliases),
            }
        )
    return rows


def _construct_seed_rows(plan: PromoterReferencePlan) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for promoter in plan.promoters:
        rows.append(
            {
                "id": promoter.id,
                "construct_seed__label": promoter.label,
                "construct_seed__manifest_id": promoter.construct_seed_manifest_id,
                "construct_seed__role": promoter.construct_seed_role,
                "construct_seed__source_ref": promoter.source_ref,
                "construct_seed__topology": "linear",
                "construct_seed__sha256": _sha256_sequence(promoter.sequence),
            }
        )
    for reference in plan.legacy_references:
        rows.append(
            {
                "id": reference.id,
                "construct_seed__label": reference.label,
                "construct_seed__manifest_id": PORT_RUN_ID,
                "construct_seed__role": "incumbent_reference",
                "construct_seed__source_ref": f"legacy_usr:{reference.source_dataset}",
                "construct_seed__topology": "linear",
                "construct_seed__sha256": _sha256_sequence(reference.sequence),
            }
        )
    return rows


def _seq_annot_rows(plan: PromoterReferencePlan, *, dataset: Dataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for promoter in plan.promoters:
        source_path = Path(promoter.source_file)
        rows.append(
            {
                "id": promoter.id,
                "seq_annot__format": "genbank",
                "seq_annot__source_file": promoter.source_file,
                "seq_annot__source_sha256": promoter.source_sha256,
                "seq_annot__source_artifact_uri": _source_artifact_uri(
                    dataset=dataset,
                    source_path=source_path,
                    source_sha256=promoter.source_sha256,
                ),
                "seq_annot__parser": "biopython",
                "seq_annot__parser_version": None,
                "seq_annot__record_id": promoter.record_id,
                "seq_annot__record_name": promoter.record_name,
                "seq_annot__description": promoter.description,
                "seq_annot__topology": promoter.topology,
                "seq_annot__molecule_type": promoter.molecule_type,
                "seq_annot__sequence_region_start_0": 0,
                "seq_annot__sequence_region_end_0": len(promoter.sequence),
                "seq_annot__features": list(promoter.seq_annot_features),
            }
        )
    return rows


def _derived_rows(plan: PromoterReferencePlan) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for promoter in plan.promoters:
        rows.append(
            {
                "id": promoter.id,
                "derived__parent_id": None,
                "derived__parent_dataset": None,
                "derived__operation": promoter.derived_operation,
                "derived__product_kind": "selected_region",
                "derived__target_length": len(promoter.sequence),
                "derived__source_interval_start_0": promoter.source_interval_start_0,
                "derived__source_interval_end_0": promoter.source_interval_end_0,
                "derived__source_intervals_0": list(promoter.source_intervals_0),
                "derived__orientation": "forward",
                "derived__template_id": None,
                "derived__template_dataset": None,
                "derived__focal_rule": promoter.derived_focal_rule,
                "derived__focal_features": [promoter.source_feature_id],
                "derived__focal_confidence": promoter.focal_confidence,
                "derived__analysis_only": False,
                "derived__added_left_bp": None,
                "derived__added_right_bp": None,
                "derived__added_sequence_source": None,
                "derived__features_retained": list(promoter.features_retained),
                "derived__features_clipped": list(promoter.features_clipped),
                "derived__features_lost": list(promoter.features_lost),
                "derived__created_by": promoter.derived_created_by,
                "derived__spec_id": promoter.derivation_spec_id,
            }
        )
    return rows


def _promoter_standard_rows(plan: PromoterReferencePlan) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for promoter in plan.promoters:
        metadata = promoter.standard_metadata
        if metadata is None:
            continue
        rows.append(
            {
                "id": promoter.id,
                "promoter_standard__collection_id": metadata.collection_id,
                "promoter_standard__promoter_id": metadata.promoter_id,
                "promoter_standard__display_name": metadata.display_name,
                "promoter_standard__role": metadata.role,
                "promoter_standard__strength_metric": metadata.strength_metric,
                "promoter_standard__strength_value": metadata.strength_value,
                "promoter_standard__strength_value_numeric": metadata.strength_value_numeric,
                "promoter_standard__strength_reference": metadata.strength_reference,
                "promoter_standard__source_record": metadata.source_record,
                "promoter_standard__notes": metadata.notes,
            }
        )
    return rows


def _sequence_view_rows(
    plan: PromoterReferencePlan,
    *,
    dataset: Dataset,
    created_at: str,
) -> list[SequenceViewRecord]:
    rows: list[SequenceViewRecord] = []
    for promoter in plan.promoters:
        rows.append(
            SequenceViewRecord(
                sequence_id=promoter.id,
                view_name=promoter.label,
                aliases=list(promoter.aliases),
                product_kind="selected_region",
                context_kind="native_reference",
                orientation="forward",
                analysis_only=False,
                source_dataset_id=dataset.name,
                source_label=promoter.label,
                derivation_spec_id=promoter.derivation_spec_id,
                source_interval_start_0=0,
                source_interval_end_0=len(promoter.sequence),
                recommended_pooling="seq_mean",
                created_at=created_at,
                created_by="usr.port_mg1655_promoter_references",
            )
        )
    for reference in plan.legacy_references:
        rows.append(
            SequenceViewRecord(
                sequence_id=reference.id,
                view_name=reference.label,
                aliases=list(reference.aliases),
                product_kind="selected_region",
                context_kind="native_reference",
                orientation="forward",
                analysis_only=False,
                source_dataset_id=dataset.name,
                source_label=reference.label,
                recommended_pooling="seq_mean",
                created_at=created_at,
                created_by="usr.port_mg1655_promoter_references",
            )
        )
    return rows


def _write_overlay_rows(dataset: Dataset, namespace: str, rows: list[dict[str, object]]) -> int:
    if not rows:
        return 0
    schema = _namespace_schema(dataset.root, namespace, rows)
    table = _table_from_rows(schema, rows)
    with dataset.write_session() as session:
        return session.write_overlay(namespace, table, key="id", note=f"{PORT_RUN_ID}:{namespace}")


def _copy_source_artifacts(dataset: Dataset, plan: PromoterReferencePlan) -> list[str]:
    copied: list[str] = []
    for promoter in plan.promoters:
        source_path = Path(promoter.source_file)
        copied.append(_copy_source_artifact(dataset, source_path, promoter.source_sha256))
    return copied


def _validate_plan_counts(
    plan: PromoterReferencePlan,
    *,
    expected_genbank_count: int | None,
    include_legacy_j23105: bool,
) -> None:
    if expected_genbank_count is not None and len(plan.promoters) != expected_genbank_count:
        raise SchemaError(f"Expected {expected_genbank_count} GenBank promoter rows, found {len(plan.promoters)}.")
    has_j23105_promoter = any(row.label == J23105_LABEL for row in plan.promoters)
    if include_legacy_j23105 and not plan.legacy_references and not has_j23105_promoter:
        raise SchemaError("Expected legacy J23105 reference, but no J23105 row was found in the legacy dataset.")


def write_promoter_reference_dataset(
    plan: PromoterReferencePlan,
    *,
    usr_root: Path,
    output_dataset: str = DEFAULT_OUTPUT_DATASET,
    expected_genbank_count: int | None = DEFAULT_EXPECTED_GENBANK_COUNT,
    include_legacy_j23105: bool = True,
) -> WriteResult:
    _validate_plan_counts(
        plan,
        expected_genbank_count=expected_genbank_count,
        include_legacy_j23105=include_legacy_j23105,
    )
    _ensure_promoter_reference_namespaces(usr_root)
    dataset = Dataset(usr_root, output_dataset)
    if dataset.dir.exists():
        raise FileExistsError(f"Output dataset already exists: {dataset.dir}")
    created_at = now_utc()
    base_rows = _base_rows(plan, created_at=created_at)
    label_rows = _label_rows(plan)
    construct_seed_rows = _construct_seed_rows(plan)
    seq_annot_rows = _seq_annot_rows(plan, dataset=dataset)
    derived_rows = _derived_rows(plan)
    promoter_standard_rows = _promoter_standard_rows(plan)

    with dataset.write_session() as session:
        session.init(
            source="projected promoter reference inserts",
            notes=(
                "Primer-flank-stripped MG1655 promoter references plus source-backed synthetic "
                "promoter standards projected from archived GenBank records."
            ),
        )
        rows_written = session.import_rows(base_rows, source="projected_promoter_reference_insert")

    label_count = _write_overlay_rows(dataset, "usr_label", label_rows)
    construct_seed_count = _write_overlay_rows(dataset, "construct_seed", construct_seed_rows)
    seq_annot_count = _write_overlay_rows(dataset, "seq_annot", seq_annot_rows)
    derived_count = _write_overlay_rows(dataset, "derived", derived_rows)
    promoter_standard_count = _write_overlay_rows(dataset, PROMOTER_STANDARD_NAMESPACE, promoter_standard_rows)
    sequence_view_count = write_sequence_views(
        dataset,
        _sequence_view_rows(plan, dataset=dataset, created_at=created_at),
        conflict_policy="error",
    )
    copied_artifacts = _copy_source_artifacts(dataset, plan)
    with dataset.maintenance("materialize_promoter_reference_labels_and_construct_seed"):
        dataset.materialize(namespaces=["usr_label", "construct_seed"], keep_overlays=True)
    dataset.log_event(
        "promoter_reference_projected_genbank_port",
        args={
            "archive_dir": plan.archive_dir,
            "synthetic_standards_dir": plan.synthetic_standards_dir,
            "legacy_dataset": plan.legacy_dataset,
            "output_dataset": dataset.name,
        },
        metrics={
            "genbank_rows": len(plan.promoters),
            "synthetic_standard_rows": len(promoter_standard_rows),
            "legacy_rows": len(plan.legacy_references),
            "total_rows": rows_written,
        },
        artifacts={"copied_genbank_sources": copied_artifacts},
    )
    dataset.validate(strict=True)
    return WriteResult(
        dataset=dataset.name,
        dataset_dir=str(dataset.dir),
        rows_written=int(rows_written),
        genbank_rows_written=len(plan.promoters),
        legacy_rows_written=len(plan.legacy_references),
        seq_annot_overlay_rows=int(seq_annot_count),
        derived_overlay_rows=int(derived_count),
        label_overlay_rows=int(label_count),
        construct_seed_overlay_rows=int(construct_seed_count),
        promoter_standard_overlay_rows=int(promoter_standard_count),
        sequence_views_written=int(sequence_view_count),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project archived GenBank promoter-reference inserts into the modern usr_promoter_references dataset."
        )
    )
    parser.add_argument("--archive-dir", type=Path, default=_default_archive_dir())
    parser.add_argument("--synthetic-standards-dir", type=Path, default=_default_synthetic_standards_dir())
    parser.add_argument(
        "--no-synthetic-standards",
        action="store_true",
        help="Do not add archived synthetic promoter standards to usr_promoter_references.",
    )
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--legacy-dataset", default=DEFAULT_LEGACY_DATASET)
    parser.add_argument("--output-dataset", default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--expected-genbank-count", type=int, default=DEFAULT_EXPECTED_GENBANK_COUNT)
    parser.add_argument("--no-legacy-j23105", action="store_true", help="Do not carry J23105 from the legacy dataset.")
    parser.add_argument(
        "--write", action="store_true", help="Actually create the output USR dataset. Default is dry-run."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    include_legacy_j23105 = not bool(args.no_legacy_j23105)
    synthetic_standards_dir = None if bool(args.no_synthetic_standards) else args.synthetic_standards_dir
    plan = build_promoter_reference_plan(
        archive_dir=args.archive_dir,
        synthetic_standards_dir=synthetic_standards_dir,
        legacy_usr_root=args.usr_root if include_legacy_j23105 else None,
        legacy_dataset=args.legacy_dataset,
        include_legacy_j23105=include_legacy_j23105,
    )
    _validate_plan_counts(
        plan,
        expected_genbank_count=args.expected_genbank_count,
        include_legacy_j23105=include_legacy_j23105,
    )
    payload: dict[str, Any] = {"plan": plan.summary(), "write": bool(args.write)}
    if args.write:
        result = write_promoter_reference_dataset(
            plan,
            usr_root=args.usr_root,
            output_dataset=args.output_dataset,
            expected_genbank_count=args.expected_genbank_count,
            include_legacy_j23105=include_legacy_j23105,
        )
        payload["result"] = result.__dict__
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
