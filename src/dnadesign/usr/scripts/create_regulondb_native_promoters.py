"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/scripts/create_regulondb_native_promoters.py

Create a native RegulonDB promoter USR dataset from a Cruncher promoter export.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.cruncher.ingest.promoters import (
    PromoterRecord,
    PromoterRegulatoryAssociation,
    load_promoter_export,
    load_promoter_regulatory_associations,
    load_skipped_promoter_source_rows,
    promoter_record_to_dict,
    skipped_source_row_to_dict,
)
from dnadesign.usr import Dataset
from dnadesign.usr.src.contracts import SchemaError, compute_id
from dnadesign.usr.src.registry import arrow_type_from_str, load_registry, registry_entry
from dnadesign.usr.src.sequence_views import (
    SequenceViewRecord,
    ViewSemanticsRecord,
    write_sequence_views,
    write_view_semantics,
)
from dnadesign.usr.src.storage.parquet import now_utc

DEFAULT_OUTPUT_DATASET = "usr_regulondb_native_promoters"
CREATED_BY = "dnadesign.usr.create_regulondb_native_promoters"
REGULONDB_NAMESPACE = "regulondb"
RELATIONS_DIRNAME = "_relations"
_STRICT_DNA = set("ACGT")
_FUZZY_NAME_RATIO_THRESHOLD = 0.92
_FUZZY_NAME_COLLISION_LIMIT = 50
_BASE_ROW_REQUIRED_METADATA = ("sigma",)

_RELATION_SCHEMAS: dict[str, pa.Schema] = {
    "promoter_aliases": pa.schema(
        [
            pa.field("usr_id", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("source_table", pa.string()),
            pa.field("source_stratum", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("promoter_name", pa.string()),
            pa.field("raw_sequence_sha256", pa.string()),
            pa.field("strand", pa.string()),
            pa.field("tss_raw", pa.string()),
            pa.field("first_gene", pa.string()),
            pa.field("tu_id", pa.string()),
            pa.field("operon_id", pa.string()),
            pa.field("confidence_level", pa.string()),
            pa.field("source_row_ref", pa.string()),
        ]
    ),
    "sigma_affiliations": pa.schema(
        [
            pa.field("usr_id", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("sigma_id", pa.string()),
            pa.field("sigma_name", pa.string()),
            pa.field("sigma_abbrev", pa.string()),
            pa.field("sigma_canonical_label", pa.string()),
            pa.field("sigma_gene_id", pa.string()),
            pa.field("sigma_gene_name", pa.string()),
            pa.field("evidence", pa.list_(pa.string())),
            pa.field("confidence", pa.string()),
            pa.field("citation_refs", pa.list_(pa.string())),
        ]
    ),
    "regulatory_interactions": pa.schema(
        [
            pa.field("usr_id", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("regulatory_interaction_id", pa.string()),
            pa.field("regulon_id", pa.string()),
            pa.field("regulon_name", pa.string()),
            pa.field("regulator_id", pa.string()),
            pa.field("regulator_name", pa.string()),
            pa.field("regulator_abbrev", pa.string()),
            pa.field("target_type", pa.string()),
            pa.field("function", pa.string()),
            pa.field("mechanism", pa.string()),
            pa.field("confidence", pa.string()),
            pa.field("evidence", pa.list_(pa.string())),
            pa.field("citation_refs", pa.list_(pa.string())),
        ]
    ),
    "tfbs_sites": pa.schema(
        [
            pa.field("usr_id", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("binding_site_id", pa.string()),
            pa.field("regulator_abbrev", pa.string()),
            pa.field("raw_coordinates_json", pa.string()),
            pa.field("interval_start_0", pa.int64()),
            pa.field("interval_end_0", pa.int64()),
            pa.field("strand", pa.string()),
            pa.field("sequence", pa.string()),
            pa.field("confidence", pa.string()),
            pa.field("evidence", pa.list_(pa.string())),
        ]
    ),
    "promoter_boxes": pa.schema(
        [
            pa.field("usr_id", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("box_kind", pa.string()),
            pa.field("sequence", pa.string()),
            pa.field("raw_coordinates_json", pa.string()),
            pa.field("interval_start_0", pa.int64()),
            pa.field("interval_end_0", pa.int64()),
            pa.field("strand", pa.string()),
            pa.field("spacer_length", pa.int64()),
        ]
    ),
    "evidence_citations": pa.schema(
        [
            pa.field("usr_id", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("evidence_text", pa.string()),
            pa.field("citation_ref", pa.string()),
        ]
    ),
    "coordinate_features": pa.schema(
        [
            pa.field("usr_id", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("feature_kind", pa.string()),
            pa.field("interval_start_0", pa.int64()),
            pa.field("interval_end_0", pa.int64()),
            pa.field("strand", pa.string()),
            pa.field("coordinate_origin", pa.string()),
            pa.field("coordinate_inclusivity", pa.string()),
            pa.field("genome_accession", pa.string()),
        ]
    ),
    "source_conflicts": pa.schema(
        [
            pa.field("usr_id", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("severity", pa.string()),
            pa.field("field", pa.string()),
            pa.field("details", pa.string()),
            pa.field("resolution", pa.string()),
        ]
    ),
    "source_rows": pa.schema(
        [
            pa.field("usr_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_release_date", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("source_table", pa.string()),
            pa.field("source_stratum", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("promoter_name", pa.string()),
            pa.field("source_row_ref", pa.string()),
            pa.field("raw_payload_sha256", pa.string()),
            pa.field("query_sha256", pa.string()),
            pa.field("normalized_record_json", pa.string()),
        ]
    ),
    "excluded_source_rows": pa.schema(
        [
            pa.field("source", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_release_date", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("source_table", pa.string()),
            pa.field("source_stratum", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("promoter_name", pa.string()),
            pa.field("raw_sequence_sha256", pa.string()),
            pa.field("sequence_length", pa.int64()),
            pa.field("strand", pa.string()),
            pa.field("tss_raw", pa.string()),
            pa.field("confidence_level", pa.string()),
            pa.field("exclusion_reason", pa.string()),
            pa.field("source_row_ref", pa.string()),
            pa.field("raw_payload_sha256", pa.string()),
            pa.field("query_sha256", pa.string()),
            pa.field("normalized_record_json", pa.string()),
        ]
    ),
    "skipped_source_rows": pa.schema(
        [
            pa.field("source", pa.string()),
            pa.field("source_release", pa.string()),
            pa.field("source_release_date", pa.string()),
            pa.field("source_route", pa.string()),
            pa.field("source_table", pa.string()),
            pa.field("source_stratum", pa.string()),
            pa.field("promoter_id", pa.string()),
            pa.field("promoter_name", pa.string()),
            pa.field("raw_sequence", pa.string()),
            pa.field("skip_reason", pa.string()),
            pa.field("source_row_ref", pa.string()),
            pa.field("raw_payload_sha256", pa.string()),
            pa.field("query_sha256", pa.string()),
            pa.field("parser_version", pa.string()),
            pa.field("export_schema_version", pa.string()),
            pa.field("normalized_skipped_row_json", pa.string()),
        ]
    ),
}


@dataclass(frozen=True)
class NativePromoterImportPlan:
    dataset: str
    export_dir: str
    source_manifest_complete: bool
    base_rows: list[dict[str, object]]
    regulondb_overlay_rows: list[dict[str, object]]
    relation_rows: dict[str, list[dict[str, object]]]
    validation_report: dict[str, object]
    conflict_report: list[dict[str, object]]

    def summary(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "export_dir": self.export_dir,
            "source_manifest_complete": self.source_manifest_complete,
            "base_rows": len(self.base_rows),
            "regulondb_overlay_rows": len(self.regulondb_overlay_rows),
            "relation_rows": {name: len(rows) for name, rows in sorted(self.relation_rows.items())},
            "validation_report": dict(self.validation_report),
            "conflict_count": len(self.conflict_report),
        }


@dataclass(frozen=True)
class WriteResult:
    dataset: str
    dataset_dir: str
    rows_written: int
    regulondb_overlay_rows: int
    sequence_view_rows: int
    view_semantics_rows: int
    relation_sidecars: dict[str, int]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_usr_root() -> Path:
    return _repo_root() / "src" / "dnadesign" / "usr" / "datasets"


def _canonical_usr_sequence(record: PromoterRecord) -> str:
    sequence = str(record.sequence or "").upper().replace("U", "T")
    if not sequence or set(sequence) - _STRICT_DNA:
        raise SchemaError(f"RegulonDB promoter {record.promoter_id!r} has invalid strict DNA sequence.")
    return sequence


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _list_unique(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text.casefold() in seen:
            continue
        seen.add(text.casefold())
        out.append(text)
    return sorted(out, key=str.casefold)


def _interval_start(interval: tuple[int, int] | None) -> int | None:
    return None if interval is None else int(interval[0])


def _interval_end(interval: tuple[int, int] | None) -> int | None:
    return None if interval is None else int(interval[1])


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _check_regulondb_namespace(usr_root: Path) -> None:
    registry = load_registry(usr_root, required=True)
    entry = registry_entry(registry, REGULONDB_NAMESPACE)
    observed = {column.name: column.type for column in entry.columns}
    required = {
        "regulondb__release": "string",
        "regulondb__primary_source": "string",
        "regulondb__source_strata_set": "list<string>",
        "regulondb__primary_promoter_id": "string",
        "regulondb__primary_promoter_name": "string",
        "regulondb__promoter_alias_count": "int64",
        "regulondb__sigma_factor_set": "list<string>",
        "regulondb__sigma_factor_count": "int64",
        "regulondb__confidence_level_set": "list<string>",
        "regulondb__has_minus10_box": "bool",
        "regulondb__has_minus35_box": "bool",
        "regulondb__box_pattern": "string",
        "regulondb__regulator_composition": "string",
        "regulondb__regulon_count": "int64",
        "regulondb__has_activator": "bool",
        "regulondb__has_repressor": "bool",
        "regulondb__tss_available": "bool",
        "regulondb__metadata_completeness_class": "string",
        "regulondb__raw_sequence_case_policy": "string",
        "regulondb__has_sequence": "bool",
        "regulondb__has_tss": "bool",
        "regulondb__has_sigma": "bool",
        "regulondb__has_confidence": "bool",
        "regulondb__has_boxes": "bool",
        "regulondb__has_regulatory_context": "bool",
        "regulondb__has_citations": "bool",
    }
    missing = sorted(name for name in required if name not in observed)
    mismatched = sorted(name for name, typ in required.items() if observed.get(name) not in {None, typ})
    if missing:
        raise SchemaError(f"Registry namespace 'regulondb' is missing required columns: {missing}")
    if mismatched:
        details = ", ".join(f"{name}={observed[name]!r}, expected {required[name]!r}" for name in mismatched)
        raise SchemaError(f"Registry namespace 'regulondb' has incompatible column types: {details}")


def _metadata_completeness_class(records: list[PromoterRecord]) -> str:
    has_tss = any(record.tss_interval_0based is not None for record in records)
    has_sigma = any(record.sigma_affiliations for record in records)
    has_boxes = any(record.boxes for record in records)
    has_confidence = any(record.confidence_level for record in records)
    has_regulatory = any(record.regulatory_sites for record in records)
    if has_tss and has_sigma and has_boxes and has_confidence and has_regulatory:
        return "complete"
    if has_tss and (has_sigma or has_boxes or has_regulatory):
        return "partial"
    return "sequence_only"


def _source_stratum(record: PromoterRecord) -> str:
    return record.provenance.source_stratum or record.source_route or "unknown"


def _regulator_composition(functions: list[str]) -> str:
    normalized = {function.strip().casefold() for function in functions if function and function.strip()}
    has_activator = any("activ" in value for value in normalized)
    has_repressor = any("repress" in value for value in normalized)
    if has_activator and has_repressor:
        return "mixed"
    if has_activator:
        return "activator"
    if has_repressor:
        return "repressor"
    if normalized:
        return "other"
    return "unknown"


def _box_pattern(kinds: list[str]) -> str:
    ordered = _list_unique(kinds)
    if not ordered:
        return "none"
    return "+".join(ordered)


_SIGMA_CANONICAL_LABELS = {
    "rpod": "sigma70",
    "sigma70": "sigma70",
    "sigma 70": "sigma70",
    "rpos": "sigma38",
    "sigma38": "sigma38",
    "sigma 38": "sigma38",
    "rpoh": "sigma32",
    "sigma32": "sigma32",
    "sigma 32": "sigma32",
    "rpoe": "sigma24",
    "sigma24": "sigma24",
    "sigma 24": "sigma24",
    "rpon": "sigma54",
    "sigma54": "sigma54",
    "sigma 54": "sigma54",
    "flia": "sigma28",
    "sigma28": "sigma28",
    "sigma 28": "sigma28",
    "feci": "sigma19",
    "sigma19": "sigma19",
    "sigma 19": "sigma19",
}


def _canonical_sigma_label(*values: object) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = re.sub(r"[^a-z0-9]+", "", text.casefold())
        if key in _SIGMA_CANONICAL_LABELS:
            return _SIGMA_CANONICAL_LABELS[key]
        return text
    return None


def _base_source(records: list[PromoterRecord]) -> str:
    releases = _list_unique(record.source_release for record in records)
    source = f"regulondb:{'+'.join(releases)}"
    if len(records) == 1:
        return f"{source}:{records[0].promoter_id}"
    return f"{source}:duplicate_promoter_sequence"


def _primary_value(values: list[str | None]) -> str | None:
    observed = _list_unique(values)
    return observed[0] if len(observed) == 1 else None


def _overlay_row(usr_id: str, records: list[PromoterRecord]) -> dict[str, object]:
    sigmas = _list_unique(
        _canonical_sigma_label(sigma.abbrev, sigma.name, sigma.sigma_id, sigma.gene_name, sigma.gene_id)
        for record in records
        for sigma in record.sigma_affiliations
    )
    confidence = _list_unique(record.confidence_level for record in records)
    box_kinds = [box.kind for record in records for box in record.boxes]
    functions = [site.function or "" for record in records for site in record.regulatory_sites]
    regulon_ids = _list_unique(site.regulon_id for record in records for site in record.regulatory_sites)
    citations = [
        citation
        for record in records
        for citation in [
            *record.citations,
            *(citation for sigma in record.sigma_affiliations for citation in sigma.citation_refs),
            *(citation for site in record.regulatory_sites for citation in site.citation_refs),
        ]
    ]
    composition = _regulator_composition(functions)
    return {
        "id": usr_id,
        "regulondb__release": "+".join(_list_unique(record.source_release for record in records)),
        "regulondb__primary_source": "regulondb",
        "regulondb__source_strata_set": _list_unique(_source_stratum(record) for record in records),
        "regulondb__primary_promoter_id": _primary_value([record.promoter_id for record in records]),
        "regulondb__primary_promoter_name": _primary_value([record.promoter_name for record in records]),
        "regulondb__promoter_alias_count": len(records),
        "regulondb__sigma_factor_set": sigmas,
        "regulondb__sigma_factor_count": len(sigmas),
        "regulondb__confidence_level_set": confidence,
        "regulondb__has_minus10_box": "minus_10" in set(box_kinds),
        "regulondb__has_minus35_box": "minus_35" in set(box_kinds),
        "regulondb__box_pattern": _box_pattern(box_kinds),
        "regulondb__regulator_composition": composition,
        "regulondb__regulon_count": len(regulon_ids),
        "regulondb__has_activator": composition in {"activator", "mixed"},
        "regulondb__has_repressor": composition in {"repressor", "mixed"},
        "regulondb__tss_available": any(record.tss_interval_0based is not None for record in records),
        "regulondb__metadata_completeness_class": _metadata_completeness_class(records),
        "regulondb__raw_sequence_case_policy": _primary_value([record.sequence_case_policy for record in records])
        or "mixed",
        "regulondb__has_sequence": True,
        "regulondb__has_tss": any(record.tss_interval_0based is not None for record in records),
        "regulondb__has_sigma": bool(sigmas),
        "regulondb__has_confidence": bool(confidence),
        "regulondb__has_boxes": bool(box_kinds),
        "regulondb__has_regulatory_context": any(record.regulatory_sites for record in records),
        "regulondb__has_citations": bool(citations),
    }


def _empty_relations() -> dict[str, list[dict[str, object]]]:
    return {name: [] for name in _RELATION_SCHEMAS}


def _add_coordinate_feature(
    rows: list[dict[str, object]],
    *,
    usr_id: str,
    record: PromoterRecord,
    feature_kind: str,
    interval: tuple[int, int] | None,
    strand: str | None,
) -> None:
    if interval is None:
        return
    rows.append(
        {
            "usr_id": usr_id,
            "promoter_id": record.promoter_id,
            "source_release": record.source_release,
            "source_route": record.source_route,
            "feature_kind": feature_kind,
            "interval_start_0": _interval_start(interval),
            "interval_end_0": _interval_end(interval),
            "strand": strand,
            "coordinate_origin": "regulondb_1based",
            "coordinate_inclusivity": "source_inclusive_normalized_half_open",
            "genome_accession": record.genome_accession,
        }
    )


def _relation_rows_for_group(
    usr_id: str,
    records: list[PromoterRecord],
    rows: dict[str, list[dict[str, object]]],
) -> None:
    for record in records:
        rows["source_rows"].append(
            {
                "usr_id": usr_id,
                "source": record.source,
                "source_release": record.source_release,
                "source_release_date": record.provenance.source_release_date,
                "source_route": record.source_route,
                "source_table": record.provenance.source_table,
                "source_stratum": _source_stratum(record),
                "promoter_id": record.promoter_id,
                "promoter_name": record.promoter_name,
                "source_row_ref": record.provenance.raw_payload_ref,
                "raw_payload_sha256": record.provenance.raw_payload_sha256,
                "query_sha256": record.provenance.query_sha256,
                "normalized_record_json": _json(promoter_record_to_dict(record)),
            }
        )
        rows["promoter_aliases"].append(
            {
                "usr_id": usr_id,
                "source_release": record.source_release,
                "source_route": record.source_route,
                "source_table": record.provenance.source_table,
                "source_stratum": _source_stratum(record),
                "promoter_id": record.promoter_id,
                "promoter_name": record.promoter_name,
                "raw_sequence_sha256": _sha256_text(record.raw_sequence),
                "strand": record.strand,
                "tss_raw": record.tss_position_raw,
                "first_gene": None if record.first_gene is None else record.first_gene.name,
                "tu_id": record.transcription_units[0].tu_id if record.transcription_units else None,
                "operon_id": None if record.operon is None else record.operon.operon_id,
                "confidence_level": record.confidence_level,
                "source_row_ref": record.provenance.raw_payload_ref or record.provenance.raw_payload_sha256,
            }
        )
        _add_coordinate_feature(
            rows["coordinate_features"],
            usr_id=usr_id,
            record=record,
            feature_kind="tss",
            interval=record.tss_interval_0based,
            strand=record.strand,
        )
        for sigma in record.sigma_affiliations:
            rows["sigma_affiliations"].append(
                {
                    "usr_id": usr_id,
                    "promoter_id": record.promoter_id,
                    "source_release": record.source_release,
                    "source_route": sigma.source_route,
                    "sigma_id": sigma.sigma_id,
                    "sigma_name": sigma.name,
                    "sigma_abbrev": sigma.abbrev,
                    "sigma_canonical_label": _canonical_sigma_label(
                        sigma.abbrev, sigma.name, sigma.sigma_id, sigma.gene_name, sigma.gene_id
                    ),
                    "sigma_gene_id": sigma.gene_id,
                    "sigma_gene_name": sigma.gene_name,
                    "evidence": list(sigma.evidence),
                    "confidence": sigma.confidence,
                    "citation_refs": list(sigma.citation_refs),
                }
            )
            for citation in sigma.citation_refs:
                rows["evidence_citations"].append(
                    {
                        "usr_id": usr_id,
                        "promoter_id": record.promoter_id,
                        "source_release": record.source_release,
                        "source_route": sigma.source_route,
                        "evidence_text": ",".join(sigma.evidence) if sigma.evidence else None,
                        "citation_ref": citation,
                    }
                )
        for box in record.boxes:
            rows["promoter_boxes"].append(
                {
                    "usr_id": usr_id,
                    "promoter_id": record.promoter_id,
                    "source_release": record.source_release,
                    "source_route": box.source_route,
                    "box_kind": box.kind,
                    "sequence": box.sequence,
                    "raw_coordinates_json": _json(box.raw_coordinates),
                    "interval_start_0": _interval_start(box.interval_0based),
                    "interval_end_0": _interval_end(box.interval_0based),
                    "strand": box.strand,
                    "spacer_length": None,
                }
            )
            _add_coordinate_feature(
                rows["coordinate_features"],
                usr_id=usr_id,
                record=record,
                feature_kind=f"box:{box.kind}",
                interval=box.interval_0based,
                strand=box.strand,
            )
        for site in record.regulatory_sites:
            rows["regulatory_interactions"].append(
                {
                    "usr_id": usr_id,
                    "promoter_id": record.promoter_id,
                    "source_release": record.source_release,
                    "source_route": record.source_route,
                    "regulatory_interaction_id": site.regulatory_interaction_id,
                    "regulon_id": site.regulon_id,
                    "regulon_name": site.regulon_name,
                    "regulator_id": site.regulator_id,
                    "regulator_name": site.regulator_name,
                    "regulator_abbrev": site.regulator_abbrev,
                    "target_type": site.target_type,
                    "function": site.function,
                    "mechanism": site.mechanism,
                    "confidence": site.confidence,
                    "evidence": list(site.evidence),
                    "citation_refs": list(site.citation_refs),
                }
            )
            if site.binding_site_id or site.interval_0based or site.sequence:
                rows["tfbs_sites"].append(
                    {
                        "usr_id": usr_id,
                        "promoter_id": record.promoter_id,
                        "source_release": record.source_release,
                        "source_route": record.source_route,
                        "binding_site_id": site.binding_site_id,
                        "regulator_abbrev": site.regulator_abbrev,
                        "raw_coordinates_json": _json(site.raw_coordinates),
                        "interval_start_0": _interval_start(site.interval_0based),
                        "interval_end_0": _interval_end(site.interval_0based),
                        "strand": site.strand,
                        "sequence": site.sequence,
                        "confidence": site.confidence,
                        "evidence": list(site.evidence),
                    }
                )
            _add_coordinate_feature(
                rows["coordinate_features"],
                usr_id=usr_id,
                record=record,
                feature_kind="tfbs",
                interval=site.interval_0based,
                strand=site.strand,
            )
            for citation in site.citation_refs:
                rows["evidence_citations"].append(
                    {
                        "usr_id": usr_id,
                        "promoter_id": record.promoter_id,
                        "source_release": record.source_release,
                        "source_route": record.source_route,
                        "evidence_text": ",".join(site.evidence) if site.evidence else None,
                        "citation_ref": citation,
                    }
                )


def _association_match_key(value: object) -> str:
    return str(value or "").strip().casefold()


def _association_alias_index(
    rows: dict[str, list[dict[str, object]]],
    *,
    key_field: str,
) -> dict[str, list[dict[str, object]]]:
    aliases_by_key: dict[str, list[dict[str, object]]] = {}
    for alias in rows["promoter_aliases"]:
        key = _association_match_key(alias.get(key_field))
        if key:
            aliases_by_key.setdefault(key, []).append(alias)
    return aliases_by_key


def _matched_aliases_for_association(
    association: PromoterRegulatoryAssociation,
    *,
    aliases_by_id: dict[str, list[dict[str, object]]],
    aliases_by_name: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], str | None]:
    promoter_id_key = _association_match_key(association.promoter_id)
    if promoter_id_key and promoter_id_key in aliases_by_id:
        return aliases_by_id[promoter_id_key], "promoter_id"
    promoter_name_key = _association_match_key(association.promoter_name)
    if promoter_name_key and promoter_name_key in aliases_by_name:
        return aliases_by_name[promoter_name_key], "promoter_name"
    return [], None


def _project_promoter_associations(
    associations: Iterable[PromoterRegulatoryAssociation],
    rows: dict[str, list[dict[str, object]]],
) -> dict[str, int]:
    association_list = list(associations)
    aliases_by_id = _association_alias_index(rows, key_field="promoter_id")
    aliases_by_name = _association_alias_index(rows, key_field="promoter_name")

    summary = {
        "promoter_association_rows": len(association_list),
        "promoter_association_matched_rows": 0,
        "promoter_association_id_matched_rows": 0,
        "promoter_association_name_matched_rows": 0,
        "promoter_association_unmatched_rows": 0,
        "promoter_association_ambiguous_rows": 0,
    }
    for association in association_list:
        alias_rows, match_kind = _matched_aliases_for_association(
            association,
            aliases_by_id=aliases_by_id,
            aliases_by_name=aliases_by_name,
        )
        if not alias_rows:
            summary["promoter_association_unmatched_rows"] += 1
            continue
        usr_ids = {str(alias["usr_id"]) for alias in alias_rows if alias.get("usr_id")}
        if len(usr_ids) != 1:
            summary["promoter_association_ambiguous_rows"] += 1
            continue
        matched_alias = sorted(
            alias_rows,
            key=lambda row: (
                str(row.get("source_release") or ""),
                str(row.get("source_route") or ""),
                str(row.get("promoter_id") or ""),
            ),
        )[0]
        usr_id = next(iter(usr_ids))
        promoter_id = association.promoter_id or matched_alias.get("promoter_id")
        rows["regulatory_interactions"].append(
            {
                "usr_id": usr_id,
                "promoter_id": promoter_id,
                "source_release": association.source_release,
                "source_route": association.source_route,
                "regulatory_interaction_id": association.regulatory_interaction_id,
                "regulon_id": association.regulon_id,
                "regulon_name": association.regulon_name,
                "regulator_id": association.regulator_id,
                "regulator_name": association.regulator_name,
                "regulator_abbrev": association.regulator_abbrev,
                "target_type": association.target_type,
                "function": association.function,
                "mechanism": association.mechanism,
                "confidence": association.confidence,
                "evidence": list(association.evidence),
                "citation_refs": list(association.citation_refs),
            }
        )
        if association.binding_site_id or association.binding_interval_0based or association.binding_site_sequence:
            rows["tfbs_sites"].append(
                {
                    "usr_id": usr_id,
                    "promoter_id": promoter_id,
                    "source_release": association.source_release,
                    "source_route": association.source_route,
                    "binding_site_id": association.binding_site_id,
                    "regulator_abbrev": association.regulator_abbrev,
                    "raw_coordinates_json": _json(association.binding_raw_coordinates or {}),
                    "interval_start_0": _interval_start(association.binding_interval_0based),
                    "interval_end_0": _interval_end(association.binding_interval_0based),
                    "strand": association.binding_site_strand,
                    "sequence": association.binding_site_sequence,
                    "confidence": association.confidence,
                    "evidence": list(association.evidence),
                }
            )
            rows["coordinate_features"].append(
                {
                    "usr_id": usr_id,
                    "promoter_id": promoter_id,
                    "source_release": association.source_release,
                    "source_route": association.source_route,
                    "feature_kind": "tfbs",
                    "interval_start_0": _interval_start(association.binding_interval_0based),
                    "interval_end_0": _interval_end(association.binding_interval_0based),
                    "strand": association.binding_site_strand,
                    "coordinate_origin": "regulondb_1based",
                    "coordinate_inclusivity": "source_inclusive_normalized_half_open",
                    "genome_accession": None,
                }
            )
        summary["promoter_association_matched_rows"] += 1
        if match_kind == "promoter_id":
            summary["promoter_association_id_matched_rows"] += 1
        elif match_kind == "promoter_name":
            summary["promoter_association_name_matched_rows"] += 1
    return summary


def _add_skipped_source_rows(export_dir: Path, rows: dict[str, list[dict[str, object]]]) -> Counter[str]:
    skipped_by_reason: Counter[str] = Counter()
    for skipped in load_skipped_promoter_source_rows(export_dir):
        payload = skipped_source_row_to_dict(skipped)
        skipped_by_reason[str(payload["skip_reason"])] += 1
        rows["skipped_source_rows"].append(
            {
                "source": payload["source"],
                "source_release": payload["source_release"],
                "source_release_date": payload.get("source_release_date"),
                "source_route": payload["source_route"],
                "source_table": payload.get("source_table"),
                "source_stratum": payload["source_stratum"],
                "promoter_id": payload.get("promoter_id"),
                "promoter_name": payload.get("promoter_name"),
                "raw_sequence": payload.get("raw_sequence"),
                "skip_reason": payload["skip_reason"],
                "source_row_ref": payload["source_row_ref"],
                "raw_payload_sha256": payload["raw_payload_sha256"],
                "query_sha256": payload["query_sha256"],
                "parser_version": payload["parser_version"],
                "export_schema_version": payload["export_schema_version"],
                "normalized_skipped_row_json": _json(payload),
            }
        )
    return skipped_by_reason


def _base_group_exclusion_reason(records: list[PromoterRecord]) -> str | None:
    if "sigma" in _BASE_ROW_REQUIRED_METADATA and not any(record.sigma_affiliations for record in records):
        return "missing_sigma"
    return None


def _add_excluded_source_rows(
    records: list[PromoterRecord], rows: dict[str, list[dict[str, object]]], *, exclusion_reason: str
) -> None:
    for record in records:
        rows["excluded_source_rows"].append(
            {
                "source": record.source,
                "source_release": record.source_release,
                "source_release_date": record.provenance.source_release_date,
                "source_route": record.source_route,
                "source_table": record.provenance.source_table,
                "source_stratum": _source_stratum(record),
                "promoter_id": record.promoter_id,
                "promoter_name": record.promoter_name,
                "raw_sequence_sha256": _sha256_text(record.raw_sequence),
                "sequence_length": record.sequence_length,
                "strand": record.strand,
                "tss_raw": record.tss_position_raw,
                "confidence_level": record.confidence_level,
                "exclusion_reason": exclusion_reason,
                "source_row_ref": record.provenance.raw_payload_ref,
                "raw_payload_sha256": record.provenance.raw_payload_sha256,
                "query_sha256": record.provenance.query_sha256,
                "normalized_record_json": _json(promoter_record_to_dict(record)),
            }
        )


def _validate_promoter_conflicts(records: list[PromoterRecord]) -> list[dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    by_promoter: dict[tuple[str, str], set[str]] = {}
    for record in records:
        by_promoter.setdefault((record.source_release, record.promoter_id), set()).add(_canonical_usr_sequence(record))
    for (release, promoter_id), sequences in sorted(by_promoter.items()):
        if len(sequences) <= 1:
            continue
        conflicts.append(
            {
                "source_release": release,
                "promoter_id": promoter_id,
                "field": "sequence",
                "severity": "fatal",
                "details": f"{len(sequences)} conflicting canonical sequence values",
            }
        )
    if conflicts:
        first = conflicts[0]
        raise SchemaError(
            "RegulonDB promoter id has conflicting canonical sequence: "
            f"{first['source_release']}:{first['promoter_id']}"
        )
    return conflicts


def _dedupe_relation_rows(
    rows: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, int]]:
    key_fields = {
        "promoter_aliases": (
            "usr_id",
            "source_release",
            "source_route",
            "source_table",
            "source_stratum",
            "promoter_id",
            "promoter_name",
            "raw_sequence_sha256",
            "strand",
            "tss_raw",
        ),
        "sigma_affiliations": (
            "usr_id",
            "promoter_id",
            "source_release",
            "source_route",
            "sigma_id",
            "sigma_name",
            "sigma_abbrev",
            "sigma_canonical_label",
            "sigma_gene_id",
            "sigma_gene_name",
            "evidence",
            "confidence",
            "citation_refs",
        ),
    }

    def _key_value(value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, dict):
            return _json(value)
        return value

    deduped: dict[str, list[dict[str, object]]] = {}
    collapsed: dict[str, int] = {}
    for name, relation_rows in rows.items():
        fields = key_fields.get(name)
        if fields is None:
            deduped[name] = list(relation_rows)
            collapsed[name] = 0
            continue
        seen: set[tuple[object, ...]] = set()
        unique_rows: list[dict[str, object]] = []
        for row in relation_rows:
            key = tuple(_key_value(row.get(field)) for field in fields)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        deduped[name] = unique_rows
        collapsed[name] = len(relation_rows) - len(unique_rows)
    return deduped, collapsed


def _refresh_overlay_regulatory_context(
    overlay_rows: list[dict[str, object]],
    relation_rows: Mapping[str, list[dict[str, object]]],
) -> None:
    interactions_by_usr: dict[str, list[dict[str, object]]] = {}
    for row in relation_rows.get("regulatory_interactions", []):
        usr_id = str(row.get("usr_id") or "").strip()
        if usr_id:
            interactions_by_usr.setdefault(usr_id, []).append(row)
    for row in overlay_rows:
        usr_id = str(row.get("id") or "").strip()
        interactions = interactions_by_usr.get(usr_id, [])
        if not interactions:
            continue
        functions = [str(interaction.get("function") or "") for interaction in interactions]
        composition = _regulator_composition(functions)
        regulons = _list_unique(
            str(value)
            for interaction in interactions
            for value in (interaction.get("regulon_id"), interaction.get("regulon_name"))
            if value
        )
        has_citations = bool(row.get("regulondb__has_citations")) or any(
            interaction.get("citation_refs") for interaction in interactions
        )
        row["regulondb__has_regulatory_context"] = True
        row["regulondb__regulator_composition"] = composition
        row["regulondb__regulon_count"] = len(regulons)
        row["regulondb__has_activator"] = composition in {"activator", "mixed"}
        row["regulondb__has_repressor"] = composition in {"repressor", "mixed"}
        row["regulondb__has_citations"] = has_citations


def _relation_sidecar_integrity_counts(
    base_rows: list[dict[str, object]], relation_rows: Mapping[str, list[dict[str, object]]]
) -> tuple[int, int]:
    base_ids = {str(row["id"]) for row in base_rows}
    orphan_count = 0
    missing_usr_id_count = 0
    for name, rows in relation_rows.items():
        schema = _RELATION_SCHEMAS.get(name)
        if schema is None:
            raise SchemaError(f"Unknown RegulonDB relation sidecar: {name}")
        if schema.get_field_index("usr_id") < 0:
            continue
        for row in rows:
            usr_id = row.get("usr_id")
            if usr_id is None:
                missing_usr_id_count += 1
            elif str(usr_id) not in base_ids:
                orphan_count += 1
    return orphan_count, missing_usr_id_count


def _validate_relation_sidecar_integrity(
    base_rows: list[dict[str, object]], relation_rows: Mapping[str, list[dict[str, object]]]
) -> None:
    orphan_count, missing_usr_id_count = _relation_sidecar_integrity_counts(base_rows, relation_rows)
    if orphan_count or missing_usr_id_count:
        raise SchemaError(
            "RegulonDB relation sidecars contain invalid usr_id references: "
            f"{orphan_count} orphan rows, {missing_usr_id_count} missing usr_id rows"
        )


def _duplicate_relation_row_counts(relation_rows: Mapping[str, list[dict[str, object]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, rows in relation_rows.items():
        if name in {"source_rows", "excluded_source_rows", "skipped_source_rows"}:
            counts[name] = 0
            continue
        seen: set[str] = set()
        duplicates = 0
        for row in rows:
            key = _json(row)
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
        counts[name] = duplicates
    return counts


def _base_sequence_fidelity_counts(base_rows: list[dict[str, object]]) -> dict[str, int]:
    sequences = [str(row.get("sequence") or "") for row in base_rows]
    sequence_counts = Counter(sequences)
    invalid_count = sum(1 for sequence in sequences if not sequence or set(sequence.upper()) - _STRICT_DNA)
    length_mismatch_count = 0
    for row in base_rows:
        sequence = str(row.get("sequence") or "")
        length = row.get("length")
        if length is not None and int(length) != len(sequence):
            length_mismatch_count += 1
    return {
        "base_duplicate_sequence_count": sum(count - 1 for count in sequence_counts.values() if count > 1),
        "base_invalid_sequence_count": invalid_count,
        "base_length_mismatch_count": length_mismatch_count,
    }


_REQUIRED_OVERLAY_METADATA_COLUMNS = (
    "regulondb__release",
    "regulondb__primary_source",
    "regulondb__source_strata_set",
    "regulondb__promoter_alias_count",
    "regulondb__sigma_factor_set",
    "regulondb__sigma_factor_count",
    "regulondb__confidence_level_set",
    "regulondb__has_minus10_box",
    "regulondb__has_minus35_box",
    "regulondb__box_pattern",
    "regulondb__regulator_composition",
    "regulondb__regulon_count",
    "regulondb__tss_available",
    "regulondb__metadata_completeness_class",
    "regulondb__raw_sequence_case_policy",
    "regulondb__has_sequence",
    "regulondb__has_tss",
    "regulondb__has_sigma",
    "regulondb__has_confidence",
    "regulondb__has_boxes",
    "regulondb__has_regulatory_context",
    "regulondb__has_citations",
)


def _is_missing_overlay_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return False
    return False


def _required_overlay_metadata_null_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in _REQUIRED_OVERLAY_METADATA_COLUMNS:
        missing = sum(1 for row in rows if column not in row or _is_missing_overlay_value(row.get(column)))
        if missing:
            counts[column] = missing
    return counts


def _sigma_fidelity_counts(
    base_rows: list[dict[str, object]],
    overlay_rows: list[dict[str, object]],
    relation_rows: Mapping[str, list[dict[str, object]]],
) -> dict[str, object]:
    sigma_rows = relation_rows.get("sigma_affiliations", [])
    sigma_missing_base_row_count = sum(1 for row in overlay_rows if not row.get("regulondb__has_sigma"))
    alias_usr_ids_without_sigma = {str(row["id"]) for row in overlay_rows if not row.get("regulondb__has_sigma")}
    sigma_missing_alias_row_count = sum(
        1 for row in relation_rows.get("promoter_aliases", []) if str(row.get("usr_id")) in alias_usr_ids_without_sigma
    )
    empty_label_count = sum(
        1
        for row in sigma_rows
        if not any(
            str(row.get(field) or "").strip()
            for field in ("sigma_id", "sigma_name", "sigma_abbrev", "sigma_canonical_label")
        )
    )
    label_counts = Counter(
        str(
            row.get("sigma_canonical_label")
            or row.get("sigma_abbrev")
            or row.get("sigma_name")
            or row.get("sigma_id")
            or "unknown"
        )
        for row in sigma_rows
    )
    return {
        "sigma_missing_base_row_count": sigma_missing_base_row_count,
        "sigma_missing_alias_row_count": sigma_missing_alias_row_count,
        "sigma_empty_label_row_count": empty_label_count,
        "sigma_label_counts": dict(sorted(label_counts.items())),
        "sigma_factor_set_count_mismatch": sum(
            1
            for row in overlay_rows
            if len(row.get("regulondb__sigma_factor_set") or []) != row.get("regulondb__sigma_factor_count")
        ),
        "base_row_count_checked_for_sigma": len(base_rows),
    }


def _normalize_promoter_name_for_match(value: object) -> str:
    text = str(value or "").casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def _fuzzy_promoter_name_collisions(
    relation_rows: Mapping[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    name_rows = [
        {
            "usr_id": str(row.get("usr_id") or ""),
            "promoter_id": row.get("promoter_id"),
            "promoter_name": row.get("promoter_name"),
            "source_release": row.get("source_release"),
            "source_route": row.get("source_route"),
            "normalized_name": _normalize_promoter_name_for_match(row.get("promoter_name")),
        }
        for row in relation_rows.get("promoter_aliases", [])
        if row.get("promoter_name")
    ]
    by_normalized: dict[str, list[dict[str, object]]] = {}
    for row in name_rows:
        normalized = str(row["normalized_name"])
        if normalized:
            by_normalized.setdefault(normalized, []).append(row)

    collisions: list[dict[str, object]] = []
    for normalized, rows in sorted(by_normalized.items()):
        usr_ids = {row["usr_id"] for row in rows}
        names = _list_unique(str(row.get("promoter_name") or "") for row in rows)
        if len(usr_ids) <= 1 or len(names) <= 1:
            continue
        collisions.append(
            {
                "kind": "normalized_name_match",
                "normalized_name": normalized,
                "left_name": names[0],
                "right_name": names[1],
                "usr_id_count": len(usr_ids),
                "source_releases": _list_unique(str(row.get("source_release") or "") for row in rows),
            }
        )
        if len(collisions) >= _FUZZY_NAME_COLLISION_LIMIT:
            return collisions

    representatives: dict[str, dict[str, object]] = {}
    for row in name_rows:
        normalized = str(row["normalized_name"])
        if normalized and normalized not in representatives:
            representatives[normalized] = row
    buckets: dict[str, list[str]] = {}
    for normalized in representatives:
        buckets.setdefault(normalized[:3], []).append(normalized)
    for names in buckets.values():
        ordered = sorted(names)
        for idx, left in enumerate(ordered):
            for right in ordered[idx + 1 :]:
                if abs(len(left) - len(right)) > 2:
                    continue
                left_row = representatives[left]
                right_row = representatives[right]
                if left_row["usr_id"] == right_row["usr_id"]:
                    continue
                ratio = difflib.SequenceMatcher(a=left, b=right).ratio()
                if ratio < _FUZZY_NAME_RATIO_THRESHOLD:
                    continue
                collisions.append(
                    {
                        "kind": "fuzzy_name_match",
                        "left_name": left_row["promoter_name"],
                        "right_name": right_row["promoter_name"],
                        "left_normalized_name": left,
                        "right_normalized_name": right,
                        "ratio": round(ratio, 3),
                        "left_usr_id": left_row["usr_id"],
                        "right_usr_id": right_row["usr_id"],
                    }
                )
                if len(collisions) >= _FUZZY_NAME_COLLISION_LIMIT:
                    return collisions
    return collisions


def build_import_plan(
    *,
    export_dir: Path,
    usr_root: Path,
    output_dataset: str = DEFAULT_OUTPUT_DATASET,
    created_at: str | None = None,
    require_promoter_associations: bool = False,
) -> NativePromoterImportPlan:
    _check_regulondb_namespace(usr_root)
    manifest, records = load_promoter_export(export_dir)
    promoter_associations = load_promoter_regulatory_associations(export_dir)
    if require_promoter_associations and not promoter_associations:
        raise SchemaError("Cruncher promoter association artifact is missing or empty.")
    conflicts = _validate_promoter_conflicts(records)
    if not manifest.complete:
        raise SchemaError("Cruncher promoter export is incomplete; USR import requires a complete export.")
    created = created_at or now_utc()
    groups: dict[str, list[PromoterRecord]] = {}
    sequences_by_id: dict[str, str] = {}
    for record in records:
        sequence = _canonical_usr_sequence(record)
        usr_id = compute_id("dna", sequence)
        groups.setdefault(usr_id, []).append(record)
        sequences_by_id[usr_id] = sequence

    base_rows: list[dict[str, object]] = []
    overlay_rows: list[dict[str, object]] = []
    relation_rows = _empty_relations()
    skipped_by_reason = _add_skipped_source_rows(export_dir, relation_rows)
    included_record_count = 0
    excluded_by_reason: Counter[str] = Counter()
    included_group_ids: set[str] = set()
    for usr_id in sorted(groups):
        grouped = sorted(
            groups[usr_id],
            key=lambda record: (record.source_release, record.source_route, record.promoter_id),
        )
        exclusion_reason = _base_group_exclusion_reason(grouped)
        if exclusion_reason is not None:
            excluded_by_reason[exclusion_reason] += len(grouped)
            _add_excluded_source_rows(grouped, relation_rows, exclusion_reason=exclusion_reason)
            continue
        sequence = sequences_by_id[usr_id]
        included_record_count += len(grouped)
        included_group_ids.add(usr_id)
        base_rows.append(
            {
                "id": usr_id,
                "bio_type": "dna",
                "sequence": sequence,
                "alphabet": "dna_4",
                "length": len(sequence),
                "source": _base_source(grouped),
                "created_at": created,
            }
        )
        overlay_rows.append(_overlay_row(usr_id, grouped))
        _relation_rows_for_group(usr_id, grouped, relation_rows)
    if not base_rows:
        raise SchemaError(
            "No RegulonDB promoter records passed strict USR base-row metadata policy: "
            f"required={list(_BASE_ROW_REQUIRED_METADATA)}"
        )
    association_projection = _project_promoter_associations(promoter_associations, relation_rows)
    relation_rows, dedupe_counts = _dedupe_relation_rows(relation_rows)
    if require_promoter_associations and association_projection["promoter_association_matched_rows"] == 0:
        raise SchemaError("Required promoter association artifact produced no matched USR regulatory interactions.")
    _refresh_overlay_regulatory_context(overlay_rows, relation_rows)
    _validate_relation_sidecar_integrity(base_rows, relation_rows)
    orphan_relation_rows, missing_relation_usr_ids = _relation_sidecar_integrity_counts(base_rows, relation_rows)
    aliases_by_usr = {
        usr_id: sum(1 for row in relation_rows["promoter_aliases"] if row["usr_id"] == usr_id)
        for usr_id in included_group_ids
    }
    for row in overlay_rows:
        usr_id = str(row["id"])
        row["regulondb__promoter_alias_count"] = aliases_by_usr.get(usr_id, 0)

    sequence_fidelity = _base_sequence_fidelity_counts(base_rows)
    duplicate_relation_row_counts = _duplicate_relation_row_counts(relation_rows)
    fuzzy_name_collisions = _fuzzy_promoter_name_collisions(relation_rows)
    sigma_fidelity = _sigma_fidelity_counts(base_rows, overlay_rows, relation_rows)
    validation_report = {
        "export_schema_version": manifest.schema_version,
        "export_record_count": len(records),
        "included_export_record_count": included_record_count,
        "base_row_count": len(base_rows),
        "duplicate_sequence_collapses": included_record_count - len(base_rows),
        "base_row_required_metadata": list(_BASE_ROW_REQUIRED_METADATA),
        "metadata_excluded_base_row_count": len(groups) - len(base_rows),
        "metadata_excluded_source_row_count": len(relation_rows["excluded_source_rows"]),
        "metadata_excluded_source_rows_by_reason": dict(sorted(excluded_by_reason.items())),
        **sequence_fidelity,
        "duplicate_promoter_alias_collapses": dedupe_counts.get("promoter_aliases", 0),
        "duplicate_sigma_affiliation_collapses": dedupe_counts.get("sigma_affiliations", 0),
        "duplicate_relation_row_counts": duplicate_relation_row_counts,
        "skipped_source_row_count": len(relation_rows["skipped_source_rows"]),
        "skipped_source_rows_by_reason": dict(sorted(skipped_by_reason.items())),
        **association_projection,
        "orphan_relation_row_count": orphan_relation_rows,
        "missing_relation_usr_id_count": missing_relation_usr_ids,
        "required_overlay_metadata_null_counts": _required_overlay_metadata_null_counts(overlay_rows),
        **sigma_fidelity,
        "fuzzy_promoter_name_collision_count": len(fuzzy_name_collisions),
        "fuzzy_promoter_name_collisions": fuzzy_name_collisions,
        "relation_counts": {name: len(rows) for name, rows in sorted(relation_rows.items())},
        "sig35_variant_absent": all("sig35_variant" not in row for row in [*base_rows, *overlay_rows]),
    }
    return NativePromoterImportPlan(
        dataset=output_dataset,
        export_dir=str(export_dir),
        source_manifest_complete=manifest.complete,
        base_rows=base_rows,
        regulondb_overlay_rows=overlay_rows,
        relation_rows=relation_rows,
        validation_report=validation_report,
        conflict_report=conflicts,
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
    row_list = [{field.name: row.get(field.name) for field in schema} for row in rows]
    return pa.Table.from_pylist(row_list, schema=schema)


def _write_regulondb_overlay(dataset: Dataset, rows: list[dict[str, object]]) -> int:
    schema = _namespace_schema(dataset.root, REGULONDB_NAMESPACE, rows)
    table = _table_from_rows(schema, rows)
    with dataset.write_session() as session:
        return session.write_overlay(REGULONDB_NAMESPACE, table, key="id", note="regulondb_native_promoter_import")


def _write_relation_sidecars(dataset_dir: Path, relation_rows: Mapping[str, list[dict[str, object]]]) -> dict[str, int]:
    relation_dir = dataset_dir / RELATIONS_DIRNAME
    tmp_relation_dir = dataset_dir / f".{RELATIONS_DIRNAME}.tmp"
    if relation_dir.exists():
        raise FileExistsError(f"Relation sidecar directory already exists: {relation_dir}")
    if tmp_relation_dir.exists():
        raise FileExistsError(f"Stale relation sidecar staging directory exists: {tmp_relation_dir}")
    tmp_relation_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    try:
        for name, schema in sorted(_RELATION_SCHEMAS.items()):
            rows = relation_rows.get(name, [])
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, tmp_relation_dir / f"{name}.parquet")
            counts[name] = len(rows)
        tmp_relation_dir.replace(relation_dir)
    except Exception:
        shutil.rmtree(tmp_relation_dir, ignore_errors=True)
        raise
    return counts


def _promoter_alias_ids_by_usr(relation_rows: Mapping[str, list[dict[str, object]]]) -> dict[str, list[str]]:
    aliases_by_usr: dict[str, list[str]] = {}
    seen_by_usr: dict[str, set[str]] = {}
    for row in relation_rows.get("promoter_aliases", []):
        usr_id = str(row.get("usr_id") or "").strip()
        promoter_id = str(row.get("promoter_id") or "").strip()
        if not usr_id or not promoter_id:
            continue
        seen = seen_by_usr.setdefault(usr_id, set())
        key = promoter_id.casefold()
        if key in seen:
            continue
        seen.add(key)
        aliases_by_usr.setdefault(usr_id, []).append(promoter_id)
    return {usr_id: sorted(values) for usr_id, values in aliases_by_usr.items()}


def _regulondb_source_record_views(
    plan: NativePromoterImportPlan,
    *,
    created_at: str,
) -> list[SequenceViewRecord]:
    aliases_by_usr = _promoter_alias_ids_by_usr(plan.relation_rows)
    rows: list[SequenceViewRecord] = []
    for row in plan.base_rows:
        sequence_id = str(row["id"])
        aliases = aliases_by_usr.get(sequence_id) or None
        if aliases and len(aliases) == 1:
            view_name = f"{aliases[0]}_source_record"
        else:
            view_name = f"regulondb_source_record_{sequence_id[:12]}"
        rows.append(
            SequenceViewRecord(
                sequence_id=sequence_id,
                view_name=view_name,
                aliases=aliases,
                product_kind="source_record",
                context_kind="native_reference",
                orientation="unknown",
                analysis_only=False,
                source_dataset_id=plan.dataset,
                source_label="regulondb_native_promoter",
                source_interval_start_0=0,
                source_interval_end_0=int(row["length"]),
                recommended_pooling="seq_mean",
                created_at=created_at,
                created_by=CREATED_BY,
            )
        )
    return rows


def _regulondb_source_record_view_semantics(
    views: Iterable[SequenceViewRecord],
    *,
    created_at: str,
) -> list[ViewSemanticsRecord]:
    return [
        ViewSemanticsRecord(
            view_id=str(view.view_id),
            sequence_id=view.sequence_id,
            source_family="regulondb_native_promoter",
            selection_basis="regulondb_curated_promoter_sequence_with_sigma",
            view_collections=["native_promoter_source_records"],
            role_tags=["native_promoter_source", "reference_source"],
            study_id=None,
            created_at=created_at,
            created_by=CREATED_BY,
        )
        for view in views
    ]


def write_import_plan(plan: NativePromoterImportPlan, *, usr_root: Path) -> WriteResult:
    _validate_relation_sidecar_integrity(plan.base_rows, plan.relation_rows)
    dataset = Dataset(usr_root, plan.dataset)
    if dataset.dir.exists():
        raise FileExistsError(f"Output dataset already exists: {dataset.dir}")
    with dataset.write_session() as session:
        session.init(
            source="cruncher regulondb native promoter export",
            notes="Native RegulonDB promoter dataset imported from a deterministic Cruncher export.",
        )
        rows_written = session.import_rows(plan.base_rows, source="regulondb_native_promoter_import")
    overlay_count = _write_regulondb_overlay(dataset, plan.regulondb_overlay_rows)
    created_at = now_utc()
    source_record_views = _regulondb_source_record_views(plan, created_at=created_at)
    source_record_semantics = _regulondb_source_record_view_semantics(source_record_views, created_at=created_at)
    actor = {"tool": CREATED_BY, "run_id": "regulondb_native_promoter_import", "dataset": plan.dataset}
    sequence_view_count = write_sequence_views(dataset, source_record_views, conflict_policy="error", actor=actor)
    view_semantics_count = write_view_semantics(dataset, source_record_semantics, conflict_policy="error", actor=actor)
    relation_counts = _write_relation_sidecars(dataset.dir, plan.relation_rows)
    with dataset.maintenance("materialize_regulondb_native_promoter_overlay"):
        dataset.materialize(namespaces=[REGULONDB_NAMESPACE], keep_overlays=True)
    dataset.log_event(
        "regulondb_native_promoter_import",
        args={"export_dir": plan.export_dir, "output_dataset": plan.dataset},
        metrics={
            "base_rows": len(plan.base_rows),
            "regulondb_overlay_rows": len(plan.regulondb_overlay_rows),
            "sequence_view_rows": int(sequence_view_count),
            "view_semantics_rows": int(view_semantics_count),
            "relation_rows": sum(relation_counts.values()),
        },
        artifacts={
            "relation_sidecars": sorted(relation_counts),
            "sequence_views": "_views/sequence_views.parquet",
            "view_semantics": "_views/view_semantics.parquet",
        },
    )
    dataset.validate(strict=True)
    return WriteResult(
        dataset=dataset.name,
        dataset_dir=str(dataset.dir),
        rows_written=int(rows_written),
        regulondb_overlay_rows=int(overlay_count),
        sequence_view_rows=int(sequence_view_count),
        view_semantics_rows=int(view_semantics_count),
        relation_sidecars=relation_counts,
    )


def run_import(
    *,
    export_dir: Path,
    usr_root: Path,
    output_dataset: str = DEFAULT_OUTPUT_DATASET,
    write: bool = False,
    require_promoter_associations: bool = False,
) -> dict[str, object]:
    plan = build_import_plan(
        export_dir=export_dir,
        usr_root=usr_root,
        output_dataset=output_dataset,
        require_promoter_associations=require_promoter_associations,
    )
    payload: dict[str, object] = {"write": bool(write), "plan": plan.summary()}
    if write:
        result = write_import_plan(plan, usr_root=usr_root)
        payload["result"] = result.__dict__
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create usr_regulondb_native_promoters from a deterministic Cruncher promoter export."
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--output-dataset", default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--write", action="store_true", help="Create the output dataset. Default is dry-run.")
    parser.add_argument(
        "--require-promoter-associations",
        action="store_true",
        help="Fail if the Cruncher export has no matched promoter regulatory association overlay.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_import(
        export_dir=args.export_dir,
        usr_root=args.usr_root,
        output_dataset=args.output_dataset,
        write=bool(args.write),
        require_promoter_associations=bool(args.require_promoter_associations),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
