"""
RegulonDB promoter intake contracts and deterministic export helpers.

This module is intentionally source-facing only: it normalizes promoter-shaped
source payloads for Cruncher and writes a stable export/cache for USR to import
offline. It does not create USR datasets or study records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .promoter_associations import (
    discover_dnadesign_data_promoter_association_sources,
    parse_promoter_association_source_file,
    parse_regulondb_network_tf_tu_associations,  # noqa: F401
    parse_regulondb_tf_riset_associations,  # noqa: F401
)
from .promoter_contracts import (
    _RELATION_NAMES,
    PROMOTER_EXPORT_SCHEMA_VERSION,
    PROMOTER_PARSER_VERSION,
    GeneRef,
    OperonRef,
    PromoterAssociationSourceFile,  # noqa: F401
    PromoterBox,
    PromoterCollectionSummary,
    PromoterDescriptor,
    PromoterExportManifest,
    PromoterQuery,
    PromoterRecord,
    PromoterRegulatoryAssociation,
    PromoterRegulatorySite,
    PromoterSchemaError,
    PromoterSigmaAffiliation,
    PromoterSourceFile,
    PromoterSourceInventory,
    PromoterSourceTriageReport,
    SkippedPromoterSourceRow,
    SourceProvenance,
    TranscriptionUnitRef,
)
from .promoter_payloads import (
    _float_or_none,
    _list_payload,
    _sha256_json,
    _text_or_none,
    parse_regulondb_promoter_payload,
)
from .promoter_tables import _iter_delimited_data_rows, _missing_table_value, _split_table_list, _table_value


def _local_box_payload(row: Mapping[str, str], *, kind: str) -> dict[str, Any] | None:
    prefix = "minus_35" if kind == "minus_35" else "minus_10"
    label = "35" if kind == "minus_35" else "10"
    sequence = _table_value(row, f"{prefix}_sequence", f"box_{label}_sequence", f"{label}_box_sequence")
    left = _table_value(row, f"{prefix}_left", f"{prefix}_start", f"box_{label}_left", f"{label}_box_left")
    right = _table_value(row, f"{prefix}_right", f"{prefix}_end", f"box_{label}_right", f"{label}_box_right")
    if sequence is None and left is None and right is None:
        return None
    return {
        "type": kind,
        "sequence": sequence,
        "leftEndPosition": left,
        "rightEndPosition": right,
        "strand": _table_value(row, f"{prefix}_strand", f"box_{label}_strand"),
    }


def _local_sigma_payloads(row: Mapping[str, str]) -> list[dict[str, Any]]:
    sigma_ids = _split_table_list(_table_value(row, "sigma_factor_id", "sigma_id"))
    sigma_names = _split_table_list(_table_value(row, "sigma_factor_name", "sigma_name", "sigma"))
    sigma_abbrevs = _split_table_list(
        _table_value(row, "sigma_factor_abbrev", "sigma_abbrev", "sigma_factor", "sigmafactor", "sigmaf")
    )
    gene_ids = _split_table_list(_table_value(row, "sigma_gene_id"))
    gene_names = _split_table_list(_table_value(row, "sigma_gene_name", "sigma_gene"))
    count = max(len(sigma_ids), len(sigma_names), len(sigma_abbrevs), len(gene_ids), len(gene_names), 0)
    payloads: list[dict[str, Any]] = []
    for index in range(count):
        payloads.append(
            {
                "_id": sigma_ids[index] if index < len(sigma_ids) else None,
                "name": sigma_names[index] if index < len(sigma_names) else None,
                "abbreviatedName": sigma_abbrevs[index] if index < len(sigma_abbrevs) else None,
                "gene": {
                    "_id": gene_ids[index] if index < len(gene_ids) else None,
                    "name": gene_names[index] if index < len(gene_names) else None,
                },
                "evidence": list(_split_table_list(_table_value(row, "sigma_evidence"))),
                "citations": list(_split_table_list(_table_value(row, "sigma_citations", "sigma_references"))),
            }
        )
    return payloads


def _local_promoter_payload(row: Mapping[str, str]) -> dict[str, Any]:
    boxes = [
        box
        for box in (
            _local_box_payload(row, kind="minus_35"),
            _local_box_payload(row, kind="minus_10"),
        )
        if box is not None
    ]
    gene_id = _table_value(row, "first_gene_id", "gene_id")
    gene_name = _table_value(row, "first_gene_name", "firstgenename", "gene_name", "first_gene")
    tu_id = _table_value(row, "tu_id", "transcription_unit_id")
    tu_name = _table_value(row, "tu_name", "transcription_unit_name")
    operon_id = _table_value(row, "operon_id")
    operon_name = _table_value(row, "operon_name")
    payload: dict[str, Any] = {
        "_id": _table_value(row, "promoter_id", "id", "promoterid", "pmid"),
        "name": _table_value(row, "promoter_name", "name", "pmname"),
        "sequence": _table_value(row, "sequence", "promoter_sequence", "promotersequence", "pmsequence", "seq"),
        "strand": _table_value(row, "strand"),
        "posTSS": _table_value(row, "posTSS", "pos_tss", "tss_position", "tss"),
        "confidenceLevel": _table_value(row, "confidence_level", "confidence", "confidenceLevel"),
        "score": _table_value(row, "score", "prediction_score"),
        "evidence": list(_split_table_list(_table_value(row, "evidence", "evidence_codes", "pmevidence"))),
        "citations": list(_split_table_list(_table_value(row, "citations", "references", "pmids", "pmid"))),
        "sigmaFactors": _local_sigma_payloads(row),
        "boxes": boxes,
    }
    if gene_id is not None or gene_name is not None:
        payload["firstGene"] = {"_id": gene_id, "name": gene_name}
    if tu_id is not None or tu_name is not None:
        payload["transcriptionUnits"] = [{"_id": tu_id, "name": tu_name}]
    if operon_id is not None or operon_name is not None:
        payload["operon"] = {"_id": operon_id, "name": operon_name}
    return payload


def _parse_regulondb_promoter_set_rows(
    path: Path,
    *,
    source_release: str,
    source_route: str,
    source_stratum: str,
    delimiter: str,
    fetched_at: datetime | None = None,
    source_release_date: str | None = None,
    skip_missing_sequence: bool = False,
    skipped_rows: list[SkippedPromoterSourceRow] | None = None,
) -> list[PromoterRecord]:
    records: list[PromoterRecord] = []
    query_base = {
        "source_table": path.name,
        "source_release": source_release,
        "source_route": source_route,
    }
    for line_number, row in _iter_delimited_data_rows(path, delimiter=delimiter):
        sequence = _table_value(row, "sequence", "promoter_sequence", "promotersequence", "pmsequence", "seq")
        if skip_missing_sequence and _missing_table_value(sequence):
            if skipped_rows is not None:
                query = {**query_base, "line_number": line_number}
                payload = _local_promoter_payload(row)
                skipped_rows.append(
                    SkippedPromoterSourceRow(
                        source="regulondb",
                        source_release=str(source_release),
                        source_release_date=source_release_date,
                        source_route=source_route,
                        source_table=path.name,
                        source_stratum=source_stratum,
                        promoter_id=_text_or_none(payload.get("_id")),
                        promoter_name=_text_or_none(payload.get("name")),
                        raw_sequence=_text_or_none(sequence),
                        skip_reason="missing_sequence",
                        source_row_ref=f"{path}:{line_number}",
                        raw_payload_sha256=_sha256_json({"source_table": path.name, "row": row}),
                        query_sha256=_sha256_json(query),
                    )
                )
            continue
        payload = _local_promoter_payload(row)
        records.append(
            parse_regulondb_promoter_payload(
                payload,
                source_release=source_release,
                source_release_date=source_release_date,
                source_route=source_route,
                query={**query_base, "line_number": line_number},
                fetched_at=fetched_at,
                source_table=path.name,
                raw_payload_ref=f"{path}:{line_number}",
                source_stratum=source_stratum,
            )
        )
    return records


def parse_regulondb_promoter_set_tsv(
    path: Path,
    *,
    source_release: str,
    fetched_at: datetime | None = None,
    source_release_date: str | None = None,
    source_route: str = "local_promoter_set",
    source_stratum: str = "local_release_pinned_curated",
    skip_missing_sequence: bool = False,
    skipped_rows: list[SkippedPromoterSourceRow] | None = None,
) -> list[PromoterRecord]:
    """Parse a release-pinned local RegulonDB PromoterSet.tsv-like table."""
    return _parse_regulondb_promoter_set_rows(
        path,
        source_release=source_release,
        source_release_date=source_release_date,
        source_route=source_route,
        source_stratum=source_stratum,
        delimiter="\t",
        fetched_at=fetched_at,
        skip_missing_sequence=skip_missing_sequence,
        skipped_rows=skipped_rows,
    )


def parse_regulondb_promoter_set_csv(
    path: Path,
    *,
    source_release: str,
    fetched_at: datetime | None = None,
    source_release_date: str | None = None,
    source_route: str = "local_promoter_set",
    source_stratum: str = "historical_curated_release",
    skip_missing_sequence: bool = False,
    skipped_rows: list[SkippedPromoterSourceRow] | None = None,
) -> list[PromoterRecord]:
    """Parse a release-pinned local RegulonDB PromoterSet.csv-like table."""
    return _parse_regulondb_promoter_set_rows(
        path,
        source_release=source_release,
        source_release_date=source_release_date,
        source_route=source_route,
        source_stratum=source_stratum,
        delimiter=",",
        fetched_at=fetched_at,
        skip_missing_sequence=skip_missing_sequence,
        skipped_rows=skipped_rows,
    )


def discover_dnadesign_data_promoter_sources(
    *,
    data_root: Path | None = None,
    provider: Any | None = None,
) -> tuple[PromoterSourceFile, ...]:
    """Return promoter source files from the public dnadesign-data source surface."""

    if provider is None:
        try:
            from dnadesign_data.regulatory_parts import iter_promoter_source_files
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "dnadesign-data promoter source discovery requires the dnadesign_data package. "
                "Install sibling dnadesign-data or pass an explicit provider."
            ) from exc
        provider = iter_promoter_source_files
    discovered = []
    for item in provider(data_root):
        if isinstance(item, PromoterSourceFile):
            discovered.append(item)
            continue
        if isinstance(item, Mapping):
            discovered.append(PromoterSourceFile(**dict(item)))
            continue
        discovered.append(PromoterSourceFile(**asdict(item)))
    return tuple(discovered)


def parse_promoter_source_file(
    source: PromoterSourceFile,
    *,
    data_root: Path,
    fetched_at: datetime | None = None,
    skipped_rows: list[SkippedPromoterSourceRow] | None = None,
) -> list[PromoterRecord]:
    """Parse source descriptors emitted by dnadesign-data into normalized promoter records."""

    path = data_root / source.path
    if source.parser_hint != "regulondb_promoter_set":
        return []
    if source.file_format == "tsv":
        return parse_regulondb_promoter_set_tsv(
            path,
            source_release=source.release,
            source_route=source.source_id,
            source_stratum=source.stratum,
            fetched_at=fetched_at,
            skip_missing_sequence=True,
            skipped_rows=skipped_rows,
        )
    if source.file_format == "csv":
        return parse_regulondb_promoter_set_csv(
            path,
            source_release=source.release,
            source_route=source.source_id,
            source_stratum=source.stratum,
            fetched_at=fetched_at,
            skip_missing_sequence=True,
            skipped_rows=skipped_rows,
        )
    return []


def _resolve_dnadesign_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    try:
        from dnadesign_data.regulatory_parts import default_data_root
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "dnadesign-data promoter source export requires an explicit data_root or the dnadesign_data package."
        ) from exc
    return default_data_root()


def _source_file_dict(source: PromoterSourceFile) -> dict[str, Any]:
    return asdict(source)


def export_dnadesign_data_promoter_superset(
    destination: Path,
    *,
    data_root: Path | None = None,
    provider: Any | None = None,
    association_provider: Any | None = None,
    require_association_sources: bool = False,
    fetched_at: datetime | None = None,
) -> PromoterExportManifest:
    """Export a single provenance-qualified promoter superset from dnadesign-data.

    The public dnadesign-data API owns source-file discovery only. Cruncher owns
    parsing and normalization. Supplemental non-base sources are recorded in the
    export manifest side artifacts and deliberately deferred until they can be
    reconciled into relation sidecars without creating duplicate promoter
    sequence rows.
    """

    resolved_root = _resolve_dnadesign_data_root(data_root)
    sources = discover_dnadesign_data_promoter_sources(data_root=resolved_root, provider=provider)
    association_sources = discover_dnadesign_data_promoter_association_sources(
        data_root=resolved_root,
        provider=association_provider,
        required=require_association_sources,
    )
    records: list[PromoterRecord] = []
    associations: list[PromoterRegulatoryAssociation] = []
    skipped_source_rows: list[SkippedPromoterSourceRow] = []
    source_inventory_rows: list[dict[str, Any]] = []
    for source in sources:
        if not source.creates_base_rows:
            source_inventory_rows.append(
                {
                    **_source_file_dict(source),
                    "parsed_record_count": 0,
                    "skipped_record_count": 0,
                    "skipped_reason": "non_base_source_deferred",
                }
            )
            continue
        skipped_before = len(skipped_source_rows)
        parsed = parse_promoter_source_file(
            source,
            data_root=resolved_root,
            fetched_at=fetched_at,
            skipped_rows=skipped_source_rows,
        )
        skipped_count = len(skipped_source_rows) - skipped_before
        records.extend(parsed)
        source_inventory_rows.append(
            {
                **_source_file_dict(source),
                "parsed_record_count": len(parsed),
                "skipped_record_count": skipped_count,
                "skipped_reason": None,
            }
        )
    for source in association_sources:
        parsed_associations = parse_promoter_association_source_file(
            source,
            data_root=resolved_root,
            fetched_at=fetched_at,
        )
        associations.extend(parsed_associations)
        source_inventory_rows.append(
            {
                **_source_file_dict(source),
                "parsed_record_count": 0,
                "skipped_record_count": 0,
                "parsed_association_count": len(parsed_associations),
                "skipped_reason": None,
            }
        )
    if require_association_sources and not associations:
        raise PromoterSchemaError(
            "No promoter regulatory associations were parsed from required promoter association sources."
        )
    if not records:
        raise PromoterSchemaError(
            "No base-row-capable promoter records were parsed from dnadesign-data source descriptors."
        )

    manifest = export_promoter_records(
        records,
        destination,
        query=PromoterQuery(
            source_release_policy="declared",
            routes=tuple(source.source_id for source in (*sources, *association_sources)),
            include_relations=True,
            source_stratum="dnadesign_data_superset",
        ),
        inventory=build_promoter_source_inventory(records),
        source_selection_status="dnadesign_data_superset",
        skipped_source_rows=skipped_source_rows,
    )
    _write_json(destination / "source_files.json", [_source_file_dict(source) for source in sources])
    _write_json(
        destination / "association_source_files.json", [_source_file_dict(source) for source in association_sources]
    )
    _write_jsonl(
        destination / "promoter_regulatory_associations.jsonl",
        (promoter_regulatory_association_to_dict(association) for association in associations),
    )
    _write_json(destination / "source_file_inventory.json", source_inventory_rows)
    manifest = PromoterExportManifest(
        schema_version=manifest.schema_version,
        parser_version=manifest.parser_version,
        export_created_at=manifest.export_created_at,
        complete=manifest.complete,
        record_count=manifest.record_count,
        source_selection_status=manifest.source_selection_status,
        source_inventory=manifest.source_inventory,
        query=manifest.query,
        artifacts={
            **manifest.artifacts,
            "source_files": "source_files.json",
            "association_source_files": "association_source_files.json",
            "promoter_regulatory_associations": "promoter_regulatory_associations.jsonl",
            "source_file_inventory": "source_file_inventory.json",
            "skipped_source_rows": "skipped_source_rows.jsonl",
        },
    )
    _write_json(destination / "manifest.json", manifest)
    return manifest


def build_promoter_source_inventory(
    records: Iterable[PromoterRecord],
    *,
    route_failure_count: int = 0,
) -> PromoterSourceInventory:
    record_list = list(records)
    total = len(record_list)

    def _rate(count: int) -> float:
        return 0.0 if total == 0 else count / total

    by_sequence: dict[str, int] = {}
    by_promoter: dict[str, set[str]] = {}
    for record in record_list:
        by_sequence[record.sequence] = by_sequence.get(record.sequence, 0) + 1
        by_promoter.setdefault(record.promoter_id, set()).add(record.sequence)
    duplicate_sequence_count = sum(count - 1 for count in by_sequence.values() if count > 1)
    conflict_count = sum(1 for sequences in by_promoter.values() if len(sequences) > 1)
    return PromoterSourceInventory(
        source_releases=tuple(sorted({record.source_release for record in record_list})),
        source_routes=tuple(sorted({record.source_route for record in record_list})),
        promoter_row_count=total,
        sequence_present_rate=_rate(sum(1 for record in record_list if record.sequence)),
        promoter_id_present_rate=_rate(sum(1 for record in record_list if record.promoter_id)),
        tss_present_rate=_rate(sum(1 for record in record_list if record.tss_interval_0based is not None)),
        sigma_present_rate=_rate(sum(1 for record in record_list if record.sigma_affiliations)),
        box_annotation_rate=_rate(sum(1 for record in record_list if record.boxes)),
        confidence_present_rate=_rate(sum(1 for record in record_list if record.confidence_level is not None)),
        regulatory_context_rate=_rate(sum(1 for record in record_list if record.regulatory_sites)),
        duplicate_sequence_count=duplicate_sequence_count,
        conflict_count=conflict_count,
        route_failure_count=route_failure_count,
    )


def build_promoter_descriptor_inventory(
    descriptors: Iterable[PromoterDescriptor],
    *,
    route_failure_count: int = 0,
) -> PromoterSourceInventory:
    descriptor_list = list(descriptors)
    total = len(descriptor_list)

    def _rate(count: int) -> float:
        return 0.0 if total == 0 else count / total

    return PromoterSourceInventory(
        source_releases=tuple(sorted({descriptor.source_release for descriptor in descriptor_list})),
        source_routes=tuple(sorted({descriptor.source_route for descriptor in descriptor_list})),
        promoter_row_count=total,
        sequence_present_rate=_rate(sum(1 for descriptor in descriptor_list if descriptor.sequence_present)),
        promoter_id_present_rate=_rate(sum(1 for descriptor in descriptor_list if descriptor.promoter_id)),
        tss_present_rate=_rate(sum(1 for descriptor in descriptor_list if descriptor.tss_present)),
        sigma_present_rate=_rate(sum(1 for descriptor in descriptor_list if descriptor.sigma_present)),
        box_annotation_rate=_rate(sum(1 for descriptor in descriptor_list if descriptor.box_annotation_present)),
        confidence_present_rate=_rate(sum(1 for descriptor in descriptor_list if descriptor.confidence_present)),
        regulatory_context_rate=0.0,
        duplicate_sequence_count=0,
        conflict_count=0,
        route_failure_count=route_failure_count,
    )


def _sigma_summary_label(sigma: PromoterSigmaAffiliation) -> str | None:
    return (
        _text_or_none(sigma.abbrev)
        or _text_or_none(sigma.name)
        or _text_or_none(sigma.sigma_id)
        or _text_or_none(sigma.gene_name)
        or _text_or_none(sigma.gene_id)
    )


def summarize_promoter_collection(records: Iterable[PromoterRecord]) -> PromoterCollectionSummary:
    """Summarize enumerated promoter records with deterministic sigma counts."""
    record_list = list(records)
    promoter_counts: dict[str, int] = {}
    sigma_counts: dict[str, int] = {}
    missing_sigma_count = 0
    multi_sigma_count = 0
    for record in record_list:
        promoter_counts[record.promoter_id] = promoter_counts.get(record.promoter_id, 0) + 1
        labels = tuple(
            dict.fromkeys(label for sigma in record.sigma_affiliations if (label := _sigma_summary_label(sigma)))
        )
        if not labels:
            missing_sigma_count += 1
            continue
        if len(labels) > 1:
            multi_sigma_count += 1
        for label in labels:
            sigma_counts[label] = sigma_counts.get(label, 0) + 1
    return PromoterCollectionSummary(
        record_count=len(record_list),
        unique_promoter_count=len(promoter_counts),
        duplicate_promoter_id_count=sum(count - 1 for count in promoter_counts.values() if count > 1),
        missing_sigma_count=missing_sigma_count,
        multi_sigma_count=multi_sigma_count,
        sigma_factor_counts=dict(sorted(sigma_counts.items())),
    )


def summarize_promoter_descriptors(descriptors: Iterable[PromoterDescriptor]) -> PromoterCollectionSummary:
    """Summarize promoter descriptors without requiring normalized sequences."""
    descriptor_list = list(descriptors)
    promoter_counts: dict[str, int] = {}
    sigma_counts: dict[str, int] = {}
    missing_sigma_count = 0
    multi_sigma_count = 0
    for descriptor in descriptor_list:
        promoter_counts[descriptor.promoter_id] = promoter_counts.get(descriptor.promoter_id, 0) + 1
        labels = tuple(dict.fromkeys(label for label in descriptor.sigma_factor_labels if _text_or_none(label)))
        if not labels:
            missing_sigma_count += 1
            continue
        if len(labels) > 1:
            multi_sigma_count += 1
        for label in labels:
            sigma_counts[label] = sigma_counts.get(label, 0) + 1
    return PromoterCollectionSummary(
        record_count=len(descriptor_list),
        unique_promoter_count=len(promoter_counts),
        duplicate_promoter_id_count=sum(count - 1 for count in promoter_counts.values() if count > 1),
        missing_sigma_count=missing_sigma_count,
        multi_sigma_count=multi_sigma_count,
        sigma_factor_counts=dict(sorted(sigma_counts.items())),
    )


def _source_inventory_blockers(
    inventory: PromoterSourceInventory,
    *,
    min_sequence_present_rate: float,
    min_promoter_id_present_rate: float,
) -> list[str]:
    blockers: list[str] = []
    if inventory.promoter_row_count <= 0:
        blockers.append("no promoter rows")
    if inventory.sequence_present_rate < min_sequence_present_rate:
        blockers.append(
            f"sequence_present_rate {inventory.sequence_present_rate:.3g} < {min_sequence_present_rate:.3g}"
        )
    if inventory.promoter_id_present_rate < min_promoter_id_present_rate:
        blockers.append(
            f"promoter_id_present_rate {inventory.promoter_id_present_rate:.3g} < {min_promoter_id_present_rate:.3g}"
        )
    if inventory.conflict_count:
        blockers.append(f"conflict_count {inventory.conflict_count} > 0")
    if inventory.route_failure_count:
        blockers.append(f"route_failure_count {inventory.route_failure_count} > 0")
    return blockers


def triage_promoter_sources(
    candidates: Mapping[str, PromoterSourceInventory],
    *,
    preferred_sources: tuple[str, ...] = (),
    min_sequence_present_rate: float = 1.0,
    min_promoter_id_present_rate: float = 1.0,
) -> PromoterSourceTriageReport:
    """Select one primary promoter source from inventory reports or block import."""
    if not candidates:
        return PromoterSourceTriageReport(
            primary_source=None,
            supplemental_sources=(),
            blocked=True,
            blocking_reasons=("no candidate promoter sources were inventoried",),
            candidate_status={},
        )

    candidate_status: dict[str, str] = {}
    eligible: set[str] = set()
    for source_id, inventory in sorted(candidates.items()):
        blockers = _source_inventory_blockers(
            inventory,
            min_sequence_present_rate=min_sequence_present_rate,
            min_promoter_id_present_rate=min_promoter_id_present_rate,
        )
        if blockers:
            candidate_status[source_id] = f"blocked: {', '.join(blockers)}"
            continue
        candidate_status[source_id] = "eligible"
        eligible.add(source_id)

    if not eligible:
        return PromoterSourceTriageReport(
            primary_source=None,
            supplemental_sources=tuple(sorted(candidates)),
            blocked=True,
            blocking_reasons=("no candidate source satisfied required promoter sequence/id/provenance gates",),
            candidate_status=candidate_status,
        )

    ordered_preferences = [source_id for source_id in preferred_sources if source_id in eligible]
    primary_source = ordered_preferences[0] if ordered_preferences else sorted(eligible)[0]
    supplemental_sources = tuple(sorted(source_id for source_id in candidates if source_id != primary_source))
    return PromoterSourceTriageReport(
        primary_source=primary_source,
        supplemental_sources=supplemental_sources,
        blocked=False,
        blocking_reasons=(),
        candidate_status=candidate_status,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _record_sort_key(record: PromoterRecord) -> tuple[str, str, str]:
    return (record.source_release, record.source_route, record.promoter_id)


def _record_to_alias_row(record: PromoterRecord) -> dict[str, Any]:
    return {
        "source": record.source,
        "source_release": record.source_release,
        "source_route": record.source_route,
        "promoter_id": record.promoter_id,
        "promoter_name": record.promoter_name,
        "raw_sequence_sha256": record.provenance.raw_payload_sha256,
        "strand": record.strand,
        "tss_raw": record.tss_position_raw,
        "first_gene": None if record.first_gene is None else record.first_gene.name,
        "tu_id": record.transcription_units[0].tu_id if record.transcription_units else None,
        "operon_id": None if record.operon is None else record.operon.operon_id,
        "confidence_level": record.confidence_level,
        "source_row_ref": record.provenance.raw_payload_ref or record.provenance.raw_payload_sha256,
    }


def _record_to_source_row(record: PromoterRecord) -> dict[str, Any]:
    return {
        "source": record.source,
        "source_release": record.source_release,
        "source_release_date": record.provenance.source_release_date,
        "source_route": record.source_route,
        "source_table": record.provenance.source_table,
        "source_stratum": record.provenance.source_stratum,
        "promoter_id": record.promoter_id,
        "promoter_name": record.promoter_name,
        "source_row_ref": record.provenance.raw_payload_ref,
        "raw_payload_sha256": record.provenance.raw_payload_sha256,
        "query_sha256": record.provenance.query_sha256,
        "normalized_record": promoter_record_to_dict(record),
    }


def _relation_rows(records: Iterable[PromoterRecord]) -> dict[str, list[dict[str, Any]]]:
    relations = {name: [] for name in _RELATION_NAMES}
    for record in sorted(records, key=_record_sort_key):
        relations["source_rows"].append(_record_to_source_row(record))
        relations["promoter_aliases"].append(_record_to_alias_row(record))
        for sigma in record.sigma_affiliations:
            row = asdict(sigma)
            row["promoter_id"] = record.promoter_id
            row["source_release"] = record.source_release
            row["source_route"] = sigma.source_route
            relations["sigma_affiliations"].append(row)
            for citation in sigma.citation_refs:
                relations["evidence_citations"].append(
                    {
                        "promoter_id": record.promoter_id,
                        "evidence": None,
                        "citation_ref": citation,
                        "source_release": record.source_release,
                        "source_route": sigma.source_route,
                    }
                )
        for box in record.boxes:
            row = asdict(box)
            row["promoter_id"] = record.promoter_id
            row["source_release"] = record.source_release
            relations["promoter_boxes"].append(row)
            if box.interval_0based is not None:
                relations["coordinate_features"].append(
                    {
                        "promoter_id": record.promoter_id,
                        "feature_kind": f"box:{box.kind}",
                        "interval_0based": box.interval_0based,
                        "strand": box.strand,
                        "source_release": record.source_release,
                        "source_route": box.source_route,
                    }
                )
        if record.tss_interval_0based is not None:
            relations["coordinate_features"].append(
                {
                    "promoter_id": record.promoter_id,
                    "feature_kind": "tss",
                    "interval_0based": record.tss_interval_0based,
                    "strand": record.strand,
                    "source_release": record.source_release,
                    "source_route": record.source_route,
                }
            )
        for site in record.regulatory_sites:
            row = asdict(site)
            row["promoter_id"] = record.promoter_id
            row["source_release"] = record.source_release
            row["source_route"] = record.source_route
            relations["regulatory_interactions"].append(row)
            if site.binding_site_id is not None:
                relations["tfbs_sites"].append(row)
            if site.interval_0based is not None:
                relations["coordinate_features"].append(
                    {
                        "promoter_id": record.promoter_id,
                        "feature_kind": "tfbs",
                        "interval_0based": site.interval_0based,
                        "strand": site.strand,
                        "source_release": record.source_release,
                        "source_route": record.source_route,
                    }
                )
            for citation in site.citation_refs:
                relations["evidence_citations"].append(
                    {
                        "promoter_id": record.promoter_id,
                        "evidence": ",".join(site.evidence) if site.evidence else None,
                        "citation_ref": citation,
                        "source_release": record.source_release,
                        "source_route": record.source_route,
                    }
                )
    return relations


def export_promoter_records(
    records: Iterable[PromoterRecord],
    destination: Path,
    *,
    query: PromoterQuery,
    inventory: PromoterSourceInventory | None = None,
    source_selection_status: str = "unselected",
    skipped_source_rows: Iterable[SkippedPromoterSourceRow] = (),
) -> PromoterExportManifest:
    """Write a deterministic promoter export/cache directory."""
    record_list = sorted(list(records), key=_record_sort_key)
    skipped_list = sorted(
        list(skipped_source_rows),
        key=lambda row: (row.source_release, row.source_route, row.source_row_ref),
    )
    source_inventory = inventory or build_promoter_source_inventory(record_list)
    destination.mkdir(parents=True, exist_ok=True)
    relations_dir = destination / "relations"
    relations_dir.mkdir(parents=True, exist_ok=True)

    manifest = PromoterExportManifest(
        schema_version=PROMOTER_EXPORT_SCHEMA_VERSION,
        parser_version=PROMOTER_PARSER_VERSION,
        export_created_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
        complete=source_inventory.route_failure_count == 0 and source_inventory.conflict_count == 0,
        record_count=len(record_list),
        source_selection_status=source_selection_status,
        source_inventory=source_inventory,
        query=query,
        artifacts={
            "promoters": "promoters.jsonl",
            "relations": "relations/",
            "raw_payload_refs": "raw_payload_refs.jsonl",
            "skipped_source_rows": "skipped_source_rows.jsonl",
        },
    )
    _write_json(destination / "manifest.json", manifest)
    _write_jsonl(destination / "promoters.jsonl", (promoter_record_to_dict(record) for record in record_list))
    _write_jsonl(destination / "skipped_source_rows.jsonl", (skipped_source_row_to_dict(row) for row in skipped_list))
    relations = _relation_rows(record_list)
    for name in _RELATION_NAMES:
        _write_jsonl(relations_dir / f"{name}.jsonl", relations[name])
    _write_jsonl(
        destination / "raw_payload_refs.jsonl",
        (
            {
                "promoter_id": record.promoter_id,
                "source_release": record.source_release,
                "source_route": record.source_route,
                "raw_payload_sha256": record.provenance.raw_payload_sha256,
                "raw_payload_ref": record.provenance.raw_payload_ref,
                "query_sha256": record.provenance.query_sha256,
            }
            for record in record_list
        ),
    )
    return manifest


def promoter_record_to_dict(record: PromoterRecord) -> dict[str, Any]:
    return _jsonable(record)


def skipped_source_row_to_dict(row: SkippedPromoterSourceRow) -> dict[str, Any]:
    return _jsonable(row)


def promoter_regulatory_association_to_dict(row: PromoterRegulatoryAssociation) -> dict[str, Any]:
    return _jsonable(row)


def skipped_source_row_from_dict(data: Mapping[str, Any]) -> SkippedPromoterSourceRow:
    return SkippedPromoterSourceRow(
        source=str(data["source"]),
        source_release=str(data["source_release"]),
        source_release_date=_text_or_none(data.get("source_release_date")),
        source_route=str(data["source_route"]),
        source_table=_text_or_none(data.get("source_table")),
        source_stratum=str(data["source_stratum"]),
        promoter_id=_text_or_none(data.get("promoter_id")),
        promoter_name=_text_or_none(data.get("promoter_name")),
        raw_sequence=_text_or_none(data.get("raw_sequence")),
        skip_reason=str(data["skip_reason"]),
        source_row_ref=str(data["source_row_ref"]),
        raw_payload_sha256=str(data["raw_payload_sha256"]),
        query_sha256=str(data["query_sha256"]),
        parser_version=str(data.get("parser_version") or PROMOTER_PARSER_VERSION),
        export_schema_version=str(data.get("export_schema_version") or PROMOTER_EXPORT_SCHEMA_VERSION),
    )


def promoter_regulatory_association_from_dict(data: Mapping[str, Any]) -> PromoterRegulatoryAssociation:
    return PromoterRegulatoryAssociation(
        source=str(data["source"]),
        source_release=str(data["source_release"]),
        source_release_date=_text_or_none(data.get("source_release_date")),
        source_route=str(data["source_route"]),
        source_table=_text_or_none(data.get("source_table")),
        source_stratum=str(data["source_stratum"]),
        source_row_ref=str(data["source_row_ref"]),
        regulatory_interaction_id=str(data["regulatory_interaction_id"]),
        promoter_id=_text_or_none(data.get("promoter_id")),
        promoter_name=_text_or_none(data.get("promoter_name")),
        regulated_entity_name=_text_or_none(data.get("regulated_entity_name")),
        regulator_id=_text_or_none(data.get("regulator_id")),
        regulator_name=_text_or_none(data.get("regulator_name")),
        regulator_abbrev=_text_or_none(data.get("regulator_abbrev")),
        regulon_id=_text_or_none(data.get("regulon_id")),
        regulon_name=_text_or_none(data.get("regulon_name")),
        target_type=_text_or_none(data.get("target_type")),
        function=_text_or_none(data.get("function")),
        mechanism=_text_or_none(data.get("mechanism")),
        confidence=_text_or_none(data.get("confidence")),
        evidence=_tuple_str(data.get("evidence")),
        citation_refs=_tuple_str(data.get("citation_refs")),
        binding_site_id=_text_or_none(data.get("binding_site_id")),
        binding_site_sequence=_text_or_none(data.get("binding_site_sequence")),
        binding_site_strand=_text_or_none(data.get("binding_site_strand")),
        binding_interval_0based=_tuple_interval(data.get("binding_interval_0based")),
        binding_raw_coordinates=dict(data.get("binding_raw_coordinates") or {}),
        raw_payload_sha256=_text_or_none(data.get("raw_payload_sha256")),
        query_sha256=_text_or_none(data.get("query_sha256")),
        parser_version=str(data.get("parser_version") or PROMOTER_PARSER_VERSION),
        export_schema_version=str(data.get("export_schema_version") or PROMOTER_EXPORT_SCHEMA_VERSION),
    )


def _datetime_from_payload(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _tuple_interval(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise PromoterSchemaError(f"Expected 2-element interval, got {value!r}.")
    return (int(value[0]), int(value[1]))


def _tuple_str(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in _list_payload(value))


def promoter_record_from_dict(data: Mapping[str, Any]) -> PromoterRecord:
    provenance_payload = dict(data["provenance"])
    provenance = SourceProvenance(
        source=str(provenance_payload["source"]),
        source_release=str(provenance_payload["source_release"]),
        source_route=str(provenance_payload["source_route"]),
        fetched_at=_datetime_from_payload(str(provenance_payload["fetched_at"])),
        raw_payload_sha256=str(provenance_payload["raw_payload_sha256"]),
        query_sha256=str(provenance_payload["query_sha256"]),
        parser_version=str(provenance_payload.get("parser_version") or PROMOTER_PARSER_VERSION),
        export_schema_version=str(provenance_payload.get("export_schema_version") or PROMOTER_EXPORT_SCHEMA_VERSION),
        source_release_date=_text_or_none(provenance_payload.get("source_release_date")),
        source_url=_text_or_none(provenance_payload.get("source_url")),
        raw_payload_ref=_text_or_none(provenance_payload.get("raw_payload_ref")),
        source_table=_text_or_none(provenance_payload.get("source_table")),
        source_stratum=str(provenance_payload.get("source_stratum") or "curated"),
    )
    return PromoterRecord(
        source=str(data["source"]),
        source_release=str(data["source_release"]),
        source_route=str(data["source_route"]),
        promoter_id=str(data["promoter_id"]),
        promoter_name=_text_or_none(data.get("promoter_name")),
        sequence=str(data["sequence"]),
        raw_sequence=str(data["raw_sequence"]),
        sequence_case_policy=str(data["sequence_case_policy"]),
        sequence_length=int(data["sequence_length"]),
        strand=_text_or_none(data.get("strand")),
        genome_accession=_text_or_none(data.get("genome_accession")),
        tss_position_raw=_text_or_none(data.get("tss_position_raw")),
        tss_interval_0based=_tuple_interval(data.get("tss_interval_0based")),
        confidence_level=_text_or_none(data.get("confidence_level")),
        score=_float_or_none(data.get("score")),
        evidence=_tuple_str(data.get("evidence")),
        citations=_tuple_str(data.get("citations")),
        sigma_affiliations=tuple(
            PromoterSigmaAffiliation(
                sigma_id=_text_or_none(item.get("sigma_id")),
                name=_text_or_none(item.get("name")),
                abbrev=_text_or_none(item.get("abbrev")),
                gene_id=_text_or_none(item.get("gene_id")),
                gene_name=_text_or_none(item.get("gene_name")),
                source_route=str(item["source_route"]),
                evidence=_tuple_str(item.get("evidence")),
                confidence=_text_or_none(item.get("confidence")),
                citation_refs=_tuple_str(item.get("citation_refs")),
            )
            for item in _list_payload(data.get("sigma_affiliations"))
        ),
        boxes=tuple(
            PromoterBox(
                kind=str(item["kind"]),
                sequence=_text_or_none(item.get("sequence")),
                raw_coordinates=dict(item.get("raw_coordinates") or {}),
                interval_0based=_tuple_interval(item.get("interval_0based")),
                strand=_text_or_none(item.get("strand")),
                source_route=str(item["source_route"]),
            )
            for item in _list_payload(data.get("boxes"))
        ),
        regulatory_sites=tuple(
            PromoterRegulatorySite(
                regulatory_interaction_id=_text_or_none(item.get("regulatory_interaction_id")),
                binding_site_id=_text_or_none(item.get("binding_site_id")),
                regulator_id=_text_or_none(item.get("regulator_id")),
                regulator_name=_text_or_none(item.get("regulator_name")),
                regulator_abbrev=_text_or_none(item.get("regulator_abbrev")),
                regulon_id=_text_or_none(item.get("regulon_id")),
                regulon_name=_text_or_none(item.get("regulon_name")),
                target_type=_text_or_none(item.get("target_type")),
                function=_text_or_none(item.get("function")),
                mechanism=_text_or_none(item.get("mechanism")),
                raw_coordinates=dict(item.get("raw_coordinates") or {}),
                interval_0based=_tuple_interval(item.get("interval_0based")),
                strand=_text_or_none(item.get("strand")),
                sequence=_text_or_none(item.get("sequence")),
                confidence=_text_or_none(item.get("confidence")),
                evidence=_tuple_str(item.get("evidence")),
                citation_refs=_tuple_str(item.get("citation_refs")),
            )
            for item in _list_payload(data.get("regulatory_sites"))
        ),
        transcription_units=tuple(
            TranscriptionUnitRef(tu_id=_text_or_none(item.get("tu_id")), name=_text_or_none(item.get("name")))
            for item in _list_payload(data.get("transcription_units"))
        ),
        operon=(
            None
            if data.get("operon") is None
            else OperonRef(
                operon_id=_text_or_none(data["operon"].get("operon_id")),
                name=_text_or_none(data["operon"].get("name")),
            )
        ),
        first_gene=(
            None
            if data.get("first_gene") is None
            else GeneRef(
                gene_id=_text_or_none(data["first_gene"].get("gene_id")),
                name=_text_or_none(data["first_gene"].get("name")),
            )
        ),
        provenance=provenance,
    )


def _inventory_from_dict(data: Mapping[str, Any]) -> PromoterSourceInventory:
    return PromoterSourceInventory(
        source_releases=tuple(data.get("source_releases") or ()),
        source_routes=tuple(data.get("source_routes") or ()),
        promoter_row_count=int(data["promoter_row_count"]),
        sequence_present_rate=float(data["sequence_present_rate"]),
        promoter_id_present_rate=float(data["promoter_id_present_rate"]),
        tss_present_rate=float(data["tss_present_rate"]),
        sigma_present_rate=float(data["sigma_present_rate"]),
        box_annotation_rate=float(data["box_annotation_rate"]),
        confidence_present_rate=float(data["confidence_present_rate"]),
        regulatory_context_rate=float(data["regulatory_context_rate"]),
        duplicate_sequence_count=int(data["duplicate_sequence_count"]),
        conflict_count=int(data["conflict_count"]),
        route_failure_count=int(data.get("route_failure_count") or 0),
    )


def _query_from_dict(data: Mapping[str, Any]) -> PromoterQuery:
    return PromoterQuery(
        source_release_policy=str(data.get("source_release_policy") or "reported"),
        source_release=_text_or_none(data.get("source_release")),
        routes=tuple(data.get("routes") or ()),
        limit=None if data.get("limit") is None else int(data["limit"]),
        page_size=int(data.get("page_size") or 100),
        include_relations=bool(data.get("include_relations", True)),
        timeout_seconds=int(data.get("timeout_seconds") or 30),
        source_stratum=str(data.get("source_stratum") or "curated"),
    )


def _manifest_from_dict(data: Mapping[str, Any]) -> PromoterExportManifest:
    return PromoterExportManifest(
        schema_version=str(data["schema_version"]),
        parser_version=str(data["parser_version"]),
        export_created_at=_datetime_from_payload(str(data["export_created_at"])),
        complete=bool(data["complete"]),
        record_count=int(data["record_count"]),
        source_selection_status=str(data["source_selection_status"]),
        source_inventory=_inventory_from_dict(data["source_inventory"]),
        query=_query_from_dict(data["query"]),
        artifacts=dict(data.get("artifacts") or {}),
    )


def load_promoter_export(export_dir: Path) -> tuple[PromoterExportManifest, list[PromoterRecord]]:
    manifest_path = export_dir / "manifest.json"
    promoters_path = export_dir / "promoters.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Promoter export manifest not found: {manifest_path}")
    if not promoters_path.exists():
        raise FileNotFoundError(f"Promoter export records not found: {promoters_path}")
    manifest = _manifest_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    records = [
        promoter_record_from_dict(json.loads(line))
        for line in promoters_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if manifest.record_count != len(records):
        raise PromoterSchemaError(
            f"Promoter export manifest record_count={manifest.record_count} "
            f"but promoters.jsonl has {len(records)} rows."
        )
    return manifest, records


def load_skipped_promoter_source_rows(export_dir: Path) -> list[SkippedPromoterSourceRow]:
    manifest_path = export_dir / "manifest.json"
    artifact_path = export_dir / "skipped_source_rows.jsonl"
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_name = (manifest_payload.get("artifacts") or {}).get("skipped_source_rows")
        if artifact_name:
            artifact_path = export_dir / str(artifact_name)
    if not artifact_path.exists():
        return []
    return [
        skipped_source_row_from_dict(json.loads(line))
        for line in artifact_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_promoter_regulatory_associations(export_dir: Path) -> list[PromoterRegulatoryAssociation]:
    manifest_path = export_dir / "manifest.json"
    artifact_path = export_dir / "promoter_regulatory_associations.jsonl"
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_name = (manifest_payload.get("artifacts") or {}).get("promoter_regulatory_associations")
        if artifact_name:
            artifact_path = export_dir / str(artifact_name)
    if not artifact_path.exists():
        return []
    return [
        promoter_regulatory_association_from_dict(json.loads(line))
        for line in artifact_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
