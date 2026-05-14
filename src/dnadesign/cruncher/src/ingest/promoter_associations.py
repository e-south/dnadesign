"""Promoter regulatory-association ingest helpers for Cruncher exports."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .promoter_contracts import (
    PromoterAssociationSourceFile,
    PromoterRegulatoryAssociation,
    PromoterSchemaError,
)
from .promoter_payloads import _normalize_strand, _sha256_json, _text_or_none
from .promoter_tables import _iter_tsv_data_rows, _split_table_list, _table_value

_NETWORK_EFFECT_TO_FUNCTION = {
    "+": "activator",
    "-": "repressor",
    "+-": "dual",
    "-+": "dual",
    "?": "unknown",
}
_NETWORK_PROMOTER_TOKEN_RE = re.compile(r"\[([^\]]+)\]")


def _network_effect_function(value: str | None) -> str:
    text = str(value or "").strip()
    return _NETWORK_EFFECT_TO_FUNCTION.get(text, text or "unknown")


def _network_evidence(value: str | None) -> tuple[str, ...]:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _network_promoter_token(value: str | None) -> str | None:
    match = _NETWORK_PROMOTER_TOKEN_RE.search(str(value or ""))
    if match is None:
        return None
    return _text_or_none(match.group(1))


def _dedupe_preserve_order(values: Iterable[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return tuple(out)


def _strict_dna_or_none(value: str | None) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    normalized = text.upper().replace("U", "T")
    if set(normalized) - set("ACGT"):
        raise PromoterSchemaError(f"RegulonDB binding-site sequence must be strict A/C/G/T, got {value!r}.")
    return normalized


def _one_based_unordered_inclusive_interval(left: object, right: object) -> tuple[int, int] | None:
    if left is None or right is None or left == "" or right == "":
        return None
    left_pos = int(left)
    right_pos = int(right)
    if left_pos <= 0 or right_pos <= 0:
        raise PromoterSchemaError(f"RegulonDB 1-based interval positions must be positive: {left!r}, {right!r}.")
    start = min(left_pos, right_pos) - 1
    end = max(left_pos, right_pos)
    if end <= start:
        raise PromoterSchemaError(f"Invalid RegulonDB 1-based inclusive interval: {left!r}, {right!r}.")
    return (start, end)


def parse_regulondb_network_tf_tu_associations(
    path: Path,
    *,
    source_release: str,
    source_route: str = "regulondb_network_tf_tu",
    source_stratum: str = "historical_curated_network_association",
    fetched_at: datetime | None = None,
    source_release_date: str | None = None,
) -> list[PromoterRegulatoryAssociation]:
    """Parse a release-pinned RegulonDB TF-to-TU network table into promoter links."""

    associations: list[PromoterRegulatoryAssociation] = []
    query_base = {
        "source_table": path.name,
        "source_release": source_release,
        "source_route": source_route,
    }
    _ = fetched_at
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = next(csv.reader([line], delimiter="\t"))
            if len(values) < 5:
                raise PromoterSchemaError(
                    f"RegulonDB network_tf_tu row at {path}:{line_number} has {len(values)} columns; expected 5."
                )
            regulator, regulated_entity, effect, evidence, confidence = values[:5]
            row_payload = {
                "regulator": regulator,
                "regulated_entity": regulated_entity,
                "effect": effect,
                "evidence": evidence,
                "confidence": confidence,
            }
            query = {**query_base, "line_number": line_number}
            regulator_abbrev = _text_or_none(regulator)
            associations.append(
                PromoterRegulatoryAssociation(
                    source="regulondb",
                    source_release=str(source_release),
                    source_release_date=source_release_date,
                    source_route=source_route,
                    source_table=path.name,
                    source_stratum=source_stratum,
                    source_row_ref=f"{path}:{line_number}",
                    regulatory_interaction_id=f"{source_route}:{line_number}",
                    promoter_id=None,
                    promoter_name=_network_promoter_token(regulated_entity),
                    regulated_entity_name=_text_or_none(regulated_entity),
                    regulator_id=None,
                    regulator_name=regulator_abbrev,
                    regulator_abbrev=regulator_abbrev,
                    regulon_id=None,
                    regulon_name=None if regulator_abbrev is None else f"{regulator_abbrev} regulon",
                    target_type="transcription_unit",
                    function=_network_effect_function(effect),
                    mechanism="network_tf_tu",
                    confidence=_text_or_none(confidence),
                    evidence=_network_evidence(evidence),
                    citation_refs=(),
                    raw_payload_sha256=_sha256_json(row_payload),
                    query_sha256=_sha256_json(query),
                )
            )
    return associations


def parse_regulondb_tf_riset_associations(
    path: Path,
    *,
    source_release: str,
    source_route: str = "regulondb_13_tf_riset",
    source_stratum: str = "current_curated_regulatory_interaction",
    fetched_at: datetime | None = None,
    source_release_date: str | None = None,
) -> list[PromoterRegulatoryAssociation]:
    """Parse RegulonDB TF-RISet rows into direct promoter regulatory associations."""

    associations: list[PromoterRegulatoryAssociation] = []
    query_base = {
        "source_table": path.name,
        "source_release": source_release,
        "source_route": source_route,
    }
    _ = fetched_at
    for line_number, row in _iter_tsv_data_rows(path):
        regulatory_interaction_id = _table_value(row, "ri_id", "riid")
        if regulatory_interaction_id is None:
            raise PromoterSchemaError(f"RegulonDB TF-RISet row at {path}:{line_number} is missing riId.")
        row_payload = dict(row)
        query = {**query_base, "line_number": line_number}
        evidence = _dedupe_preserve_order(
            [
                *_split_table_list(_table_value(row, "tfrs_evidence", "tfrsevidence")),
                *_split_table_list(_table_value(row, "ri_evidence", "rievidence")),
                *_split_table_list(_table_value(row, "add_evidence", "addevidence")),
            ]
        )
        citations = _dedupe_preserve_order(
            [
                *_split_table_list(_table_value(row, "tfrs_pmids", "tfrspmids")),
                *_split_table_list(_table_value(row, "ri_pmids", "ripmids")),
            ]
        )
        left = _table_value(row, "tfrs_left", "tfrsleft")
        right = _table_value(row, "tfrs_right", "tfrsright")
        interval = _one_based_unordered_inclusive_interval(left, right)
        strand = _normalize_strand(_table_value(row, "strand"))
        regulator_name = _table_value(row, "regulator_name", "regulatorname")
        associations.append(
            PromoterRegulatoryAssociation(
                source="regulondb",
                source_release=str(source_release),
                source_release_date=source_release_date,
                source_route=source_route,
                source_table=path.name,
                source_stratum=source_stratum,
                source_row_ref=f"{path}:{line_number}",
                regulatory_interaction_id=regulatory_interaction_id,
                promoter_id=_table_value(row, "promoter_id", "promoterid"),
                promoter_name=_table_value(row, "promoter_name", "promotername"),
                regulated_entity_name=_table_value(row, "target_tu_or_gene", "targettuorgene"),
                regulator_id=_table_value(row, "regulator_id", "regulatorid"),
                regulator_name=regulator_name,
                regulator_abbrev=regulator_name,
                regulon_id=None,
                regulon_name=None if regulator_name is None else f"{regulator_name} regulon",
                target_type=_table_value(row, "ri_type", "ritype"),
                function=_table_value(row, "ri_function", "rifunction"),
                mechanism="tf_riset",
                confidence=_table_value(row, "confidence_level", "confidencelevel"),
                evidence=evidence,
                citation_refs=citations,
                binding_site_id=_table_value(row, "tfrs_id", "tfrsid"),
                binding_site_sequence=_strict_dna_or_none(_table_value(row, "tfrs_seq", "tfrsseq")),
                binding_site_strand=strand,
                binding_interval_0based=interval,
                binding_raw_coordinates={"left": left, "right": right, "strand": _table_value(row, "strand")},
                raw_payload_sha256=_sha256_json(row_payload),
                query_sha256=_sha256_json(query),
            )
        )
    return associations


def discover_dnadesign_data_promoter_association_sources(
    *,
    data_root: Path | None = None,
    provider: Any | None = None,
    required: bool = False,
) -> tuple[PromoterAssociationSourceFile, ...]:
    """Return promoter association source files from dnadesign-data when available."""

    if provider is None:
        try:
            from dnadesign_data.catalog.regulatory_parts import iter_promoter_association_source_files
        except (ImportError, ModuleNotFoundError) as exc:
            if required:
                raise PromoterSchemaError(
                    "dnadesign_data is required to discover promoter association sources."
                ) from exc
            return ()
        provider = iter_promoter_association_source_files
    discovered = []
    for item in provider(data_root):
        if isinstance(item, PromoterAssociationSourceFile):
            discovered.append(item)
            continue
        if isinstance(item, Mapping):
            discovered.append(PromoterAssociationSourceFile(**dict(item)))
            continue
        discovered.append(PromoterAssociationSourceFile(**asdict(item)))
    if required and not discovered:
        raise PromoterSchemaError("No required promoter association sources were discovered from dnadesign-data.")
    return tuple(discovered)


def parse_promoter_association_source_file(
    source: PromoterAssociationSourceFile,
    *,
    data_root: Path,
    fetched_at: datetime | None = None,
) -> list[PromoterRegulatoryAssociation]:
    """Parse source descriptors emitted by dnadesign-data into association records."""

    path = data_root / source.path
    if source.parser_hint == "regulondb_tf_riset" and source.file_format == "tsv":
        return parse_regulondb_tf_riset_associations(
            path,
            source_release=source.release,
            source_route=source.source_id,
            source_stratum=source.stratum,
            fetched_at=fetched_at,
        )
    if source.parser_hint == "regulondb_network_tf_tu" and source.file_format == "tsv":
        return parse_regulondb_network_tf_tu_associations(
            path,
            source_release=source.release,
            source_route=source.source_id,
            source_stratum=source.stratum,
            fetched_at=fetched_at,
        )
    raise PromoterSchemaError(
        f"Unsupported promoter association source parser {source.parser_hint!r} for {source.source_id!r}."
    )
