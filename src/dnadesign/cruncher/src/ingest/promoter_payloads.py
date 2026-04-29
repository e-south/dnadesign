"""RegulonDB promoter payload normalization helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from .promoter_contracts import (
    _STRICT_DNA_RE,
    GeneRef,
    OperonRef,
    PromoterBox,
    PromoterRecord,
    PromoterRegulatorySite,
    PromoterSchemaError,
    PromoterSigmaAffiliation,
    SourceProvenance,
    TranscriptionUnitRef,
)


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _required_text(mapping: Mapping[str, Any], *keys: str, label: str) -> str:
    value = _first_present(mapping, *keys)
    text = str(value or "").strip()
    if not text:
        raise PromoterSchemaError(f"RegulonDB promoter payload is missing required {label}.")
    return text


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sequence_or_raise(value: Any) -> tuple[str, str]:
    raw = _text_or_none(value)
    if raw is None:
        raise PromoterSchemaError("RegulonDB promoter payload is missing required sequence.")
    canonical = raw.upper().replace("U", "T")
    if not _STRICT_DNA_RE.match(canonical):
        raise PromoterSchemaError("RegulonDB promoter sequence must be strict A/C/G/T DNA after U-to-T conversion.")
    return canonical, raw


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PromoterSchemaError(f"RegulonDB promoter score must be numeric, got {value!r}.") from exc


def _normalize_strand(value: Any) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    norm = text.casefold()
    if norm in {"+", "plus", "forward", "1"}:
        return "+"
    if norm in {"-", "minus", "reverse", "-1"}:
        return "-"
    raise PromoterSchemaError(f"Unrecognized RegulonDB strand value: {value!r}.")


def _one_based_single_interval(value: Any) -> tuple[int, int] | None:
    if value is None or value == "":
        return None
    pos = int(value)
    if pos <= 0:
        raise PromoterSchemaError(f"RegulonDB 1-based position must be positive, got {value!r}.")
    return (pos - 1, pos)


def _one_based_inclusive_interval(left: Any, right: Any) -> tuple[int, int] | None:
    if left is None or right is None or left == "" or right == "":
        return None
    start = int(left) - 1
    end = int(right)
    if start < 0 or end <= start:
        raise PromoterSchemaError(f"Invalid RegulonDB 1-based inclusive interval: {left!r}, {right!r}.")
    return (start, end)


def _list_payload(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _mapping_payload(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _collect_citation_refs(value: Any) -> tuple[str, ...]:
    refs: list[str] = []
    for item in _list_payload(value):
        if isinstance(item, Mapping):
            publication = _mapping_payload(item.get("publication"))
            ref = _first_present(
                item,
                "pmid",
                "PMID",
                "citationId",
                "_id",
                "id",
            )
            if ref is None and publication:
                ref = _first_present(publication, "pmid", "PMID", "_id", "id", "citation")
        else:
            ref = item
        text = _text_or_none(ref)
        if text and text not in refs:
            refs.append(text)
    return tuple(refs)


def _collect_evidence(value: Any) -> tuple[str, ...]:
    evidence: list[str] = []
    for item in _list_payload(value):
        if isinstance(item, Mapping):
            evidence_payload = _mapping_payload(item.get("evidence"))
            raw = _first_present(item, "code", "name", "type", "text", "_id", "id")
            if raw is None and evidence_payload:
                raw = _first_present(evidence_payload, "code", "name", "type", "text", "_id", "id")
        else:
            raw = item
        text = _text_or_none(raw)
        if text and text not in evidence:
            evidence.append(text)
    return tuple(evidence)


def _normalize_box_kind(value: Any) -> str:
    text = _text_or_none(value) or "unknown"
    norm = text.strip().casefold().replace(" ", "_").replace("-", "minus_")
    norm = norm.replace("__", "_")
    if norm in {"10", "minus_10", "minus10"}:
        return "minus_10"
    if norm in {"35", "minus_35", "minus35"}:
        return "minus_35"
    return norm


def _parse_boxes(payload: Mapping[str, Any], *, source_route: str) -> tuple[PromoterBox, ...]:
    boxes: list[PromoterBox] = []
    for raw_box in _list_payload(_first_present(payload, "boxes", "promoterBoxes", "boxAnnotations")):
        box = _mapping_payload(raw_box)
        if not box:
            continue
        left = _first_present(box, "leftEndPosition", "left", "start", "begin")
        right = _first_present(box, "rightEndPosition", "right", "end", "stop")
        boxes.append(
            PromoterBox(
                kind=_normalize_box_kind(_first_present(box, "type", "kind", "box", "name")),
                sequence=_text_or_none(_first_present(box, "sequence", "seq")),
                raw_coordinates={"left": left, "right": right, "origin": "regulondb_1based_inclusive"},
                interval_0based=_one_based_inclusive_interval(left, right),
                strand=_normalize_strand(_first_present(box, "strand")),
                source_route=source_route,
            )
        )
    return tuple(boxes)


def _parse_sigma_affiliations(payload: Mapping[str, Any], *, source_route: str) -> tuple[PromoterSigmaAffiliation, ...]:
    affiliations: list[PromoterSigmaAffiliation] = []
    for raw_sigma in _list_payload(_first_present(payload, "sigmaFactors", "sigma_affiliations", "sigmas")):
        sigma = _mapping_payload(raw_sigma)
        if not sigma:
            continue
        gene = _mapping_payload(_first_present(sigma, "gene", "sigmaGene"))
        has_identity = any(
            _text_or_none(value)
            for value in (
                _first_present(sigma, "_id", "id", "sigma_id"),
                _first_present(sigma, "name", "sigmaName"),
                _first_present(sigma, "abbreviatedName", "abbrev", "sigmaAbbrev"),
                _first_present(gene, "_id", "id", "gene_id"),
                _first_present(gene, "name", "geneName"),
            )
        )
        if not has_identity:
            continue
        affiliations.append(
            PromoterSigmaAffiliation(
                sigma_id=_text_or_none(_first_present(sigma, "_id", "id", "sigma_id")),
                name=_text_or_none(_first_present(sigma, "name", "sigmaName")),
                abbrev=_text_or_none(_first_present(sigma, "abbreviatedName", "abbrev", "sigmaAbbrev")),
                gene_id=_text_or_none(_first_present(gene, "_id", "id", "gene_id")),
                gene_name=_text_or_none(_first_present(gene, "name", "geneName")),
                source_route=source_route,
                evidence=_collect_evidence(_first_present(sigma, "evidence", "evidences")),
                confidence=_text_or_none(_first_present(sigma, "confidence", "confidenceLevel")),
                citation_refs=_collect_citation_refs(_first_present(sigma, "citations", "references")),
            )
        )
    return tuple(affiliations)


def _parse_regulatory_sites(payload: Mapping[str, Any]) -> tuple[PromoterRegulatorySite, ...]:
    sites: list[PromoterRegulatorySite] = []
    for raw_interaction in _list_payload(
        _first_present(payload, "regulatoryInteractions", "regulatory_interactions", "regulators")
    ):
        interaction = _mapping_payload(raw_interaction)
        if not interaction:
            continue
        regulator = _mapping_payload(_first_present(interaction, "regulator", "transcriptionFactor"))
        regulon = _mapping_payload(_first_present(interaction, "regulon"))
        site = _mapping_payload(_first_present(interaction, "regulatoryBindingSites", "bindingSite", "tfbs"))
        left = _first_present(site, "leftEndPosition", "chrLeftPosition", "left", "start", "begin")
        right = _first_present(site, "rightEndPosition", "chrRightPosition", "right", "end", "stop")
        sites.append(
            PromoterRegulatorySite(
                regulatory_interaction_id=_text_or_none(_first_present(interaction, "_id", "id", "ri_id")),
                binding_site_id=_text_or_none(_first_present(site, "_id", "id", "site_id")),
                regulator_id=_text_or_none(_first_present(regulator, "_id", "id", "regulator_id")),
                regulator_name=_text_or_none(_first_present(regulator, "name")),
                regulator_abbrev=_text_or_none(_first_present(regulator, "abbreviatedName", "abbrev")),
                regulon_id=_text_or_none(_first_present(regulon, "_id", "id", "regulon_id")),
                regulon_name=_text_or_none(_first_present(regulon, "name")),
                target_type=_text_or_none(_first_present(interaction, "targetType", "target_type")),
                function=_text_or_none(_first_present(interaction, "function", "regulatoryFunction")),
                mechanism=_text_or_none(_first_present(interaction, "mechanism")),
                raw_coordinates={"left": left, "right": right, "origin": "regulondb_1based_inclusive"},
                interval_0based=_one_based_inclusive_interval(left, right),
                strand=_normalize_strand(_first_present(site, "strand")),
                sequence=_text_or_none(_first_present(site, "sequence", "seq")),
                confidence=_text_or_none(_first_present(interaction, "confidence", "confidenceLevel")),
                evidence=_collect_evidence(_first_present(interaction, "evidence", "evidences")),
                citation_refs=_collect_citation_refs(_first_present(interaction, "citations", "references")),
            )
        )
    return tuple(sites)


def _parse_tus(payload: Mapping[str, Any]) -> tuple[TranscriptionUnitRef, ...]:
    refs: list[TranscriptionUnitRef] = []
    for raw_tu in _list_payload(_first_present(payload, "transcriptionUnits", "transcription_units", "tus")):
        tu = _mapping_payload(raw_tu)
        if not tu:
            continue
        refs.append(
            TranscriptionUnitRef(
                tu_id=_text_or_none(_first_present(tu, "_id", "id", "tu_id")),
                name=_text_or_none(_first_present(tu, "name")),
            )
        )
    return tuple(refs)


def _parse_operon(payload: Mapping[str, Any]) -> OperonRef | None:
    operon = _mapping_payload(_first_present(payload, "operon"))
    if not operon:
        return None
    return OperonRef(
        operon_id=_text_or_none(_first_present(operon, "_id", "id", "operon_id")),
        name=_text_or_none(_first_present(operon, "name")),
    )


def _parse_gene(payload: Mapping[str, Any]) -> GeneRef | None:
    gene = _mapping_payload(_first_present(payload, "firstGene", "first_gene", "gene"))
    if not gene:
        return None
    return GeneRef(
        gene_id=_text_or_none(_first_present(gene, "_id", "id", "gene_id")),
        name=_text_or_none(_first_present(gene, "name")),
    )


def _parse_tss(payload: Mapping[str, Any]) -> tuple[str | None, tuple[int, int] | None]:
    raw_tss = _first_present(payload, "posTSS", "tssPosition", "tss_position", "tss")
    if raw_tss is not None and raw_tss != "":
        return _text_or_none(raw_tss), _one_based_single_interval(raw_tss)
    left = _first_present(payload, "tssLeftEndPosition", "tss_left", "tss_start")
    right = _first_present(payload, "tssRightEndPosition", "tss_right", "tss_end")
    interval = _one_based_inclusive_interval(left, right)
    if interval is None:
        return None, None
    return f"{left}..{right}", interval


def parse_regulondb_promoter_payload(
    payload: Mapping[str, Any],
    *,
    source_release: str,
    source_route: str,
    query: Mapping[str, Any],
    fetched_at: datetime | None = None,
    source_release_date: str | None = None,
    source_url: str | None = None,
    source_table: str | None = None,
    raw_payload_ref: str | None = None,
    source_stratum: str = "curated",
) -> PromoterRecord:
    """Normalize one promoter-shaped RegulonDB payload."""
    promoter_id = _required_text(payload, "_id", "id", "promoter_id", "promoterId", label="promoter id")
    sequence, raw_sequence = _sequence_or_raise(_first_present(payload, "sequence", "promoterSequence", "seq"))
    fetched = fetched_at or datetime.now(timezone.utc)
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    raw_tss, tss_interval = _parse_tss(payload)
    raw_checksum = _sha256_json(payload)
    query_checksum = _sha256_json(query)
    provenance = SourceProvenance(
        source="regulondb",
        source_release=str(source_release),
        source_release_date=source_release_date,
        source_route=source_route,
        source_url=source_url,
        source_table=source_table,
        raw_payload_ref=raw_payload_ref,
        fetched_at=fetched,
        raw_payload_sha256=raw_checksum,
        query_sha256=query_checksum,
        source_stratum=source_stratum,
    )
    return PromoterRecord(
        source="regulondb",
        source_release=str(source_release),
        source_route=source_route,
        promoter_id=promoter_id,
        promoter_name=_text_or_none(_first_present(payload, "name", "promoter_name", "promoterName")),
        sequence=sequence,
        raw_sequence=raw_sequence,
        sequence_case_policy="uppercase_canonical_preserve_raw",
        sequence_length=len(sequence),
        strand=_normalize_strand(_first_present(payload, "strand")),
        genome_accession=_text_or_none(_first_present(payload, "genomeAccession", "genome_accession", "chromosome")),
        tss_position_raw=_text_or_none(raw_tss),
        tss_interval_0based=tss_interval,
        confidence_level=_text_or_none(_first_present(payload, "confidenceLevel", "confidence", "confidence_level")),
        score=_float_or_none(_first_present(payload, "score", "predictionScore", "prediction_score")),
        evidence=_collect_evidence(_first_present(payload, "evidence", "evidences")),
        citations=_collect_citation_refs(_first_present(payload, "citations", "references")),
        sigma_affiliations=_parse_sigma_affiliations(payload, source_route=source_route),
        boxes=_parse_boxes(payload, source_route=source_route),
        regulatory_sites=_parse_regulatory_sites(payload),
        transcription_units=_parse_tus(payload),
        operon=_parse_operon(payload),
        first_gene=_parse_gene(payload),
        provenance=provenance,
    )
