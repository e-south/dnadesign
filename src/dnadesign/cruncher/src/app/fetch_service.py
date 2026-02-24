"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/fetch_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, TextIO, Tuple

from dnadesign.cruncher.ingest.adapters.base import SourceAdapter
from dnadesign.cruncher.ingest.models import (
    GenomicInterval,
    MotifQuery,
    MotifRecord,
    SiteInstance,
    SiteQuery,
)
from dnadesign.cruncher.ingest.normalize import normalize_site_sequence
from dnadesign.cruncher.ingest.sequence_provider import SequenceProvider
from dnadesign.cruncher.store.catalog_index import CatalogIndex

logger = logging.getLogger(__name__)


def write_motif_record(root: Path, record: MotifRecord) -> Path:
    dest = root / "normalized" / "motifs" / record.descriptor.source
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{record.descriptor.motif_id}.json"
    organism = None
    if record.descriptor.organism is not None:
        organism = {
            "taxon": record.descriptor.organism.taxon,
            "name": record.descriptor.organism.name,
            "strain": record.descriptor.organism.strain,
            "assembly": record.descriptor.organism.assembly,
        }
    provenance = {
        "retrieved_at": record.provenance.retrieved_at.isoformat(),
        "source_version": record.provenance.source_version,
        "source_url": record.provenance.source_url,
        "license": record.provenance.license,
        "citation": record.provenance.citation,
        "raw_artifact_paths": list(record.provenance.raw_artifact_paths),
        "tags": record.provenance.tags,
    }
    payload = {
        "descriptor": {
            "source": record.descriptor.source,
            "motif_id": record.descriptor.motif_id,
            "tf_name": record.descriptor.tf_name,
            "length": record.descriptor.length,
            "alphabet": record.descriptor.alphabet,
            "kind": record.descriptor.kind,
            "organism": organism,
            "tags": record.descriptor.tags,
        },
        "matrix": record.matrix,
        "log_odds_matrix": record.log_odds_matrix,
        "matrix_semantics": record.matrix_semantics,
        "background": record.background,
        "provenance": provenance,
        "checksums": {
            "sha256_raw": record.checksums.sha256_raw,
            "sha256_norm": record.checksums.sha256_norm,
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def _site_to_dict(site: SiteInstance) -> Dict[str, object]:
    coord = None
    if site.coordinate is not None:
        coord = {
            "contig": site.coordinate.contig,
            "start": site.coordinate.start,
            "end": site.coordinate.end,
            "assembly": site.coordinate.assembly,
        }
    org = None
    if site.organism is not None:
        org = {
            "taxon": site.organism.taxon,
            "name": site.organism.name,
            "strain": site.organism.strain,
            "assembly": site.organism.assembly,
        }
    return {
        "source": site.source,
        "site_id": site.site_id,
        "motif_ref": site.motif_ref,
        "organism": org,
        "coordinate": coord,
        "sequence": site.sequence,
        "strand": site.strand,
        "score": site.score,
        "evidence": site.evidence,
        "provenance": {
            "retrieved_at": site.provenance.retrieved_at.isoformat(),
            "source_version": site.provenance.source_version,
            "source_url": site.provenance.source_url,
            "license": site.provenance.license,
            "citation": site.provenance.citation,
            "raw_artifact_paths": list(site.provenance.raw_artifact_paths),
            "tags": site.provenance.tags,
        },
    }


def _hydrate_site_sequence(site: SiteInstance, provider: Optional[SequenceProvider]) -> SiteInstance:
    if provider is None:
        if site.sequence is None and site.coordinate is not None:
            raise ValueError(
                "Binding-site coordinates require genome hydration. Configure ingest.genome_source "
                "or pass --genome-fasta to fetch sequences."
            )
        return site
    if site.sequence:
        return site
    if site.coordinate is None:
        raise ValueError(f"Cannot hydrate sequence for site '{site.site_id}': missing coordinates.")
    seq = provider.fetch(site.coordinate)
    seq = normalize_site_sequence(seq, False)
    tags = dict(site.provenance.tags)
    tags["sequence_hydrated"] = True
    tags["sequence_source"] = provider.source_id
    if site.coordinate.assembly:
        tags["sequence_assembly"] = site.coordinate.assembly
    provenance = replace(site.provenance, tags=tags)
    return replace(site, sequence=seq, provenance=provenance)


def _parse_motif_ref(motif_ref: str) -> Tuple[str, str]:
    if ":" not in motif_ref:
        raise ValueError(f"Invalid motif_ref '{motif_ref}'. Expected '<source>:<motif_id>'.")
    source, motif_id = motif_ref.split(":", 1)
    return source, motif_id


def _motif_path(root: Path, source: str, motif_id: str) -> Path:
    return root / "normalized" / "motifs" / source / f"{motif_id}.json"


def _sites_path(root: Path, source: str, motif_id: str) -> Path:
    return root / "normalized" / "sites" / source / f"{motif_id}.jsonl"


def _new_site_count_payload() -> Dict[str, object]:
    return {
        "total": 0,
        "with_seq": 0,
        "organism": None,
        "length_sum": 0,
        "length_count": 0,
        "length_min": None,
        "length_max": None,
        "length_source": None,
        "site_kind": None,
        "dataset_id": None,
        "dataset_source": None,
        "dataset_method": None,
        "reference_genome": None,
    }


def _merge_count_label(payload: Dict[str, object], key: str, value: object) -> None:
    if value in {None, ""}:
        return
    current = payload.get(key)
    if current is None:
        payload[key] = value
    elif current != value:
        payload[key] = "mixed"


def _update_length_counts(
    *,
    payload: Dict[str, object],
    length: int,
    length_source: str,
) -> None:
    payload["length_sum"] = int(payload["length_sum"]) + int(length)
    payload["length_count"] = int(payload["length_count"]) + 1
    current_min = payload["length_min"]
    current_max = payload["length_max"]
    payload["length_min"] = length if current_min is None else min(current_min, length)
    payload["length_max"] = length if current_max is None else max(current_max, length)
    current_source = payload["length_source"]
    if current_source is None:
        payload["length_source"] = length_source
    elif current_source != length_source:
        payload["length_source"] = "mixed"


def _update_site_counts(payload: Dict[str, object], site: SiteInstance) -> None:
    payload["total"] = int(payload["total"]) + 1
    if site.sequence:
        payload["with_seq"] = int(payload["with_seq"]) + 1
    if payload["organism"] is None and site.organism is not None:
        payload["organism"] = {
            "taxon": site.organism.taxon,
            "name": site.organism.name,
            "strain": site.organism.strain,
            "assembly": site.organism.assembly,
        }
    tags = site.provenance.tags or {}
    _merge_count_label(payload, "site_kind", tags.get("record_kind"))
    _merge_count_label(payload, "dataset_id", tags.get("dataset_id"))
    _merge_count_label(payload, "dataset_source", tags.get("dataset_source"))
    _merge_count_label(payload, "dataset_method", tags.get("dataset_method"))
    reference_genome = tags.get("reference_genome") or tags.get("sequence_assembly")
    _merge_count_label(payload, "reference_genome", reference_genome)

    length = None
    length_source = None
    if site.sequence:
        length = len(site.sequence)
        length_source = "sequence"
    elif site.coordinate is not None:
        length = site.coordinate.end - site.coordinate.start
        length_source = "coordinate"
    if length is None:
        return

    _update_length_counts(payload=payload, length=int(length), length_source=str(length_source))


def _update_catalog_sites_from_counts(
    *,
    catalog: CatalogIndex,
    key: str,
    payload: Dict[str, object],
    tf_name: str,
) -> None:
    site_total = int(payload["total"])
    site_with_seq = int(payload["with_seq"])
    source, motif_id = key.split(":", 1)
    length_count = int(payload.get("length_count", 0))
    mean = None
    if length_count > 0:
        mean = float(payload.get("length_sum", 0)) / length_count
    catalog.upsert_sites(
        source=source,
        motif_id=motif_id,
        tf_name=tf_name,
        site_count=site_with_seq,
        site_total=site_total,
        organism=payload.get("organism"),
        site_kind=payload.get("site_kind"),
        site_length_mean=mean,
        site_length_min=payload.get("length_min"),
        site_length_max=payload.get("length_max"),
        site_length_count=length_count,
        site_length_source=payload.get("length_source"),
        dataset_id=payload.get("dataset_id"),
        dataset_source=payload.get("dataset_source"),
        dataset_method=payload.get("dataset_method"),
        reference_genome=payload.get("reference_genome"),
    )


def _collect_sites_for_query(
    *,
    adapter: SourceAdapter,
    root: Path,
    query: SiteQuery,
    sequence_provider: Optional[SequenceProvider],
    written: list[Path],
) -> tuple[int, Dict[str, Dict[str, object]]]:
    writers: Dict[str, Tuple[Path, TextIO]] = {}
    counts: Dict[str, Dict[str, object]] = {}
    seen = 0
    try:
        for site in adapter.list_sites(query):
            site = _hydrate_site_sequence(site, sequence_provider)
            seen += 1
            source, motif_id = _parse_motif_ref(site.motif_ref)
            key = f"{source}:{motif_id}"
            if key not in writers:
                dest = root / "normalized" / "sites" / source
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / f"{motif_id}.jsonl"
                writers[key] = (path, path.open("w"))
                counts[key] = _new_site_count_payload()
                written.append(path)
            _, fh = writers[key]
            _update_site_counts(counts[key], site)
            fh.write(json.dumps(_site_to_dict(site)) + "\n")
    finally:
        for _, fh in writers.values():
            fh.close()
    return seen, counts


def _hydrate_site_payload_in_place(
    *,
    payload: Dict[str, object],
    provider: SequenceProvider,
) -> tuple[bool, object]:
    coord_payload = payload.get("coordinate")
    seq = payload.get("sequence")
    if seq is None and coord_payload:
        interval = GenomicInterval(
            contig=coord_payload["contig"],
            start=int(coord_payload["start"]),
            end=int(coord_payload["end"]),
            assembly=coord_payload.get("assembly"),
        )
        seq = normalize_site_sequence(provider.fetch(interval), False)
        payload["sequence"] = seq
        provenance = payload.get("provenance") or {}
        tags = dict(provenance.get("tags") or {})
        tags["sequence_hydrated"] = True
        tags["sequence_source"] = provider.source_id
        if interval.assembly:
            tags["sequence_assembly"] = interval.assembly
        provenance["tags"] = tags
        payload["provenance"] = provenance
        return True, coord_payload
    return False, coord_payload


def _update_counts_from_site_payload(
    *,
    counts: Dict[str, object],
    payload: Dict[str, object],
    coord_payload: object,
) -> None:
    counts["total"] = int(counts["total"]) + 1
    if payload.get("sequence"):
        counts["with_seq"] = int(counts["with_seq"]) + 1
    org = payload.get("organism")
    if counts["organism"] is None and org is not None:
        counts["organism"] = org
    tags = (payload.get("provenance") or {}).get("tags") or {}
    _merge_count_label(counts, "site_kind", tags.get("record_kind"))
    _merge_count_label(counts, "dataset_id", tags.get("dataset_id"))
    _merge_count_label(counts, "dataset_source", tags.get("dataset_source"))
    _merge_count_label(counts, "dataset_method", tags.get("dataset_method"))
    reference_genome = tags.get("reference_genome") or tags.get("sequence_assembly")
    _merge_count_label(counts, "reference_genome", reference_genome)
    length = None
    length_source = None
    if payload.get("sequence"):
        length = len(payload["sequence"])
        length_source = "sequence"
    elif coord_payload:
        length = int(coord_payload["end"]) - int(coord_payload["start"])
        length_source = "coordinate"
    if length is not None and length_source is not None:
        _update_length_counts(payload=counts, length=int(length), length_source=str(length_source))


def _hydrate_sites_file(
    *,
    path: Path,
    provider: SequenceProvider,
) -> Tuple[bool, Dict[str, object]]:
    updated = False
    counts: Dict[str, object] = _new_site_count_payload()
    temp_path = path.with_suffix(".jsonl.tmp")
    with path.open() as fh, temp_path.open("w") as out:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            hydrated, coord_payload = _hydrate_site_payload_in_place(payload=payload, provider=provider)
            if hydrated:
                updated = True
            _update_counts_from_site_payload(counts=counts, payload=payload, coord_payload=coord_payload)
            out.write(json.dumps(payload) + "\n")
    if updated:
        temp_path.replace(path)
    else:
        temp_path.unlink(missing_ok=True)
    return updated, counts


def _append_cached_motif_path(
    *,
    root: Path,
    entry: object,
    written: list[Path],
) -> None:
    if not entry.has_matrix:
        raise ValueError(f"Cached entry '{entry.source}:{entry.motif_id}' has no motif matrix.")
    path = _motif_path(root, entry.source, entry.motif_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached motif file: {path}")
    written.append(path)


def _fetch_motif_offline_by_id(
    *,
    adapter: SourceAdapter,
    root: Path,
    catalog: CatalogIndex,
    motif_id: str,
    written: list[Path],
) -> None:
    candidates = [
        entry
        for entry in catalog.entries.values()
        if entry.motif_id == motif_id and entry.source == adapter.source_id and entry.has_matrix
    ]
    if not candidates:
        raise ValueError(f"No cached motif matrix found for motif_id '{motif_id}'.")
    for entry in candidates:
        _append_cached_motif_path(root=root, entry=entry, written=written)


def _fetch_motif_online_by_id(
    *,
    adapter: SourceAdapter,
    root: Path,
    catalog: CatalogIndex,
    motif_id: str,
    update: bool,
    written: list[Path],
) -> None:
    path = _motif_path(root, adapter.source_id, motif_id)
    if path.exists() and not update:
        logger.info("Skipping motif %s (already cached). Use --update to refresh.", motif_id)
        written.append(path)
        return
    logger.info("Fetching motif by id: %s", motif_id)
    record = adapter.get_motif(motif_id)
    written.append(write_motif_record(root, record))
    catalog.upsert_from_record(record)


def _fetch_motif_offline_by_name(
    *,
    adapter: SourceAdapter,
    root: Path,
    catalog: CatalogIndex,
    name: str,
    fetch_all: bool,
    written: list[Path],
) -> None:
    candidates = catalog.list(tf_name=name, source=adapter.source_id, include_synonyms=True)
    if not candidates:
        raise ValueError(f"No cached motifs found for '{name}'.")
    if len(candidates) > 1 and not fetch_all:
        options = ", ".join(f"{c.source}:{c.motif_id}" for c in candidates)
        raise ValueError(
            f"Multiple cached motifs found for '{name}'. Candidates: {options}. "
            "Use --all or --motif-id to disambiguate."
        )
    selected = candidates if fetch_all else candidates[:1]
    for entry in selected:
        _append_cached_motif_path(root=root, entry=entry, written=written)


def _fetch_motif_online_by_name(
    *,
    adapter: SourceAdapter,
    root: Path,
    catalog: CatalogIndex,
    name: str,
    fetch_all: bool,
    update: bool,
    written: list[Path],
) -> None:
    if not update:
        cached = catalog.list(tf_name=name, source=adapter.source_id, include_synonyms=True)
        if any(entry.has_matrix and _motif_path(root, entry.source, entry.motif_id).exists() for entry in cached):
            logger.info(
                "Skipping TF '%s' (cached motif exists). Use --update to refresh.",
                name,
            )
            return
    logger.info("Searching motifs for TF '%s'", name)
    records = adapter.list_motifs(MotifQuery(tf_name=name))
    if not records:
        raise ValueError(f"No motifs found for {name}")
    if len(records) > 1 and not fetch_all:
        options = ", ".join(f"{rec.source}:{rec.motif_id}" for rec in records)
        raise ValueError(f"Multiple motifs found for {name}. Candidates: {options}. Use --all to fetch all.")
    selected = records if fetch_all else records[:1]
    for rec in selected:
        path = _motif_path(root, rec.source, rec.motif_id)
        if path.exists() and not update:
            logger.info(
                "Skipping motif %s:%s (already cached). Use --update to refresh.",
                rec.source,
                rec.motif_id,
            )
            written.append(path)
            continue
        logger.info("Fetching motif %s:%s (%s)", rec.source, rec.motif_id, rec.tf_name)
        record = adapter.get_motif(rec.motif_id)
        written.append(write_motif_record(root, record))
        catalog.upsert_from_record(record)


def fetch_motifs(
    adapter: SourceAdapter,
    root: Path,
    *,
    names: Iterable[str],
    motif_ids: Optional[Iterable[str]] = None,
    fetch_all: bool = False,
    update: bool = False,
    offline: bool = False,
) -> list[Path]:
    written: list[Path] = []
    caps = adapter.capabilities()
    if "motifs:get" not in caps or "motifs:list" not in caps:
        raise ValueError("Adapter does not support motif retrieval")
    catalog = CatalogIndex.load(root)
    motif_ids = list(motif_ids or [])
    if offline and update:
        raise ValueError("offline and update are mutually exclusive")
    for motif_id in motif_ids:
        if offline:
            _fetch_motif_offline_by_id(
                adapter=adapter,
                root=root,
                catalog=catalog,
                motif_id=motif_id,
                written=written,
            )
            continue
        _fetch_motif_online_by_id(
            adapter=adapter,
            root=root,
            catalog=catalog,
            motif_id=motif_id,
            update=update,
            written=written,
        )
    for name in names:
        if offline:
            _fetch_motif_offline_by_name(
                adapter=adapter,
                root=root,
                catalog=catalog,
                name=name,
                fetch_all=fetch_all,
                written=written,
            )
            continue
        _fetch_motif_online_by_name(
            adapter=adapter,
            root=root,
            catalog=catalog,
            name=name,
            fetch_all=fetch_all,
            update=update,
            written=written,
        )
    if written:
        catalog.save(root)
    return written


def _append_cached_sites_path(
    *,
    root: Path,
    entry: object,
    written: list[Path],
) -> None:
    if not entry.has_sites:
        raise ValueError(f"Cached entry '{entry.source}:{entry.motif_id}' has no binding sites.")
    path = _sites_path(root, entry.source, entry.motif_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sites file: {path}")
    written.append(path)


def _update_catalog_from_site_counts(
    *,
    catalog: CatalogIndex,
    counts: Dict[str, Dict[str, object]],
    resolve_tf_name: Callable[[str], str],
) -> None:
    for key, payload in counts.items():
        if int(payload.get("length_count", 0)) <= 0:
            continue
        _update_catalog_sites_from_counts(
            catalog=catalog,
            key=key,
            payload=payload,
            tf_name=resolve_tf_name(key),
        )


def _collect_sites_and_update_catalog(
    *,
    adapter: SourceAdapter,
    root: Path,
    query: SiteQuery,
    sequence_provider: Optional[SequenceProvider],
    written: list[Path],
    catalog: CatalogIndex,
    resolve_tf_name: Callable[[str], str],
    empty_message: str,
) -> None:
    seen, counts = _collect_sites_for_query(
        adapter=adapter,
        root=root,
        query=query,
        sequence_provider=sequence_provider,
        written=written,
    )
    if seen == 0:
        raise ValueError(empty_message)
    _update_catalog_from_site_counts(
        catalog=catalog,
        counts=counts,
        resolve_tf_name=resolve_tf_name,
    )


def _fetch_sites_offline_by_motif_id(
    *,
    adapter: SourceAdapter,
    root: Path,
    catalog: CatalogIndex,
    motif_id: str,
    written: list[Path],
) -> None:
    candidates = [
        entry for entry in catalog.entries.values() if entry.motif_id == motif_id and entry.source == adapter.source_id
    ]
    if not candidates:
        raise ValueError(f"No cached entries found for motif_id '{motif_id}'.")
    for entry in candidates:
        _append_cached_sites_path(root=root, entry=entry, written=written)


def _fetch_sites_online_by_motif_id(
    *,
    adapter: SourceAdapter,
    root: Path,
    catalog: CatalogIndex,
    motif_id: str,
    limit: Optional[int],
    dataset_id: Optional[str],
    update: bool,
    sequence_provider: Optional[SequenceProvider],
    written: list[Path],
) -> None:
    path = _sites_path(root, adapter.source_id, motif_id)
    if path.exists() and not update:
        logger.info(
            "Skipping motif_id '%s' (cached sites exist). Use --update to refresh.",
            motif_id,
        )
        written.append(path)
        return
    logger.info("Fetching binding sites for motif_id '%s'", motif_id)

    def _resolve_tf_name(key: str) -> str:
        existing = catalog.entries.get(key)
        if existing is not None:
            return existing.tf_name
        return motif_id

    _collect_sites_and_update_catalog(
        adapter=adapter,
        root=root,
        query=SiteQuery(motif_id=motif_id, limit=limit, dataset_id=dataset_id),
        sequence_provider=sequence_provider,
        written=written,
        catalog=catalog,
        resolve_tf_name=_resolve_tf_name,
        empty_message=f"No binding sites found for motif_id '{motif_id}'",
    )


def _fetch_sites_offline_by_name(
    *,
    adapter: SourceAdapter,
    root: Path,
    catalog: CatalogIndex,
    name: str,
    dataset_id: Optional[str],
    written: list[Path],
) -> None:
    candidates = catalog.list(tf_name=name, source=adapter.source_id, include_synonyms=True)
    if not candidates:
        raise ValueError(f"No cached entries found for '{name}'.")
    if dataset_id:
        candidates = [c for c in candidates if c.dataset_id == dataset_id]
        if not candidates:
            raise ValueError(f"No cached entries found for '{name}' with dataset_id '{dataset_id}'.")
    if len(candidates) > 1:
        options = ", ".join(f"{c.source}:{c.motif_id}" for c in candidates)
        raise ValueError(
            f"Multiple cached site entries found for '{name}'. Candidates: {options}. Use --motif-id to disambiguate."
        )
    _append_cached_sites_path(root=root, entry=candidates[0], written=written)


def _fetch_sites_online_by_name(
    *,
    adapter: SourceAdapter,
    root: Path,
    catalog: CatalogIndex,
    name: str,
    limit: Optional[int],
    dataset_id: Optional[str],
    update: bool,
    sequence_provider: Optional[SequenceProvider],
    written: list[Path],
) -> None:
    if not update:
        cached = catalog.list(tf_name=name, source=adapter.source_id, include_synonyms=True)
        if dataset_id:
            cached = [entry for entry in cached if entry.dataset_id == dataset_id]
        if any(entry.has_sites and _sites_path(root, entry.source, entry.motif_id).exists() for entry in cached):
            logger.info(
                "Skipping TF '%s' (cached sites exist). Use --update to refresh.",
                name,
            )
            return
    logger.info("Fetching binding sites for TF '%s'", name)
    _collect_sites_and_update_catalog(
        adapter=adapter,
        root=root,
        query=SiteQuery(tf_name=name, limit=limit, dataset_id=dataset_id),
        sequence_provider=sequence_provider,
        written=written,
        catalog=catalog,
        resolve_tf_name=lambda _key: name,
        empty_message=f"No binding sites found for {name}",
    )


def fetch_sites(
    adapter: SourceAdapter,
    root: Path,
    *,
    names: Iterable[str],
    motif_ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    dataset_id: Optional[str] = None,
    update: bool = False,
    offline: bool = False,
    sequence_provider: Optional[SequenceProvider] = None,
) -> List[Path]:
    written: List[Path] = []
    caps = adapter.capabilities()
    if "sites:list" not in caps:
        raise ValueError("Adapter does not support site retrieval")
    catalog = CatalogIndex.load(root)
    if offline and update:
        raise ValueError("offline and update are mutually exclusive")
    motif_ids = list(motif_ids or [])
    for motif_id in motif_ids:
        if offline:
            _fetch_sites_offline_by_motif_id(
                adapter=adapter,
                root=root,
                catalog=catalog,
                motif_id=motif_id,
                written=written,
            )
            continue
        _fetch_sites_online_by_motif_id(
            adapter=adapter,
            root=root,
            catalog=catalog,
            motif_id=motif_id,
            limit=limit,
            dataset_id=dataset_id,
            update=update,
            sequence_provider=sequence_provider,
            written=written,
        )
    for name in names:
        if offline:
            _fetch_sites_offline_by_name(
                adapter=adapter,
                root=root,
                catalog=catalog,
                name=name,
                dataset_id=dataset_id,
                written=written,
            )
            continue
        _fetch_sites_online_by_name(
            adapter=adapter,
            root=root,
            catalog=catalog,
            name=name,
            limit=limit,
            dataset_id=dataset_id,
            update=update,
            sequence_provider=sequence_provider,
            written=written,
        )
    if written:
        catalog.save(root)
    return written


def _targets_from_all_cached_sites(catalog: CatalogIndex) -> list[tuple[str, str, str]]:
    targets: list[tuple[str, str, str]] = []
    for entry in catalog.entries.values():
        if entry.has_sites:
            targets.append((entry.tf_name, entry.source, entry.motif_id))
    return targets


def _append_targets_for_motif_ids(
    *,
    catalog: CatalogIndex,
    motif_ids: list[str],
    targets: list[tuple[str, str, str]],
) -> None:
    for motif_id in motif_ids:
        candidates = [entry for entry in catalog.entries.values() if entry.motif_id == motif_id]
        if not candidates:
            raise ValueError(f"No cached entries found for motif_id '{motif_id}'.")
        for entry in candidates:
            targets.append((entry.tf_name, entry.source, entry.motif_id))


def _append_targets_for_names(
    *,
    catalog: CatalogIndex,
    names: list[str],
    targets: list[tuple[str, str, str]],
) -> None:
    for name in names:
        candidates = catalog.list(tf_name=name, include_synonyms=True)
        if not candidates:
            raise ValueError(f"No cached entries found for '{name}'.")
        for entry in candidates:
            targets.append((entry.tf_name, entry.source, entry.motif_id))


def _resolve_hydrate_targets(
    *,
    catalog: CatalogIndex,
    names: Iterable[str] | None,
    motif_ids: Iterable[str] | None,
) -> list[tuple[str, str, str]]:
    targets: list[tuple[str, str, str]] = []
    motif_id_values = list(motif_ids or [])
    name_values = list(names or [])
    if not name_values and not motif_id_values:
        targets = _targets_from_all_cached_sites(catalog)
        if not targets:
            raise ValueError("No cached binding sites found to hydrate.")
        return targets
    _append_targets_for_motif_ids(catalog=catalog, motif_ids=motif_id_values, targets=targets)
    _append_targets_for_names(catalog=catalog, names=name_values, targets=targets)
    return targets


def _upsert_hydrated_site_counts(
    *,
    catalog: CatalogIndex,
    tf_name: str,
    source: str,
    motif_id: str,
    counts: Dict[str, object],
) -> None:
    length_count = int(counts.get("length_count", 0))
    mean = None
    if length_count > 0:
        mean = float(counts.get("length_sum", 0)) / length_count
    catalog.upsert_sites(
        source=source,
        motif_id=motif_id,
        tf_name=tf_name,
        site_count=int(counts.get("with_seq", 0)),
        site_total=int(counts.get("total", 0)),
        organism=counts.get("organism"),
        site_kind=counts.get("site_kind"),
        site_length_mean=mean,
        site_length_min=counts.get("length_min"),
        site_length_max=counts.get("length_max"),
        site_length_count=length_count,
        site_length_source=counts.get("length_source"),
        dataset_id=counts.get("dataset_id"),
        dataset_source=counts.get("dataset_source"),
        dataset_method=counts.get("dataset_method"),
        reference_genome=counts.get("reference_genome"),
    )


def hydrate_sites(
    root: Path,
    *,
    names: Iterable[str] | None,
    motif_ids: Optional[Iterable[str]] = None,
    sequence_provider: SequenceProvider,
) -> List[Path]:
    if sequence_provider is None:
        raise ValueError("Sequence provider is required to hydrate cached sites.")
    catalog = CatalogIndex.load(root)
    written: list[Path] = []
    targets = _resolve_hydrate_targets(
        catalog=catalog,
        names=names,
        motif_ids=motif_ids,
    )
    seen: set[str] = set()
    for tf_name, source, motif_id in targets:
        key = f"{source}:{motif_id}"
        if key in seen:
            continue
        seen.add(key)
        path = _sites_path(root, source, motif_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached sites file: {path}")
        updated, counts = _hydrate_sites_file(path=path, provider=sequence_provider)
        if updated:
            written.append(path)
        _upsert_hydrated_site_counts(
            catalog=catalog,
            tf_name=tf_name,
            source=source,
            motif_id=motif_id,
            counts=counts,
        )
    if written:
        catalog.save(root)
    return written
