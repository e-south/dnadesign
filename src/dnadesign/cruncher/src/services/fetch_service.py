"""Fetch motifs/sites from adapters into the local catalog cache."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple

from dnadesign.cruncher.ingest.adapters.base import SourceAdapter
from dnadesign.cruncher.ingest.models import GenomicInterval, MotifQuery, MotifRecord, SiteInstance, SiteQuery
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


def _hydrate_sites_file(
    *,
    path: Path,
    provider: SequenceProvider,
) -> Tuple[bool, Dict[str, object]]:
    updated = False
    counts: Dict[str, object] = {
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
    temp_path = path.with_suffix(".jsonl.tmp")
    with path.open() as fh, temp_path.open("w") as out:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
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
                updated = True
            counts["total"] = int(counts["total"]) + 1
            if payload.get("sequence"):
                counts["with_seq"] = int(counts["with_seq"]) + 1
            org = payload.get("organism")
            if counts["organism"] is None and org is not None:
                counts["organism"] = org
            tags = (payload.get("provenance") or {}).get("tags") or {}
            record_kind = tags.get("record_kind")
            if record_kind:
                current = counts["site_kind"]
                if current is None:
                    counts["site_kind"] = record_kind
                elif current != record_kind:
                    counts["site_kind"] = "mixed"
            dataset_id = tags.get("dataset_id")
            if dataset_id:
                current = counts["dataset_id"]
                if current is None:
                    counts["dataset_id"] = dataset_id
                elif current != dataset_id:
                    counts["dataset_id"] = "mixed"
            dataset_source = tags.get("dataset_source")
            if dataset_source:
                current = counts["dataset_source"]
                if current is None:
                    counts["dataset_source"] = dataset_source
                elif current != dataset_source:
                    counts["dataset_source"] = "mixed"
            dataset_method = tags.get("dataset_method")
            if dataset_method:
                current = counts["dataset_method"]
                if current is None:
                    counts["dataset_method"] = dataset_method
                elif current != dataset_method:
                    counts["dataset_method"] = "mixed"
            reference_genome = tags.get("reference_genome") or tags.get("sequence_assembly")
            if reference_genome:
                current = counts["reference_genome"]
                if current is None:
                    counts["reference_genome"] = reference_genome
                elif current != reference_genome:
                    counts["reference_genome"] = "mixed"
            length = None
            length_source = None
            if payload.get("sequence"):
                length = len(payload["sequence"])
                length_source = "sequence"
            elif coord_payload:
                length = int(coord_payload["end"]) - int(coord_payload["start"])
                length_source = "coordinate"
            if length is not None:
                counts["length_sum"] = int(counts["length_sum"]) + int(length)
                counts["length_count"] = int(counts["length_count"]) + 1
                current_min = counts["length_min"]
                current_max = counts["length_max"]
                counts["length_min"] = length if current_min is None else min(current_min, length)
                counts["length_max"] = length if current_max is None else max(current_max, length)
                current_source = counts["length_source"]
                if current_source is None:
                    counts["length_source"] = length_source
                elif current_source != length_source:
                    counts["length_source"] = "mixed"
            out.write(json.dumps(payload) + "\n")
    if updated:
        temp_path.replace(path)
    else:
        temp_path.unlink(missing_ok=True)
    return updated, counts


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
            candidates = [
                entry
                for entry in catalog.entries.values()
                if entry.motif_id == motif_id and entry.source == adapter.source_id and entry.has_matrix
            ]
            if not candidates:
                raise ValueError(f"No cached motif matrix found for motif_id '{motif_id}'.")
            for entry in candidates:
                path = _motif_path(root, entry.source, entry.motif_id)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached motif file: {path}")
                written.append(path)
            continue
        path = _motif_path(root, adapter.source_id, motif_id)
        if path.exists() and not update:
            logger.info("Skipping motif %s (already cached). Use --update to refresh.", motif_id)
            written.append(path)
            continue
        logger.info("Fetching motif by id: %s", motif_id)
        record = adapter.get_motif(motif_id)
        written.append(write_motif_record(root, record))
        catalog.upsert_from_record(record)
    for name in names:
        if offline:
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
                if not entry.has_matrix:
                    raise ValueError(f"Cached entry '{entry.source}:{entry.motif_id}' has no motif matrix.")
                path = _motif_path(root, entry.source, entry.motif_id)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached motif file: {path}")
                written.append(path)
            continue
        if not update:
            cached = catalog.list(tf_name=name, include_synonyms=True)
            if any(entry.has_matrix and _motif_path(root, entry.source, entry.motif_id).exists() for entry in cached):
                logger.info("Skipping TF '%s' (cached motif exists). Use --update to refresh.", name)
                continue
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
                logger.info("Skipping motif %s:%s (already cached). Use --update to refresh.", rec.source, rec.motif_id)
                written.append(path)
                continue
            logger.info("Fetching motif %s:%s (%s)", rec.source, rec.motif_id, rec.tf_name)
            record = adapter.get_motif(rec.motif_id)
            written.append(write_motif_record(root, record))
            catalog.upsert_from_record(record)
    if written:
        catalog.save(root)
    return written


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
            candidates = [
                entry
                for entry in catalog.entries.values()
                if entry.motif_id == motif_id and entry.source == adapter.source_id
            ]
            if not candidates:
                raise ValueError(f"No cached entries found for motif_id '{motif_id}'.")
            for entry in candidates:
                if not entry.has_sites:
                    raise ValueError(f"Cached entry '{entry.source}:{entry.motif_id}' has no binding sites.")
                path = _sites_path(root, entry.source, entry.motif_id)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached sites file: {path}")
                written.append(path)
            continue
        path = _sites_path(root, adapter.source_id, motif_id)
        if path.exists() and not update:
            logger.info("Skipping motif_id '%s' (cached sites exist). Use --update to refresh.", motif_id)
            written.append(path)
            continue
        logger.info("Fetching binding sites for motif_id '%s'", motif_id)
        writers: Dict[str, Tuple[Path, TextIO]] = {}
        counts: Dict[str, Dict[str, object]] = {}
        seen = 0
        try:
            for site in adapter.list_sites(SiteQuery(motif_id=motif_id, limit=limit, dataset_id=dataset_id)):
                site = _hydrate_site_sequence(site, sequence_provider)
                seen += 1
                source, site_motif_id = _parse_motif_ref(site.motif_ref)
                key = f"{source}:{site_motif_id}"
                if key not in writers:
                    dest = root / "normalized" / "sites" / source
                    dest.mkdir(parents=True, exist_ok=True)
                    path = dest / f"{site_motif_id}.jsonl"
                    writers[key] = (path, path.open("w"))
                    counts[key] = {
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
                    written.append(path)
                path, fh = writers[key]
                site_total = int(counts[key]["total"])
                site_with_seq = int(counts[key]["with_seq"])
                if site.sequence:
                    site_with_seq += 1
                site_total += 1
                if counts[key]["organism"] is None and site.organism is not None:
                    counts[key]["organism"] = {
                        "taxon": site.organism.taxon,
                        "name": site.organism.name,
                        "strain": site.organism.strain,
                        "assembly": site.organism.assembly,
                    }
                tags = site.provenance.tags or {}
                record_kind = tags.get("record_kind")
                if record_kind:
                    current = counts[key]["site_kind"]
                    if current is None:
                        counts[key]["site_kind"] = record_kind
                    elif current != record_kind:
                        counts[key]["site_kind"] = "mixed"
                dataset_id = tags.get("dataset_id")
                if dataset_id:
                    current = counts[key]["dataset_id"]
                    if current is None:
                        counts[key]["dataset_id"] = dataset_id
                    elif current != dataset_id:
                        counts[key]["dataset_id"] = "mixed"
                dataset_source = tags.get("dataset_source")
                if dataset_source:
                    current = counts[key]["dataset_source"]
                    if current is None:
                        counts[key]["dataset_source"] = dataset_source
                    elif current != dataset_source:
                        counts[key]["dataset_source"] = "mixed"
                dataset_method = tags.get("dataset_method")
                if dataset_method:
                    current = counts[key]["dataset_method"]
                    if current is None:
                        counts[key]["dataset_method"] = dataset_method
                    elif current != dataset_method:
                        counts[key]["dataset_method"] = "mixed"
                reference_genome = tags.get("reference_genome") or tags.get("sequence_assembly")
                if reference_genome:
                    current = counts[key]["reference_genome"]
                    if current is None:
                        counts[key]["reference_genome"] = reference_genome
                    elif current != reference_genome:
                        counts[key]["reference_genome"] = "mixed"
                length = None
                length_source = None
                if site.sequence:
                    length = len(site.sequence)
                    length_source = "sequence"
                elif site.coordinate is not None:
                    length = site.coordinate.end - site.coordinate.start
                    length_source = "coordinate"
                if length is not None:
                    counts[key]["length_sum"] = int(counts[key]["length_sum"]) + int(length)
                    counts[key]["length_count"] = int(counts[key]["length_count"]) + 1
                    current_min = counts[key]["length_min"]
                    current_max = counts[key]["length_max"]
                    counts[key]["length_min"] = length if current_min is None else min(current_min, length)
                    counts[key]["length_max"] = length if current_max is None else max(current_max, length)
                    current_source = counts[key]["length_source"]
                    if current_source is None:
                        counts[key]["length_source"] = length_source
                    elif current_source != length_source:
                        counts[key]["length_source"] = "mixed"
                counts[key]["total"] = site_total
                counts[key]["with_seq"] = site_with_seq
                fh.write(json.dumps(_site_to_dict(site)) + "\n")
        finally:
            for _, fh in writers.values():
                fh.close()
        if seen == 0:
            raise ValueError(f"No binding sites found for motif_id '{motif_id}'")
        for key, payload in counts.items():
            site_total = int(payload["total"])
            site_with_seq = int(payload["with_seq"])
            source, site_motif_id = key.split(":", 1)
            tf_name = motif_id
            existing = catalog.entries.get(key)
            if existing is not None:
                tf_name = existing.tf_name
            length_count = int(payload.get("length_count", 0))
            mean = None
            if length_count > 0:
                mean = float(payload.get("length_sum", 0)) / length_count
                catalog.upsert_sites(
                    source=source,
                    motif_id=site_motif_id,
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
    for name in names:
        if offline:
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
                    f"Multiple cached site entries found for '{name}'. Candidates: {options}. "
                    "Use --motif-id to disambiguate."
                )
            entry = candidates[0]
            if not entry.has_sites:
                raise ValueError(f"Cached entry '{entry.source}:{entry.motif_id}' has no binding sites.")
            path = _sites_path(root, entry.source, entry.motif_id)
            if not path.exists():
                raise FileNotFoundError(f"Missing cached sites file: {path}")
            written.append(path)
            continue
        if not update:
            cached = catalog.list(tf_name=name, include_synonyms=True)
            if dataset_id:
                cached = [entry for entry in cached if entry.dataset_id == dataset_id]
            if any(entry.has_sites and _sites_path(root, entry.source, entry.motif_id).exists() for entry in cached):
                logger.info("Skipping TF '%s' (cached sites exist). Use --update to refresh.", name)
                continue
        logger.info("Fetching binding sites for TF '%s'", name)
        writers: Dict[str, Tuple[Path, TextIO]] = {}
        counts: Dict[str, Dict[str, object]] = {}
        seen = 0
        try:
            for site in adapter.list_sites(SiteQuery(tf_name=name, limit=limit, dataset_id=dataset_id)):
                site = _hydrate_site_sequence(site, sequence_provider)
                seen += 1
                source, motif_id = _parse_motif_ref(site.motif_ref)
                key = f"{source}:{motif_id}"
                if key not in writers:
                    dest = root / "normalized" / "sites" / source
                    dest.mkdir(parents=True, exist_ok=True)
                    path = dest / f"{motif_id}.jsonl"
                    writers[key] = (path, path.open("w"))
                    counts[key] = {
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
                    written.append(path)
                path, fh = writers[key]
                site_total = int(counts[key]["total"])
                site_with_seq = int(counts[key]["with_seq"])
                if site.sequence:
                    site_with_seq += 1
                site_total += 1
                if counts[key]["organism"] is None and site.organism is not None:
                    counts[key]["organism"] = {
                        "taxon": site.organism.taxon,
                        "name": site.organism.name,
                        "strain": site.organism.strain,
                        "assembly": site.organism.assembly,
                    }
                tags = site.provenance.tags or {}
                record_kind = tags.get("record_kind")
                if record_kind:
                    current = counts[key]["site_kind"]
                    if current is None:
                        counts[key]["site_kind"] = record_kind
                    elif current != record_kind:
                        counts[key]["site_kind"] = "mixed"
                dataset_id = tags.get("dataset_id")
                if dataset_id:
                    current = counts[key]["dataset_id"]
                    if current is None:
                        counts[key]["dataset_id"] = dataset_id
                    elif current != dataset_id:
                        counts[key]["dataset_id"] = "mixed"
                dataset_source = tags.get("dataset_source")
                if dataset_source:
                    current = counts[key]["dataset_source"]
                    if current is None:
                        counts[key]["dataset_source"] = dataset_source
                    elif current != dataset_source:
                        counts[key]["dataset_source"] = "mixed"
                dataset_method = tags.get("dataset_method")
                if dataset_method:
                    current = counts[key]["dataset_method"]
                    if current is None:
                        counts[key]["dataset_method"] = dataset_method
                    elif current != dataset_method:
                        counts[key]["dataset_method"] = "mixed"
                reference_genome = tags.get("reference_genome") or tags.get("sequence_assembly")
                if reference_genome:
                    current = counts[key]["reference_genome"]
                    if current is None:
                        counts[key]["reference_genome"] = reference_genome
                    elif current != reference_genome:
                        counts[key]["reference_genome"] = "mixed"
                length = None
                length_source = None
                if site.sequence:
                    length = len(site.sequence)
                    length_source = "sequence"
                elif site.coordinate is not None:
                    length = site.coordinate.end - site.coordinate.start
                    length_source = "coordinate"
                if length is not None:
                    counts[key]["length_sum"] = int(counts[key]["length_sum"]) + int(length)
                    counts[key]["length_count"] = int(counts[key]["length_count"]) + 1
                    current_min = counts[key]["length_min"]
                    current_max = counts[key]["length_max"]
                    counts[key]["length_min"] = length if current_min is None else min(current_min, length)
                    counts[key]["length_max"] = length if current_max is None else max(current_max, length)
                    current_source = counts[key]["length_source"]
                    if current_source is None:
                        counts[key]["length_source"] = length_source
                    elif current_source != length_source:
                        counts[key]["length_source"] = "mixed"
                counts[key]["total"] = site_total
                counts[key]["with_seq"] = site_with_seq
                fh.write(json.dumps(_site_to_dict(site)) + "\n")
        finally:
            for _, fh in writers.values():
                fh.close()
        if seen == 0:
            raise ValueError(f"No binding sites found for {name}")
        for key, payload in counts.items():
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
                    tf_name=name,
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
    if written:
        catalog.save(root)
    return written


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
    motif_ids = list(motif_ids or [])
    names = list(names or [])
    written: list[Path] = []
    targets: list[Tuple[str, str, str]] = []
    if not names and not motif_ids:
        for entry in catalog.entries.values():
            if entry.has_sites:
                targets.append((entry.tf_name, entry.source, entry.motif_id))
        if not targets:
            raise ValueError("No cached binding sites found to hydrate.")
    for motif_id in motif_ids:
        candidates = [entry for entry in catalog.entries.values() if entry.motif_id == motif_id]
        if not candidates:
            raise ValueError(f"No cached entries found for motif_id '{motif_id}'.")
        for entry in candidates:
            targets.append((entry.tf_name, entry.source, entry.motif_id))
    for name in names:
        candidates = catalog.list(tf_name=name, include_synonyms=True)
        if not candidates:
            raise ValueError(f"No cached entries found for '{name}'.")
        for entry in candidates:
            targets.append((entry.tf_name, entry.source, entry.motif_id))
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
    if written:
        catalog.save(root)
    return written
