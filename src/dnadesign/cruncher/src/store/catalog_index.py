"""Catalog index stored in the local .cruncher cache."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dnadesign.cruncher.ingest.models import MotifDescriptor, MotifRecord, OrganismRef

CATALOG_VERSION = 1


@dataclass
class CatalogEntry:
    source: str
    motif_id: str
    tf_name: str
    kind: str
    organism: Optional[Dict[str, object]] = None
    matrix_length: Optional[int] = None
    matrix_source: Optional[str] = None
    matrix_semantics: Optional[str] = None
    has_matrix: bool = False
    has_sites: bool = False
    site_count: int = 0
    site_total: int = 0
    site_kind: Optional[str] = None
    site_length_mean: Optional[float] = None
    site_length_min: Optional[int] = None
    site_length_max: Optional[int] = None
    site_length_count: int = 0
    site_length_source: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_source: Optional[str] = None
    dataset_method: Optional[str] = None
    reference_genome: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.motif_id}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "motif_id": self.motif_id,
            "tf_name": self.tf_name,
            "kind": self.kind,
            "organism": self.organism,
            "matrix_length": self.matrix_length,
            "matrix_source": self.matrix_source,
            "matrix_semantics": self.matrix_semantics,
            "has_matrix": self.has_matrix,
            "has_sites": self.has_sites,
            "site_count": self.site_count,
            "site_total": self.site_total,
            "site_kind": self.site_kind,
            "site_length_mean": self.site_length_mean,
            "site_length_min": self.site_length_min,
            "site_length_max": self.site_length_max,
            "site_length_count": self.site_length_count,
            "site_length_source": self.site_length_source,
            "dataset_id": self.dataset_id,
            "dataset_source": self.dataset_source,
            "dataset_method": self.dataset_method,
            "reference_genome": self.reference_genome,
            "updated_at": self.updated_at,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CatalogEntry":
        return cls(
            source=data["source"],
            motif_id=data["motif_id"],
            tf_name=data["tf_name"],
            kind=data.get("kind", "PFM"),
            organism=data.get("organism"),
            matrix_length=data.get("matrix_length"),
            matrix_source=data.get("matrix_source"),
            matrix_semantics=data.get("matrix_semantics"),
            has_matrix=bool(data.get("has_matrix", False)),
            has_sites=bool(data.get("has_sites", False)),
            site_count=int(data.get("site_count", 0)),
            site_total=int(data.get("site_total", 0)),
            site_kind=data.get("site_kind"),
            site_length_mean=(float(data["site_length_mean"]) if data.get("site_length_mean") is not None else None),
            site_length_min=(int(data["site_length_min"]) if data.get("site_length_min") is not None else None),
            site_length_max=(int(data["site_length_max"]) if data.get("site_length_max") is not None else None),
            site_length_count=int(data.get("site_length_count", 0)),
            site_length_source=data.get("site_length_source"),
            dataset_id=data.get("dataset_id"),
            dataset_source=data.get("dataset_source"),
            dataset_method=data.get("dataset_method"),
            reference_genome=data.get("reference_genome"),
            updated_at=data.get("updated_at") or datetime.now(timezone.utc).isoformat(),
            tags=dict(data.get("tags") or {}),
        )


@dataclass
class CatalogIndex:
    entries: Dict[str, CatalogEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "CatalogIndex":
        path = catalog_path(root)
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text())
        if payload.get("version") != CATALOG_VERSION:
            raise ValueError(f"Unsupported catalog version {payload.get('version')}")
        raw_entries = payload.get("entries", {})
        entries = {key: CatalogEntry.from_dict(value) for key, value in raw_entries.items()}
        return cls(entries=entries)

    def save(self, root: Path) -> None:
        path = catalog_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CATALOG_VERSION,
            "entries": {key: entry.to_dict() for key, entry in self.entries.items()},
        }
        path.write_text(json.dumps(payload, indent=2))

    def upsert_from_record(self, record: MotifRecord) -> None:
        key = f"{record.descriptor.source}:{record.descriptor.motif_id}"
        existing = self.entries.get(key)
        entry = existing or CatalogEntry(
            source=record.descriptor.source,
            motif_id=record.descriptor.motif_id,
            tf_name=record.descriptor.tf_name,
            kind=record.descriptor.kind,
        )
        entry.tf_name = record.descriptor.tf_name
        entry.kind = record.descriptor.kind
        if record.descriptor.organism is not None:
            entry.organism = {
                "taxon": record.descriptor.organism.taxon,
                "name": record.descriptor.organism.name,
                "strain": record.descriptor.organism.strain,
                "assembly": record.descriptor.organism.assembly,
            }
        entry.matrix_length = record.descriptor.length
        entry.matrix_source = record.descriptor.tags.get("matrix_source") if record.descriptor.tags else None
        entry.matrix_semantics = record.matrix_semantics
        entry.has_matrix = True
        entry.tags.update(record.descriptor.tags or {})
        entry.updated_at = datetime.now(timezone.utc).isoformat()
        self.entries[key] = entry

    def upsert_sites(
        self,
        *,
        source: str,
        motif_id: str,
        tf_name: str,
        site_count: int,
        site_total: int,
        organism: Optional[Dict[str, object]] = None,
        site_kind: Optional[str] = None,
        site_length_mean: Optional[float] = None,
        site_length_min: Optional[int] = None,
        site_length_max: Optional[int] = None,
        site_length_count: int = 0,
        site_length_source: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_source: Optional[str] = None,
        dataset_method: Optional[str] = None,
        reference_genome: Optional[str] = None,
    ) -> None:
        key = f"{source}:{motif_id}"
        existing = self.entries.get(key)
        entry = existing or CatalogEntry(source=source, motif_id=motif_id, tf_name=tf_name, kind="PFM")
        entry.tf_name = tf_name
        entry.has_sites = True
        entry.site_count = site_count
        entry.site_total = site_total
        entry.site_kind = site_kind or entry.site_kind
        entry.site_length_mean = site_length_mean
        entry.site_length_min = site_length_min
        entry.site_length_max = site_length_max
        entry.site_length_count = site_length_count
        entry.site_length_source = site_length_source
        entry.dataset_id = dataset_id or entry.dataset_id
        entry.dataset_source = dataset_source or entry.dataset_source
        entry.dataset_method = dataset_method or entry.dataset_method
        entry.reference_genome = reference_genome or entry.reference_genome
        if organism is not None:
            entry.organism = organism
        entry.updated_at = datetime.now(timezone.utc).isoformat()
        self.entries[key] = entry

    def list(
        self,
        *,
        tf_name: Optional[str] = None,
        source: Optional[str] = None,
        organism: Optional[dict[str, object]] = None,
        include_synonyms: bool = False,
    ) -> List[CatalogEntry]:
        entries = list(self.entries.values())
        if source:
            entries = [entry for entry in entries if entry.source == source]
        if tf_name:
            tf_norm = tf_name.lower()
            if include_synonyms:
                entries = [
                    entry for entry in entries if entry.tf_name.lower() == tf_norm or tf_norm in _synonyms(entry)
                ]
            else:
                entries = [entry for entry in entries if entry.tf_name.lower() == tf_norm]
        if organism:

            def _match_org(entry: CatalogEntry) -> bool:
                if entry.organism is None:
                    return False
                if organism.get("taxon") is not None and entry.organism.get("taxon") != organism.get("taxon"):
                    return False
                if organism.get("name") and entry.organism.get("name"):
                    return entry.organism.get("name", "").lower() == str(organism.get("name")).lower()
                if organism.get("strain") and entry.organism.get("strain"):
                    return entry.organism.get("strain", "").lower() == str(organism.get("strain")).lower()
                return True

            entries = [entry for entry in entries if _match_org(entry)]
        return sorted(entries, key=lambda e: (e.tf_name.lower(), e.source, e.motif_id))

    def search(
        self,
        *,
        query: str,
        source: Optional[str] = None,
        organism: Optional[dict[str, object]] = None,
        regex: bool = False,
        case_sensitive: bool = False,
        fuzzy: bool = False,
        min_score: float = 0.6,
        limit: Optional[int] = None,
    ) -> List[CatalogEntry]:
        entries = self.list(source=source, organism=organism)
        q = query.strip()
        if not q:
            return entries
        if fuzzy:
            try:
                from Levenshtein import ratio as levenshtein_ratio
            except ImportError as exc:  # pragma: no cover - dependency managed via pyproject
                raise RuntimeError("python-Levenshtein is required for fuzzy search.") from exc
            q_norm = q if case_sensitive else q.lower()
            scored: list[tuple[float, CatalogEntry]] = []
            for entry in entries:
                candidates = [entry.tf_name] + _synonyms(entry, raw=True)
                best = 0.0
                for cand in candidates:
                    if not cand:
                        continue
                    cand_norm = cand if case_sensitive else cand.lower()
                    score = levenshtein_ratio(q_norm, cand_norm)
                    if score > best:
                        best = score
                if best >= min_score:
                    scored.append((best, entry))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            ranked = [entry for _, entry in scored]
            if limit is not None:
                return ranked[:limit]
            return ranked
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(q, flags=flags)
            return [
                entry
                for entry in entries
                if pattern.search(entry.tf_name) or any(pattern.search(s) for s in _synonyms(entry, raw=True))
            ]
        if case_sensitive:
            return [entry for entry in entries if q in entry.tf_name or any(q in s for s in _synonyms(entry, raw=True))]
        q = q.lower()
        return [
            entry
            for entry in entries
            if q in entry.tf_name.lower() or any(q in s.lower() for s in _synonyms(entry, raw=True))
        ]


def _synonyms(entry: CatalogEntry, *, raw: bool = False) -> list[str]:
    raw_syn = entry.tags.get("synonyms", "") if entry.tags else ""
    if not raw_syn:
        return []
    parts = [part.strip() for part in re.split(r"[;,|]+", str(raw_syn)) if part.strip()]
    if raw:
        return parts
    return [part.lower() for part in parts]


def catalog_path(root: Path) -> Path:
    return root / "catalog.json"


def build_descriptor(entry: CatalogEntry) -> MotifDescriptor:
    organism = None
    if entry.organism is not None:
        organism = {
            "taxon": entry.organism.get("taxon"),
            "name": entry.organism.get("name"),
            "strain": entry.organism.get("strain"),
            "assembly": entry.organism.get("assembly"),
        }
    return MotifDescriptor(
        source=entry.source,
        motif_id=entry.motif_id,
        tf_name=entry.tf_name,
        organism=None if organism is None else OrganismRef(**organism),
        length=entry.matrix_length or 0,
        kind=entry.kind,
        tags=entry.tags,
    )
