"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/ingest/adapters/local_sites.py

Ingest binding-site sequences from a local FASTA file.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Set

from dnadesign.cruncher.ingest.models import (
    DatasetDescriptor,
    DatasetQuery,
    MotifDescriptor,
    MotifQuery,
    MotifRecord,
    OrganismRef,
    Provenance,
    SiteInstance,
    SiteQuery,
)
from dnadesign.cruncher.ingest.normalize import normalize_site_sequence


@dataclass(frozen=True, slots=True)
class LocalSiteAdapterConfig:
    source_id: str
    path: Path
    tf_name: Optional[str] = None
    organism: Optional[OrganismRef] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    record_kind: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)


class LocalSiteAdapter:
    """Ingest binding sites from a local FASTA file."""

    def __init__(self, config: LocalSiteAdapterConfig) -> None:
        self.source_id = config.source_id
        self._config = config
        self._path = config.path
        if not self._path.exists():
            raise FileNotFoundError(f"Local site FASTA does not exist: {self._path}")
        if not self._path.is_file():
            raise FileNotFoundError(f"Local site FASTA is not a file: {self._path}")

    def capabilities(self) -> Set[str]:
        return {"sites:list", "motifs:list", "motifs:iter"}

    def list_motifs(self, query: MotifQuery) -> list[MotifDescriptor]:
        del query
        return []

    def iter_motifs(self, query: MotifQuery, *, page_size: int = 200) -> Iterable[MotifDescriptor]:
        del query
        del page_size
        return []

    def get_motif(self, motif_id: str) -> MotifRecord:
        raise ValueError(f"Motif lookup is not supported for local site source '{self.source_id}'.")

    def list_sites(self, query: SiteQuery) -> Iterable[SiteInstance]:
        return self._iter_sites(query)

    def get_sites_for_motif(self, motif_id: str, query: SiteQuery) -> Iterable[SiteInstance]:
        site_query = SiteQuery(
            organism=query.organism,
            motif_id=motif_id,
            tf_name=query.tf_name,
            dataset_id=query.dataset_id,
            region=query.region,
            limit=query.limit,
        )
        return self._iter_sites(site_query)

    def list_datasets(self, query: DatasetQuery) -> list[DatasetDescriptor]:
        del query
        return []

    def _iter_sites(self, query: SiteQuery) -> Iterable[SiteInstance]:
        filter_name = query.motif_id or query.tf_name
        remaining = int(query.limit) if query.limit is not None else None
        for header, sequence in self._iter_fasta():
            tf_name, evidence, strand = self._parse_header(header)
            if filter_name and tf_name.lower() != filter_name.lower():
                continue
            seq = normalize_site_sequence(sequence, False)
            provenance = self._provenance()
            yield SiteInstance(
                source=self.source_id,
                site_id=self._site_id(header, seq),
                motif_ref=f"{self.source_id}:{tf_name}",
                organism=self._config.organism,
                coordinate=None,
                sequence=seq,
                strand=strand,
                score=None,
                evidence=evidence,
                provenance=provenance,
            )
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    return

    def _iter_fasta(self) -> Iterable[tuple[str, str]]:
        header: Optional[str] = None
        seq_parts: list[str] = []
        with self._path.open() as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        yield header, "".join(seq_parts)
                    header = line[1:].strip()
                    seq_parts = []
                    continue
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)

    def _parse_header(self, header: str) -> tuple[str, dict[str, str], Optional[str]]:
        parts = [part.strip() for part in header.split("|") if part.strip()]
        if not parts:
            raise ValueError("FASTA header is empty.")
        tf_name = parts[0]
        if self._config.tf_name:
            if tf_name.lower() != self._config.tf_name.lower():
                raise ValueError(
                    f"FASTA header TF '{tf_name}' does not match configured tf_name '{self._config.tf_name}'."
                )
            tf_name = self._config.tf_name
        evidence: dict[str, str] = {"header": header}
        if len(parts) > 1:
            evidence["peak_id"] = parts[1]
        if len(parts) > 2:
            evidence["coord"] = parts[2]
        for token in parts[3:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                evidence[key] = value
        strand = None
        if "strand" in evidence:
            strand = self._normalize_strand(evidence["strand"])
        return tf_name, evidence, strand

    def _normalize_strand(self, value: str) -> Optional[str]:
        raw = value.strip().lower()
        if raw in {"+", "plus", "forward"}:
            return "+"
        if raw in {"-", "minus", "reverse"}:
            return "-"
        raise ValueError(f"Unrecognized strand value: {value}")

    def _provenance(self) -> Provenance:
        tags = dict(self._config.tags)
        if self._config.record_kind:
            tags["record_kind"] = self._config.record_kind
        return Provenance(
            retrieved_at=datetime.now(timezone.utc),
            source_version=self._config.source_version,
            source_url=self._config.source_url,
            license=self._config.license,
            citation=self._config.citation,
            raw_artifact_paths=(self._relativize_path(self._path),),
            tags=tags,
        )

    def _site_id(self, header: str, sequence: str) -> str:
        raw = f"{header}|{sequence}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _relativize_path(self, path: Path) -> str:
        try:
            rel = path.relative_to(self._path.parent)
            return rel.as_posix()
        except ValueError:
            return str(path)
