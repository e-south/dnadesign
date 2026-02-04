"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/ingest/sequence_provider.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Protocol

from dnadesign.cruncher.ingest.http_client import (
    HttpRetryPolicy,
    download_to,
    request_json,
)
from dnadesign.cruncher.ingest.models import GenomicInterval

_NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class SequenceProvider(Protocol):
    source_id: str

    def fetch(self, interval: GenomicInterval) -> str: ...

    def close(self) -> None: ...


@dataclass
class FastaSequenceProvider:
    fasta_path: Path
    assembly_id: str | None = None
    contig_aliases: Dict[str, str] = field(default_factory=dict)
    _index: object = field(init=False, repr=False)
    source_id: str = field(init=False, default="fasta")

    def __post_init__(self) -> None:
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_path}")
        try:
            import pysam
        except ImportError as exc:  # pragma: no cover - pysam is a hard dependency
            raise RuntimeError("pysam is required to load FASTA sequences.") from exc
        index_path = self.fasta_path.with_suffix(self.fasta_path.suffix + ".fai")
        if not index_path.exists():
            pysam.faidx(str(self.fasta_path))
        self._index = pysam.FastaFile(str(self.fasta_path))

    def fetch(self, interval: GenomicInterval) -> str:
        if interval.start < 0 or interval.end <= interval.start:
            raise ValueError("Genomic interval start/end are invalid.")
        if self.assembly_id and interval.assembly and interval.assembly != self.assembly_id:
            raise ValueError(
                f"Interval assembly '{interval.assembly}' does not match FASTA assembly '{self.assembly_id}'."
            )
        contig = interval.contig
        if contig not in self._index.references:
            alias = self.contig_aliases.get(contig)
            if alias and alias in self._index.references:
                contig = alias
            else:
                raise KeyError(f"Contig '{contig}' not found in FASTA index.")
        length = self._index.get_reference_length(contig)
        if interval.end > length:
            raise ValueError(f"Interval end {interval.end} exceeds contig '{contig}' length {length}.")
        seq = self._index.fetch(contig, interval.start, interval.end)
        return seq.upper()

    def close(self) -> None:
        close = getattr(self._index, "close", None)
        if callable(close):
            close()


@dataclass
class NCBISequenceProvider:
    cache_root: Path
    email: str | None = None
    tool: str = "cruncher"
    api_key: str | None = None
    timeout_seconds: int = 30
    retry_policy: HttpRetryPolicy = field(default_factory=HttpRetryPolicy)
    refresh: bool = False
    offline: bool = False
    contig_aliases: Dict[str, str] = field(default_factory=dict)
    _providers: Dict[str, FastaSequenceProvider] = field(init=False, default_factory=dict)
    source_id: str = field(init=False, default="ncbi")

    def fetch(self, interval: GenomicInterval) -> str:
        if not interval.assembly:
            raise ValueError(
                "Missing assembly/contig accession for NCBI hydration. "
                "Ensure the source provides referenceGenome/assemblyGenomeId or set ingest.genome_assembly."
            )
        accession = interval.assembly
        provider = self._providers.get(accession)
        if provider is None:
            fasta_path = self._ensure_fasta(accession)
            provider = FastaSequenceProvider(
                fasta_path=fasta_path,
                assembly_id=accession,
                contig_aliases=self.contig_aliases,
            )
            self._providers[accession] = provider
        return provider.fetch(interval)

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()

    def _ensure_fasta(self, accession: str) -> Path:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        accession_dir = self.cache_root / accession
        accession_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = accession_dir / f"{accession}.fna"
        manifest_path = accession_dir / "manifest.json"
        if fasta_path.exists() and not self.refresh:
            return fasta_path
        if self.offline:
            raise ValueError(f"Genome FASTA for '{accession}' not available offline.")
        if accession.startswith(("GCF_", "GCA_")):
            url = self._download_assembly_fasta(accession, fasta_path, accession_dir)
        else:
            url = self._download_nuccore_fasta(accession, fasta_path)
        checksum = _sha256_path(fasta_path)
        manifest = {
            "accession": accession,
            "source": "ncbi",
            "url": url,
            "sha256": checksum,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return fasta_path

    def _download_nuccore_fasta(self, accession: str, dest: Path) -> str:
        params = {
            "db": "nuccore",
            "id": accession,
            "rettype": "fasta",
            "retmode": "text",
        }
        url = self._eutils_url("efetch.fcgi", params)
        download_to(
            url,
            str(dest),
            timeout=self.timeout_seconds,
            retry=self.retry_policy,
        )
        return url

    def _download_assembly_fasta(self, accession: str, dest: Path, accession_dir: Path) -> str:
        uid = self._assembly_uid(accession)
        ftp_path = self._assembly_ftp_path(uid)
        base = ftp_path.rstrip("/").split("/")[-1]
        gz_name = f"{base}_genomic.fna.gz"
        url = f"{ftp_path}/{gz_name}"
        gz_path = accession_dir / gz_name
        download_to(
            url,
            str(gz_path),
            timeout=self.timeout_seconds,
            retry=self.retry_policy,
        )
        with gzip.open(gz_path, "rb") as gz, dest.open("wb") as out:
            shutil.copyfileobj(gz, out)
        return url

    def _assembly_uid(self, accession: str) -> str:
        params = {"db": "assembly", "term": accession, "retmode": "json"}
        data = request_json(
            self._eutils_url("esearch.fcgi", params),
            timeout=self.timeout_seconds,
            retry=self.retry_policy,
        )
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            raise ValueError(f"No NCBI assembly UID found for accession '{accession}'.")
        return ids[0]

    def _assembly_ftp_path(self, uid: str) -> str:
        params = {"db": "assembly", "id": uid, "retmode": "json"}
        data = request_json(
            self._eutils_url("esummary.fcgi", params),
            timeout=self.timeout_seconds,
            retry=self.retry_policy,
        )
        record = data.get("result", {}).get(uid, {})
        ftp_path = record.get("ftppath_refseq") or record.get("ftppath_genbank")
        if not ftp_path:
            raise ValueError(f"No FTP path found in NCBI assembly summary for UID '{uid}'.")
        return ftp_path

    def _eutils_url(self, endpoint: str, params: Dict[str, str]) -> str:
        full_params = dict(params)
        if self.tool:
            full_params["tool"] = self.tool
        if self.email:
            full_params["email"] = self.email
        if self.api_key:
            full_params["api_key"] = self.api_key
        query = urllib.parse.urlencode(full_params)
        return f"{_NCBI_EUTILS_BASE}/{endpoint}?{query}"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
