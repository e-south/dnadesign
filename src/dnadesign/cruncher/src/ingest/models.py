"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/ingest/models.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass(frozen=True, slots=True)
class OrganismRef:
    taxon: Optional[int] = None
    name: Optional[str] = None
    strain: Optional[str] = None
    assembly: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Provenance:
    retrieved_at: datetime
    source_version: Optional[str] = None
    source_url: Optional[str] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    raw_artifact_paths: Tuple[str, ...] = ()
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Checksums:
    sha256_raw: str
    sha256_norm: str


@dataclass(frozen=True, slots=True)
class MotifDescriptor:
    source: str
    motif_id: str
    tf_name: str
    organism: Optional[OrganismRef]
    alphabet: Literal["ACGT"] = "ACGT"
    length: int = 0
    kind: Literal["PFM", "PWM"] = "PFM"
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MotifRecord:
    descriptor: MotifDescriptor
    matrix: List[List[float]]
    matrix_semantics: Literal["probabilities", "weights"]
    background: Optional[Tuple[float, float, float, float]]
    provenance: Provenance
    checksums: Checksums


@dataclass(frozen=True, slots=True)
class GenomicInterval:
    contig: str
    start: int
    end: int
    assembly: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SiteInstance:
    source: str
    site_id: str
    motif_ref: str
    organism: Optional[OrganismRef]
    coordinate: Optional[GenomicInterval]
    sequence: Optional[str]
    strand: Optional[Literal["+", "-"]]
    score: Optional[float]
    evidence: Dict[str, Any]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class MotifQuery:
    source: Optional[str] = None
    organism: Optional[OrganismRef] = None
    tf_name: Optional[str] = None
    regex: Optional[str] = None
    collection: Optional[str] = None
    limit: Optional[int] = None


@dataclass(frozen=True, slots=True)
class SiteQuery:
    organism: Optional[OrganismRef] = None
    motif_id: Optional[str] = None
    tf_name: Optional[str] = None
    dataset_id: Optional[str] = None
    region: Optional[GenomicInterval] = None
    limit: Optional[int] = None


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    source: str
    dataset_id: str
    dataset_type: Optional[str] = None
    dataset_source: Optional[str] = None
    method: Optional[str] = None
    tf_names: Tuple[str, ...] = ()
    reference_genome: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetQuery:
    tf_name: Optional[str] = None
    dataset_type: Optional[str] = None
    dataset_source: Optional[str] = None
