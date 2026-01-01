"""Normalization helpers for motifs and binding sites."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

from dnadesign.cruncher.ingest.models import (
    Checksums,
    MotifDescriptor,
    MotifRecord,
    OrganismRef,
    Provenance,
)

logger = logging.getLogger(__name__)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def normalize_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(v) for v in row] for row in matrix]


def validate_prob_matrix(matrix: Sequence[Sequence[float]]) -> None:
    if not matrix:
        raise ValueError("matrix is empty")
    width = len(matrix)
    for i, row in enumerate(matrix):
        if len(row) != 4:
            raise ValueError(f"matrix row {i} length must be 4")
        s = sum(row)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"matrix row {i} must sum to 1.0 (got {s:.6f})")
        if any(v < 0 for v in row):
            raise ValueError("matrix rows must be non-negative")
    if width < 1:
        raise ValueError("matrix must have at least 1 row")


def normalize_site_sequence(seq: str, uppercase_only: bool) -> str:
    if not seq:
        raise ValueError("binding-site sequence is empty")
    has_upper = any(ch.isupper() for ch in seq)
    has_lower = any(ch.islower() for ch in seq)
    if uppercase_only and has_upper and has_lower:
        seq = "".join(ch for ch in seq if ch.isupper())
    seq = seq.strip().upper()
    if not seq:
        raise ValueError("binding-site sequence is empty after normalization")
    if any(ch not in "ACGT" for ch in seq):
        raise ValueError("binding-site sequence contains non-ACGT characters")
    return seq


def compute_pwm_from_sites(
    sequences: Iterable[str],
    *,
    min_sites: int = 2,
    return_count: bool = False,
    strict_min_sites: bool = True,
) -> List[List[float]] | tuple[List[List[float]], int]:
    if min_sites < 1:
        raise ValueError("min_sites must be >= 1")
    length: Optional[int] = None
    counts: List[List[int]] = []
    site_count = 0
    for seq in sequences:
        if not seq:
            continue
        if length is None:
            length = len(seq)
            if length == 0:
                raise ValueError("binding-site sequence is empty")
            counts = [[0, 0, 0, 0] for _ in range(length)]
        if len(seq) != length:
            raise ValueError("binding-site sequences must have equal length")
        for i, ch in enumerate(seq):
            if ch not in "ACGT":
                raise ValueError("binding-site sequence contains non-ACGT characters")
            if ch == "A":
                counts[i][0] += 1
            elif ch == "C":
                counts[i][1] += 1
            elif ch == "G":
                counts[i][2] += 1
            else:
                counts[i][3] += 1
        site_count += 1
    if site_count == 0:
        raise ValueError("no sequences available to compute PWM")
    if site_count < min_sites:
        msg = f"Only {site_count} binding-site sequences available (min_sites={min_sites})."
        if strict_min_sites:
            raise ValueError(msg)
        logger.warning("%s PWM may be unreliable.", msg)
    matrix: List[List[float]] = []
    for row in counts:
        total = sum(row)
        if total == 0:
            raise ValueError("binding-site column has no A/C/G/T bases")
        matrix.append([row[0] / total, row[1] / total, row[2] / total, row[3] / total])
    if return_count:
        return matrix, site_count
    return matrix


def build_motif_record(
    *,
    source: str,
    motif_id: str,
    tf_name: str,
    matrix: Sequence[Sequence[float]],
    matrix_semantics: str,
    organism: Optional[OrganismRef],
    raw_payload: str,
    retrieved_at: Optional[datetime] = None,
    source_version: Optional[str] = None,
    source_url: Optional[str] = None,
    license: Optional[str] = None,
    citation: Optional[str] = None,
    raw_artifact_paths: Optional[Iterable[str]] = None,
    tags: Optional[dict[str, str]] = None,
) -> MotifRecord:
    norm_matrix = normalize_matrix(matrix)
    if matrix_semantics == "probabilities":
        validate_prob_matrix(norm_matrix)
    descriptor = MotifDescriptor(
        source=source,
        motif_id=motif_id,
        tf_name=tf_name,
        organism=organism,
        length=len(norm_matrix),
        kind="PFM" if matrix_semantics == "probabilities" else "PWM",
        tags=tags or {},
    )
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    raw_paths = tuple(raw_artifact_paths or ())
    provenance = Provenance(
        retrieved_at=retrieved_at,
        source_version=source_version,
        source_url=source_url,
        license=license,
        citation=citation,
        raw_artifact_paths=raw_paths,
    )
    checksum_raw = _sha256_text(raw_payload)
    checksum_norm = _sha256_text(str(norm_matrix))
    checksums = Checksums(sha256_raw=checksum_raw, sha256_norm=checksum_norm)
    return MotifRecord(
        descriptor=descriptor,
        matrix=norm_matrix,
        matrix_semantics=matrix_semantics,
        background=None,
        provenance=provenance,
        checksums=checksums,
    )
