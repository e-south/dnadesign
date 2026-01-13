"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/ingest/normalize.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

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
PROB_ROW_TOL = 1e-4


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def normalize_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(v) for v in row] for row in matrix]


def normalize_prob_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    norm = normalize_matrix(matrix)
    if not norm:
        raise ValueError("matrix is empty")
    for i, row in enumerate(norm):
        if len(row) != 4:
            raise ValueError(f"matrix row {i} length must be 4")
        if any(v < 0 for v in row):
            raise ValueError("matrix rows must be non-negative")
        s = sum(row)
        if s <= 0:
            raise ValueError(f"matrix row {i} must sum to > 0 (got {s:.6f})")
        if abs(s - 1.0) > PROB_ROW_TOL:
            raise ValueError(f"matrix row {i} must sum to 1.0 (got {s:.6f})")
        norm[i] = [v / s for v in row]
    return norm


def validate_prob_matrix(matrix: Sequence[Sequence[float]]) -> None:
    normalize_prob_matrix(matrix)


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
    pseudocounts: float = 0.5,
) -> List[List[float]] | tuple[List[List[float]], int]:
    if min_sites < 1:
        raise ValueError("min_sites must be >= 1")
    if pseudocounts < 0:
        raise ValueError("pseudocounts must be >= 0")
    length: Optional[int] = None
    site_count = 0
    sequences_clean: list[str] = []
    for seq in sequences:
        if not seq:
            continue
        if length is None:
            length = len(seq)
            if length == 0:
                raise ValueError("binding-site sequence is empty")
        if len(seq) != length:
            raise ValueError("binding-site sequences must have equal length")
        for i, ch in enumerate(seq):
            if ch not in "ACGT":
                raise ValueError("binding-site sequence contains non-ACGT characters")
        sequences_clean.append(seq)
        site_count += 1
    if site_count == 0:
        raise ValueError("no sequences available to compute PWM")
    if site_count < min_sites:
        msg = f"Only {site_count} binding-site sequences available (min_sites={min_sites})."
        if strict_min_sites:
            raise ValueError(msg)
        logger.warning("%s PWM may be unreliable.", msg)
    try:
        from Bio import motifs
        from Bio.Seq import Seq
    except ImportError as exc:
        raise ImportError(
            "Biopython is required to build PWMs from sites. "
            "Install with `uv add biopython` (already a project dependency)."
        ) from exc

    instances = [Seq(seq) for seq in sequences_clean]
    motif = motifs.create(instances)
    pwm = motif.counts.normalize(pseudocounts=pseudocounts)
    matrix = [[float(pwm[base][idx]) for base in "ACGT"] for idx in range(motif.length)]
    if return_count:
        return matrix, site_count
    return matrix


def build_motif_record(
    *,
    source: str,
    motif_id: str,
    tf_name: str,
    matrix: Sequence[Sequence[float]],
    log_odds_matrix: Optional[Sequence[Sequence[float]]] = None,
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
    background: Optional[Sequence[float]] = None,
) -> MotifRecord:
    if matrix_semantics == "probabilities":
        norm_matrix = normalize_prob_matrix(matrix)
    else:
        norm_matrix = normalize_matrix(matrix)
    norm_log_odds = None
    if log_odds_matrix is not None:
        norm_log_odds = normalize_matrix(log_odds_matrix)
        if len(norm_log_odds) != len(norm_matrix):
            raise ValueError("log_odds_matrix must match matrix length")
        for i, row in enumerate(norm_log_odds):
            if len(row) != 4:
                raise ValueError(f"log_odds_matrix row {i} length must be 4")
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
    norm_background = None
    if background is not None:
        if len(background) != 4:
            raise ValueError("background must have 4 values (A,C,G,T)")
        norm_background = tuple(float(v) for v in background)
    return MotifRecord(
        descriptor=descriptor,
        matrix=norm_matrix,
        log_odds_matrix=norm_log_odds,
        matrix_semantics=matrix_semantics,
        background=norm_background,
        provenance=provenance,
        checksums=checksums,
    )
