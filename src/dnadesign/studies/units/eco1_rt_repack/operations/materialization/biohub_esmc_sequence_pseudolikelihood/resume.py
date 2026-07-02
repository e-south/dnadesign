"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sequence_pseudolikelihood/resume.py

Resume helpers for Eco1 ESMC sequence pseudo-likelihood scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from .constants import POSITION_PLL_FILE_NAME

_REQUIRED_POSITION_COLUMNS = {
    "sequence_id",
    "sequence_hash",
    "model",
    "scoring_method_id",
    "biohub_request_hash",
    "biohub_query_hash",
    "canonical_position",
    "residue_index_zero_based",
    "residue",
    "masked_sequence_hash",
    "token_count",
    "vocab_size",
    "logit_residue_offset",
    "residue_log_probability",
    "raw_logits_response_hash",
    "retrieved_at",
    "status",
    "failure_reason",
}


@dataclass(frozen=True)
class CachedPseudolikelihoodRows:
    """Accepted cached PLL position rows keyed by sequence id, position, and query hash."""

    position_rows_by_key: dict[tuple[str, int, str], dict[str, object]]

    @classmethod
    def empty(cls) -> "CachedPseudolikelihoodRows":
        return cls(position_rows_by_key={})

    def position_row(self, sequence_id: str, canonical_position: int, query_hash: str) -> dict[str, object] | None:
        return self.position_rows_by_key.get((sequence_id, canonical_position, query_hash))


def load_cached_rows(scoring_root: Path, *, request_hash: str) -> CachedPseudolikelihoodRows:
    """Load accepted position PLL rows from a previous matching request."""

    position_path = scoring_root / POSITION_PLL_FILE_NAME
    if not position_path.exists():
        return CachedPseudolikelihoodRows.empty()
    metadata = pq.read_metadata(position_path).metadata or {}
    if metadata.get(b"request_hash", b"").decode("utf-8") != request_hash:
        raise ValueError("stale pseudo-likelihood cache: request_hash metadata does not match current request")
    schema = pq.read_schema(position_path)
    missing_columns = sorted(_REQUIRED_POSITION_COLUMNS - set(schema.names))
    if missing_columns:
        raise ValueError(
            "stale pseudo-likelihood cache: position table is missing required columns " + ", ".join(missing_columns)
        )
    rows_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in pq.read_table(position_path).to_pylist():
        if row.get("status") != "accepted":
            continue
        if not _accepted_position_row_has_current_values(row):
            raise ValueError("stale pseudo-likelihood cache: accepted rows must have current metric columns")
        key = (str(row["sequence_id"]), int(row["canonical_position"]), str(row["biohub_query_hash"]))
        rows_by_key[key] = dict(row)
    return CachedPseudolikelihoodRows(position_rows_by_key=rows_by_key)


def _accepted_position_row_has_current_values(row: dict[str, object]) -> bool:
    required_non_null = (
        "token_count",
        "vocab_size",
        "logit_residue_offset",
        "residue_log_probability",
        "raw_logits_response_hash",
    )
    for column in required_non_null:
        value = row.get(column)
        if value is None or value == "":
            return False
    return True
