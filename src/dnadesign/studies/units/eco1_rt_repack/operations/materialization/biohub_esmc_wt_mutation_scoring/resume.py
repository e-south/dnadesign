"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/resume.py

Resume helpers for Eco1 WT-only ESMC mutation scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from .constants import (
    POSITION_ENTROPY_FILE_NAME,
    SUBSTITUTION_LLR_FILE_NAME,
)

_REQUIRED_POSITION_COLUMNS = {
    "sequence_id",
    "sequence_hash",
    "model",
    "scoring_method_id",
    "biohub_request_hash",
    "biohub_query_hash",
    "canonical_position",
    "residue_index_zero_based",
    "wt_aa",
    "masked_sequence_hash",
    "token_count",
    "vocab_size",
    "logit_residue_offset",
    "entropy_bits",
    "canonical_entropy_bits",
    "wt_log_probability",
    "fraction_negative_alternate_llr",
    "best_alt_aa",
    "best_alt_llr",
    "worst_alt_aa",
    "worst_alt_llr",
    "raw_logits_response_hash",
    "retrieved_at",
    "status",
    "failure_reason",
}
_REQUIRED_SUBSTITUTION_COLUMNS = {
    "sequence_id",
    "sequence_hash",
    "model",
    "scoring_method_id",
    "biohub_request_hash",
    "biohub_query_hash",
    "canonical_position",
    "residue_index_zero_based",
    "wt_aa",
    "alt_aa",
    "masked_sequence_hash",
    "wt_log_probability",
    "alt_log_probability",
    "llr",
    "retrieved_at",
    "status",
}


@dataclass(frozen=True)
class CachedMutationScoringRows:
    """Accepted cached rows keyed by one-based position and query hash."""

    position_rows_by_key: dict[tuple[int, str], dict[str, object]]
    substitution_rows_by_key: dict[tuple[int, str], list[dict[str, object]]]

    @classmethod
    def empty(cls) -> "CachedMutationScoringRows":
        return cls(position_rows_by_key={}, substitution_rows_by_key={})

    def position_row(self, canonical_position: int, query_hash: str) -> dict[str, object] | None:
        return self.position_rows_by_key.get((canonical_position, query_hash))

    def substitution_rows(self, canonical_position: int, query_hash: str) -> list[dict[str, object]]:
        return [*self.substitution_rows_by_key.get((canonical_position, query_hash), [])]


def load_cached_rows(scoring_root: Path, *, request_hash: str) -> CachedMutationScoringRows:
    """Load accepted rows from a previous matching mutation-scoring request."""

    position_path = scoring_root / POSITION_ENTROPY_FILE_NAME
    substitution_path = scoring_root / SUBSTITUTION_LLR_FILE_NAME
    if not position_path.exists() or not substitution_path.exists():
        return CachedMutationScoringRows.empty()
    metadata = pq.read_metadata(position_path).metadata or {}
    if metadata.get(b"request_hash", b"").decode("utf-8") != request_hash:
        return CachedMutationScoringRows.empty()
    position_schema = pq.read_schema(position_path)
    substitution_schema = pq.read_schema(substitution_path)
    if not _REQUIRED_POSITION_COLUMNS.issubset(set(position_schema.names)):
        return CachedMutationScoringRows.empty()
    if not _REQUIRED_SUBSTITUTION_COLUMNS.issubset(set(substitution_schema.names)):
        return CachedMutationScoringRows.empty()
    position_rows = pq.read_table(position_path).to_pylist()
    substitution_rows = pq.read_table(substitution_path).to_pylist()
    positions_by_key: dict[tuple[int, str], dict[str, object]] = {}
    substitutions_by_key: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in position_rows:
        if row.get("status") != "accepted":
            continue
        if not _accepted_position_row_has_current_values(row):
            continue
        key = (int(row["canonical_position"]), str(row["biohub_query_hash"]))
        positions_by_key[key] = dict(row)
    for row in substitution_rows:
        key = (int(row["canonical_position"]), str(row["biohub_query_hash"]))
        substitutions_by_key.setdefault(key, []).append(dict(row))
    return CachedMutationScoringRows(
        position_rows_by_key=positions_by_key,
        substitution_rows_by_key=substitutions_by_key,
    )


def _accepted_position_row_has_current_values(row: dict[str, object]) -> bool:
    required_non_null = (
        "token_count",
        "vocab_size",
        "logit_residue_offset",
        "entropy_bits",
        "canonical_entropy_bits",
        "wt_log_probability",
        "fraction_negative_alternate_llr",
        "best_alt_aa",
        "best_alt_llr",
        "worst_alt_aa",
        "worst_alt_llr",
        "raw_logits_response_hash",
    )
    for column in required_non_null:
        value = row.get(column)
        if value is None or value == "":
            return False
    return True
