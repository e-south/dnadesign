"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/scoring/esmc_masked_marginal/tables.py

Parquet writers and validators for ESMC masked-marginal DMS artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.permuter.src.scoring.esmc_masked_marginal.contracts import MaskedMarginalArtifacts

POSITION_ENTROPY_SCHEMA_ID = "permuter.esmc_masked_marginal.position_entropy"
SUBSTITUTION_LLR_SCHEMA_ID = "permuter.esmc_masked_marginal.substitution_llr"
DEFAULT_POSITION_ENTROPY_FILE_NAME = "position_entropy.parquet"
DEFAULT_SUBSTITUTION_LLR_FILE_NAME = "substitution_llr.parquet"
DEFAULT_MANIFEST_FILE_NAME = "mutation_scoring_manifest.yaml"

_POSITION_SCHEMA = pa.schema(
    [
        ("sequence_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("model", pa.string()),
        ("scoring_method_id", pa.string()),
        ("biohub_request_hash", pa.string()),
        ("biohub_query_hash", pa.string()),
        ("canonical_position", pa.int32()),
        ("residue_index_zero_based", pa.int32()),
        ("wt_aa", pa.string()),
        ("masked_sequence_hash", pa.string()),
        ("token_count", pa.int32()),
        ("vocab_size", pa.int32()),
        ("logit_residue_offset", pa.int32()),
        ("entropy_bits", pa.float64()),
        ("canonical_entropy_bits", pa.float64()),
        ("wt_log_probability", pa.float64()),
        ("fraction_negative_alternate_llr", pa.float64()),
        ("best_alt_aa", pa.string()),
        ("best_alt_llr", pa.float64()),
        ("worst_alt_aa", pa.string()),
        ("worst_alt_llr", pa.float64()),
        ("raw_logits_response_hash", pa.string()),
        ("retrieved_at", pa.string()),
        ("status", pa.string()),
        ("failure_reason", pa.string()),
    ]
)
_SUBSTITUTION_SCHEMA = pa.schema(
    [
        ("sequence_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("model", pa.string()),
        ("scoring_method_id", pa.string()),
        ("biohub_request_hash", pa.string()),
        ("biohub_query_hash", pa.string()),
        ("canonical_position", pa.int32()),
        ("residue_index_zero_based", pa.int32()),
        ("wt_aa", pa.string()),
        ("alt_aa", pa.string()),
        ("masked_sequence_hash", pa.string()),
        ("wt_log_probability", pa.float64()),
        ("alt_log_probability", pa.float64()),
        ("llr", pa.float64()),
        ("retrieved_at", pa.string()),
        ("status", pa.string()),
    ]
)


def write_masked_marginal_artifacts(
    *,
    output_root: Path,
    position_rows: Sequence[Mapping[str, Any]],
    substitution_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    request_hash: str,
    position_file_name: str = DEFAULT_POSITION_ENTROPY_FILE_NAME,
    substitution_file_name: str = DEFAULT_SUBSTITUTION_LLR_FILE_NAME,
    manifest_file_name: str = DEFAULT_MANIFEST_FILE_NAME,
) -> MaskedMarginalArtifacts:
    """Write masked-marginal position and substitution tables."""

    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = MaskedMarginalArtifacts(
        position_entropy_path=output_root / position_file_name,
        substitution_llr_path=output_root / substitution_file_name,
        manifest_path=output_root / manifest_file_name,
    )
    _write_table(
        artifacts.position_entropy_path,
        position_rows,
        schema=_POSITION_SCHEMA,
        schema_id=POSITION_ENTROPY_SCHEMA_ID,
        request_hash=request_hash,
    )
    _write_table(
        artifacts.substitution_llr_path,
        substitution_rows,
        schema=_SUBSTITUTION_SCHEMA,
        schema_id=SUBSTITUTION_LLR_SCHEMA_ID,
        request_hash=request_hash,
    )
    artifacts.manifest_path.write_text(yaml.safe_dump(dict(manifest), sort_keys=False), encoding="utf-8")
    return artifacts


def validate_masked_marginal_artifacts(
    *,
    artifacts: MaskedMarginalArtifacts,
    expected_position_count: int,
    request_hash: str,
) -> list[str]:
    """Return validation issue strings for masked-marginal artifacts."""

    issues: list[str] = []
    for path, schema_id, schema in (
        (artifacts.position_entropy_path, POSITION_ENTROPY_SCHEMA_ID, _POSITION_SCHEMA),
        (artifacts.substitution_llr_path, SUBSTITUTION_LLR_SCHEMA_ID, _SUBSTITUTION_SCHEMA),
    ):
        if not path.exists():
            issues.append(f"{path}: missing")
            continue
        metadata = pq.read_metadata(path).metadata or {}
        observed_schema_id = metadata.get(b"schema_id", b"").decode("utf-8")
        observed_request_hash = metadata.get(b"request_hash", b"").decode("utf-8")
        if observed_schema_id != schema_id:
            issues.append(f"{path}: schema_id {observed_schema_id!r} != {schema_id!r}")
        if observed_request_hash != request_hash:
            issues.append(f"{path}: request_hash {observed_request_hash!r} != {request_hash!r}")
        table = pq.read_table(path)
        missing_columns = sorted(set(schema.names) - set(table.column_names))
        if missing_columns:
            issues.append(f"{path}: missing columns {missing_columns}")
    if issues:
        return issues
    position_rows = pq.read_table(artifacts.position_entropy_path).to_pylist()
    substitution_rows = pq.read_table(artifacts.substitution_llr_path).to_pylist()
    if len(position_rows) != expected_position_count:
        issues.append(f"position row count {len(position_rows)} != {expected_position_count}")
    accepted_positions = [row for row in position_rows if row.get("status") == "accepted"]
    expected_substitution_count = len(accepted_positions) * 19
    if len(substitution_rows) != expected_substitution_count:
        issues.append(f"substitution row count {len(substitution_rows)} != {expected_substitution_count}")
    if not artifacts.manifest_path.exists():
        issues.append(f"{artifacts.manifest_path}: missing")
    return issues


def _write_table(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: pa.Schema,
    schema_id: str,
    request_hash: str,
) -> None:
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_id"] = schema_id.encode("utf-8")
    metadata[b"request_hash"] = request_hash.encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)
