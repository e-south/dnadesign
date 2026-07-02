"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/scoring/esmc_pseudolikelihood/tables.py

Parquet writers and validators for ESMC pseudo-likelihood artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.permuter.src.scoring.esmc_pseudolikelihood.contracts import EsmcPseudolikelihoodArtifacts

POSITION_PLL_SCHEMA_ID = "permuter.esmc_pseudolikelihood.position_pll"
SEQUENCE_PLL_SCHEMA_ID = "permuter.esmc_pseudolikelihood.sequence_pll"
DEFAULT_POSITION_PLL_FILE_NAME = "position_pll.parquet"
DEFAULT_SEQUENCE_PLL_FILE_NAME = "sequence_pll.parquet"
DEFAULT_MANIFEST_FILE_NAME = "pseudolikelihood_manifest.yaml"

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
        ("residue", pa.string()),
        ("masked_sequence_hash", pa.string()),
        ("token_count", pa.int32()),
        ("vocab_size", pa.int32()),
        ("logit_residue_offset", pa.int32()),
        ("residue_log_probability", pa.float64()),
        ("raw_logits_response_hash", pa.string()),
        ("retrieved_at", pa.string()),
        ("status", pa.string()),
        ("failure_reason", pa.string()),
    ]
)
_SEQUENCE_SCHEMA = pa.schema(
    [
        ("sequence_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("model", pa.string()),
        ("scoring_method_id", pa.string()),
        ("biohub_request_hash", pa.string()),
        ("sequence_length", pa.int32()),
        ("accepted_position_count", pa.int32()),
        ("errored_position_count", pa.int32()),
        ("pll_total", pa.float64()),
        ("pll_mean_per_residue", pa.float64()),
        ("pseudo_perplexity", pa.float64()),
        ("delta_pll_total_vs_wt", pa.float64()),
        ("delta_pll_mean_vs_wt", pa.float64()),
        ("status", pa.string()),
        ("failure_reason", pa.string()),
    ]
)


def build_sequence_pseudolikelihood_rows(
    *,
    position_rows: Sequence[Mapping[str, Any]],
    expected_lengths_by_sequence_id: Mapping[str, int],
    wt_sequence_id: str,
) -> list[dict[str, object]]:
    """Summarize accepted position PLL rows into one row per sequence."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in position_rows:
        grouped.setdefault(str(row.get("sequence_id") or ""), []).append(row)
    sequence_rows = [
        _sequence_row(
            sequence_id=sequence_id,
            rows=rows,
            expected_length=int(expected_lengths_by_sequence_id[sequence_id]),
        )
        for sequence_id, rows in grouped.items()
        if sequence_id
    ]
    wt_rows = [row for row in sequence_rows if row["sequence_id"] == wt_sequence_id]
    if len(wt_rows) != 1:
        raise ValueError(f"Expected exactly one WT pseudo-likelihood sequence row for {wt_sequence_id!r}")
    if wt_rows[0].get("status") != "accepted":
        return sequence_rows
    wt_total = _accepted_float(wt_rows[0].get("pll_total"))
    wt_mean = _accepted_float(wt_rows[0].get("pll_mean_per_residue"))
    for row in sequence_rows:
        if row["status"] != "accepted":
            row["delta_pll_total_vs_wt"] = None
            row["delta_pll_mean_vs_wt"] = None
            continue
        total = _accepted_float(row.get("pll_total"))
        mean = _accepted_float(row.get("pll_mean_per_residue"))
        row["delta_pll_total_vs_wt"] = float(total - wt_total)
        row["delta_pll_mean_vs_wt"] = float(mean - wt_mean)
    return sorted(sequence_rows, key=lambda row: (row["sequence_id"] != wt_sequence_id, str(row["sequence_id"])))


def write_pseudolikelihood_artifacts(
    *,
    output_root: Path,
    position_rows: Sequence[Mapping[str, Any]],
    sequence_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    request_hash: str,
    position_file_name: str = DEFAULT_POSITION_PLL_FILE_NAME,
    sequence_file_name: str = DEFAULT_SEQUENCE_PLL_FILE_NAME,
    manifest_file_name: str = DEFAULT_MANIFEST_FILE_NAME,
) -> EsmcPseudolikelihoodArtifacts:
    """Write pseudo-likelihood position, sequence, and manifest artifacts."""

    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = EsmcPseudolikelihoodArtifacts(
        position_pll_path=output_root / position_file_name,
        sequence_pll_path=output_root / sequence_file_name,
        manifest_path=output_root / manifest_file_name,
    )
    _write_table(
        artifacts.position_pll_path,
        position_rows,
        schema=_POSITION_SCHEMA,
        schema_id=POSITION_PLL_SCHEMA_ID,
        request_hash=request_hash,
    )
    _write_table(
        artifacts.sequence_pll_path,
        sequence_rows,
        schema=_SEQUENCE_SCHEMA,
        schema_id=SEQUENCE_PLL_SCHEMA_ID,
        request_hash=request_hash,
    )
    artifacts.manifest_path.write_text(yaml.safe_dump(dict(manifest), sort_keys=False), encoding="utf-8")
    return artifacts


def validate_pseudolikelihood_artifacts(
    *,
    artifacts: EsmcPseudolikelihoodArtifacts,
    expected_sequence_count: int,
    request_hash: str,
) -> list[str]:
    """Return validation issue strings for pseudo-likelihood artifacts."""

    issues: list[str] = []
    for path, schema_id, schema in (
        (artifacts.position_pll_path, POSITION_PLL_SCHEMA_ID, _POSITION_SCHEMA),
        (artifacts.sequence_pll_path, SEQUENCE_PLL_SCHEMA_ID, _SEQUENCE_SCHEMA),
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
    sequence_rows = pq.read_table(artifacts.sequence_pll_path).to_pylist()
    if len(sequence_rows) != expected_sequence_count:
        issues.append(f"sequence row count {len(sequence_rows)} != {expected_sequence_count}")
    if not artifacts.manifest_path.exists():
        issues.append(f"{artifacts.manifest_path}: missing")
    return issues


def _sequence_row(*, sequence_id: str, rows: list[Mapping[str, Any]], expected_length: int) -> dict[str, object]:
    accepted_rows = [row for row in rows if row.get("status") == "accepted"]
    errored_count = len(rows) - len(accepted_rows)
    if len(rows) != expected_length:
        errored_count += expected_length - len(rows)
    if errored_count or len(accepted_rows) != expected_length:
        return {
            "sequence_id": sequence_id,
            "sequence_hash": str(rows[0].get("sequence_hash") or "") if rows else "",
            "model": str(rows[0].get("model") or "") if rows else "",
            "scoring_method_id": str(rows[0].get("scoring_method_id") or "") if rows else "",
            "biohub_request_hash": str(rows[0].get("biohub_request_hash") or "") if rows else "",
            "sequence_length": expected_length,
            "accepted_position_count": len(accepted_rows),
            "errored_position_count": max(errored_count, 1),
            "pll_total": None,
            "pll_mean_per_residue": None,
            "pseudo_perplexity": None,
            "delta_pll_total_vs_wt": None,
            "delta_pll_mean_vs_wt": None,
            "status": "partial",
            "failure_reason": "sequence_has_missing_or_errored_position_rows",
        }
    pll_total = float(sum(float(row["residue_log_probability"]) for row in accepted_rows))
    pll_mean = pll_total / expected_length
    return {
        "sequence_id": sequence_id,
        "sequence_hash": str(accepted_rows[0]["sequence_hash"]),
        "model": str(accepted_rows[0]["model"]),
        "scoring_method_id": str(accepted_rows[0]["scoring_method_id"]),
        "biohub_request_hash": str(accepted_rows[0]["biohub_request_hash"]),
        "sequence_length": expected_length,
        "accepted_position_count": len(accepted_rows),
        "errored_position_count": 0,
        "pll_total": pll_total,
        "pll_mean_per_residue": pll_mean,
        "pseudo_perplexity": float(math.exp(-pll_mean)),
        "delta_pll_total_vs_wt": None,
        "delta_pll_mean_vs_wt": None,
        "status": "accepted",
        "failure_reason": "",
    }


def _accepted_float(value: object) -> float:
    if value is None:
        raise ValueError("WT pseudo-likelihood row must be accepted before deltas can be computed")
    return float(value)


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
