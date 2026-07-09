"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/candidate_pool.py

Aggregate Eco1 RT v3 policy-specific ProteinMPNN candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    write_yaml,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    DEFAULT_CREATED_AT,
    DEFAULT_GENERATION_POLICIES_ROOT,
    GENERATION_POLICY_VERSION,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.manifest_io import (
    load_valid_generation_policy_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.models import (
    MaterializedGenerationPolicyCandidatePool,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri

_CANDIDATE_TABLE_NAME = "candidate_table.parquet"
_OUTPUT_TABLE_NAME = "candidate_pool.parquet"
_OUTPUT_MANIFEST_NAME = "candidate_pool_manifest.yaml"
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.candidate_pool"


def materialize_generation_policy_candidate_pool(
    *,
    repo_root: Path | None = None,
    generation_policy_root: Path | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedGenerationPolicyCandidatePool:
    """Aggregate accepted v3 candidate tables into one deduplicated pool."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    policy_root = _resolve_path(root, generation_policy_root or DEFAULT_GENERATION_POLICIES_ROOT)
    policy_manifest_path = policy_root / "generation_policy_manifest.yaml"
    policy_manifest = load_valid_generation_policy_manifest(policy_manifest_path)
    policy_ids = [str(policy["policy_id"]) for policy in policy_manifest["generation_policies"]]

    input_tables = _load_policy_candidate_tables(policy_root=policy_root, policy_ids=policy_ids)
    pool_rows, duplicate_count = _deduplicate_rows(input_tables=input_tables, policy_ids=policy_ids)
    if not pool_rows:
        raise ValueError("generation-policy candidate pool requires accepted candidate rows")

    candidate_pool_path = policy_root / _OUTPUT_TABLE_NAME
    pq.write_table(pa.Table.from_pylist(pool_rows), candidate_pool_path)
    manifest = _build_manifest(
        created_at=created_at,
        policy_root=policy_root,
        policy_manifest_path=policy_manifest_path,
        policy_manifest=policy_manifest,
        input_tables=input_tables,
        candidate_pool_path=candidate_pool_path,
        pool_rows=pool_rows,
        duplicate_count=duplicate_count,
    )
    manifest_path = policy_root / _OUTPUT_MANIFEST_NAME
    write_yaml(manifest_path, manifest)
    return MaterializedGenerationPolicyCandidatePool(
        policy_manifest_path=policy_manifest_path,
        candidate_pool_path=candidate_pool_path,
        manifest_path=manifest_path,
    )


def _load_policy_candidate_tables(
    *,
    policy_root: Path,
    policy_ids: list[str],
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for policy_index, policy_id in enumerate(policy_ids):
        table_path = policy_root / policy_id / _CANDIDATE_TABLE_NAME
        if not table_path.exists():
            raise FileNotFoundError(table_path)
        rows = pq.read_table(table_path).to_pylist()
        accepted_rows = [row for row in rows if str(row.get("status")) == "accepted"]
        inputs.append(
            {
                "policy_id": policy_id,
                "policy_index": policy_index,
                "candidate_table_path": table_path,
                "row_count": len(rows),
                "accepted_row_count": len(accepted_rows),
                "rows": accepted_rows,
            }
        )
    return inputs


def _deduplicate_rows(
    *,
    input_tables: list[dict[str, Any]],
    policy_ids: list[str],
) -> tuple[list[dict[str, Any]], int]:
    rows_by_hash: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    policy_index_by_id = {policy_id: index for index, policy_id in enumerate(policy_ids)}
    for table in input_tables:
        policy_id = str(table["policy_id"])
        policy_index = int(table["policy_index"])
        for row in table["rows"]:
            sequence_hash = _require_text(row.get("sequence_hash"), "sequence_hash")
            enriched = dict(row)
            enriched["_policy_id"] = policy_id
            enriched["_policy_index"] = policy_index
            enriched["_candidate_table_path"] = str(table["candidate_table_path"])
            rows_by_hash[sequence_hash].append(enriched)

    pooled_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    for sequence_hash, duplicate_rows in rows_by_hash.items():
        duplicate_rows.sort(
            key=lambda row: (
                int(row["_policy_index"]),
                int(row.get("rank") or 10**9),
                str(row.get("candidate_id")),
            )
        )
        selected = duplicate_rows[0]
        duplicate_count += max(0, len(duplicate_rows) - 1)
        source_policy_ids = sorted({str(row["_policy_id"]) for row in duplicate_rows}, key=policy_index_by_id.get)
        source_candidate_ids = sorted(str(row["candidate_id"]) for row in duplicate_rows)
        source_candidate_tables = sorted({str(row["_candidate_table_path"]) for row in duplicate_rows})
        pooled = {
            key: value
            for key, value in selected.items()
            if key not in {"_policy_id", "_policy_index", "_candidate_table_path"}
        }
        pooled.update(
            {
                "policy_id": str(selected["_policy_id"]),
                "policy_version": GENERATION_POLICY_VERSION,
                "primary_policy_id": str(selected["_policy_id"]),
                "source_policy_ids": source_policy_ids,
                "source_candidate_ids": source_candidate_ids,
                "source_candidate_tables": source_candidate_tables,
                "source_policy_count": len(source_policy_ids),
                "sequence_hash": sequence_hash,
            }
        )
        pooled_rows.append(pooled)

    pooled_rows.sort(key=lambda row: (int(row.get("rank") or 10**9), str(row["candidate_id"])))
    for rank, row in enumerate(pooled_rows, start=1):
        row["rank"] = rank
    return pooled_rows, duplicate_count


def _build_manifest(
    *,
    created_at: str,
    policy_root: Path,
    policy_manifest_path: Path,
    policy_manifest: Mapping[str, Any],
    input_tables: list[dict[str, Any]],
    candidate_pool_path: Path,
    pool_rows: list[dict[str, Any]],
    duplicate_count: int,
) -> dict[str, Any]:
    rows_by_policy = {
        str(table["policy_id"]): [row for row in pool_rows if str(row["primary_policy_id"]) == str(table["policy_id"])]
        for table in input_tables
    }
    return {
        "schema_id": "eco1_rt.generation_policy_candidate_pool_manifest",
        "schema_version": 1,
        "status": "materialized",
        "created_by": _CREATED_BY,
        "created_at": created_at,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "policy_manifest_hash": policy_manifest["policy_manifest_hash"],
        "generation_policy_manifest_path": str(policy_manifest_path),
        "candidate_pool_path": str(candidate_pool_path),
        "input_candidate_tables": [
            {
                "policy_id": str(table["policy_id"]),
                "candidate_table_path": str(table["candidate_table_path"]),
                "candidate_table_hash": sha256_uri(table["candidate_table_path"]),
                "row_count": int(table["row_count"]),
                "accepted_row_count": int(table["accepted_row_count"]),
                "deduplicated_primary_row_count": len(rows_by_policy[str(table["policy_id"])]),
            }
            for table in input_tables
        ],
        "candidate_pool_row_count": len(pool_rows),
        "duplicate_sequence_count": duplicate_count,
        "candidate_pool_hash": sha256_uri(candidate_pool_path),
        "policy_root": str(policy_root),
    }


def _resolve_path(repo_root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    return resolved if resolved.is_absolute() else (repo_root / resolved).resolve()


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
