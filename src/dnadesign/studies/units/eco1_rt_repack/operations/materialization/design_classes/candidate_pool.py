"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/candidate_pool.py

Candidate-pool aggregation for Eco1 RT design classes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    load_yaml,
    write_yaml,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    BASELINE_CLASS_ID,
    CANDIDATE_POOL_FILE_NAME,
    CANDIDATE_POOL_MANIFEST_FILE_NAME,
    DEFAULT_DESIGN_CLASSES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.models import (
    MaterializedDesignClassCandidatePool,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri


def materialize_design_class_candidate_pool(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    baseline_candidate_table_path: Path | None = None,
) -> MaterializedDesignClassCandidatePool:
    """Write a nonredundant accepted-candidate pool across available design classes."""

    root = repo_root.expanduser().resolve()
    class_root = _resolve(root, output_root or DEFAULT_DESIGN_CLASSES_ROOT)
    source_root = _resolve(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    manifest_path = class_root / "design_class_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _load_yaml(manifest_path)
    records, availability = _available_candidate_records(
        manifest=manifest,
        source_root=source_root,
        baseline_candidate_table_path=baseline_candidate_table_path,
    )
    if not records:
        raise ValueError("No accepted candidate rows are available for design-class candidate-pool materialization")
    pooled = _deduplicate_records(records)
    pool_path = class_root / CANDIDATE_POOL_FILE_NAME
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"schema_id": b"eco1_rt.design_class_candidate_pool",
        b"schema_version": b"1",
        b"status": b"materialized",
    }
    table = pa.Table.from_pylist(pooled)
    pq.write_table(table.replace_schema_metadata(metadata), pool_path)
    pool_manifest_path = class_root / CANDIDATE_POOL_MANIFEST_FILE_NAME
    pool_manifest = {
        "schema_id": "eco1_rt.design_class_candidate_pool_manifest",
        "schema_version": 1,
        "status": "materialized",
        "candidate_pool_path": str(pool_path),
        "candidate_pool_hash": sha256_uri(pool_path),
        "source_candidate_table_count": len({record["source_candidate_table"] for record in records}),
        "input_candidate_row_count": len(records),
        "nonredundant_candidate_count": len(pooled),
        "duplicate_sequence_count": sum(1 for row in pooled if int(row["duplicate_candidate_count"]) > 1),
        "deduplication_key": "sequence_hash",
        "baseline_priority": BASELINE_CLASS_ID,
        "baseline_candidate_table_included": availability["baseline_candidate_table_included"],
        "generated_candidate_table_count": len(availability["generated_design_class_ids"]),
        "generated_design_class_ids": availability["generated_design_class_ids"],
        "pending_design_class_ids": availability["pending_design_class_ids"],
        "source_candidate_table_hashes": {
            path: sha256_uri(Path(path)) for path in sorted({record["source_candidate_table"] for record in records})
        },
    }
    write_yaml(pool_manifest_path, pool_manifest)
    return MaterializedDesignClassCandidatePool(candidate_pool_path=pool_path, manifest_path=pool_manifest_path)


def _available_candidate_records(
    *,
    manifest: dict[str, Any],
    source_root: Path,
    baseline_candidate_table_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    baseline_included = False
    generated_included: list[str] = []
    pending: list[str] = []
    class_rows = manifest.get("design_classes")
    if not isinstance(class_rows, list):
        raise ValueError("design_class_manifest.yaml must contain design_classes")
    for priority, class_row in enumerate(class_rows):
        design_class_id = str(class_row["design_class_id"])
        if design_class_id == BASELINE_CLASS_ID:
            table_path = baseline_candidate_table_path or source_root / "candidate_table.parquet"
        else:
            table_path = Path(str(class_row["class_root"])) / "candidate_table.parquet"
        if not table_path.exists():
            if design_class_id != BASELINE_CLASS_ID:
                pending.append(design_class_id)
            continue
        if design_class_id == BASELINE_CLASS_ID:
            baseline_included = True
        else:
            generated_included.append(design_class_id)
        for row in pq.read_table(table_path).to_pylist():
            if str(row.get("status")) != "accepted":
                continue
            record = dict(row)
            record["design_class_id"] = design_class_id
            record["mask_policy_id"] = design_class_id
            record["class_priority"] = priority
            record["source_candidate_id"] = str(row["candidate_id"])
            record["source_candidate_table"] = str(table_path)
            records.append(record)
    return records, {
        "baseline_candidate_table_included": baseline_included,
        "generated_design_class_ids": generated_included,
        "pending_design_class_ids": pending,
    }


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sequence_hash"])].append(record)
    pooled: list[dict[str, Any]] = []
    for sequence_hash, duplicate_rows in grouped.items():
        duplicate_rows.sort(key=lambda row: (int(row["class_priority"]), int(row["rank"]), str(row["candidate_id"])))
        chosen = dict(duplicate_rows[0])
        chosen["duplicate_design_class_ids"] = [str(row["design_class_id"]) for row in duplicate_rows]
        chosen["duplicate_candidate_ids"] = [str(row["source_candidate_id"]) for row in duplicate_rows]
        chosen["duplicate_candidate_count"] = len(duplicate_rows)
        chosen["sequence_hash"] = sequence_hash
        pooled.append(chosen)
    pooled.sort(key=lambda row: (int(row["class_priority"]), int(row["rank"]), str(row["candidate_id"])))
    for rank, row in enumerate(pooled, start=1):
        row["rank"] = rank
    return pooled


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml(path)
