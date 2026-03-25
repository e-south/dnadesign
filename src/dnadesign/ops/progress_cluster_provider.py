"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/progress_cluster_provider.py

Provider-owned cluster status builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq

from .progress_support import required_path


def provide_cluster_run_index_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    del repo_root
    return cluster_run_index_progress(inputs.get("cluster_results_root"))


def cluster_run_index_progress(cluster_results_root: object) -> tuple[str, str, dict[str, object]]:
    resolved_root = required_path(
        cluster_results_root,
        flag_name="--cluster-results-root",
        progress_kind="cluster-run-index",
    )
    index_path = resolved_root / "index.parquet"
    if not index_path.exists():
        return (
            "missing",
            "cluster run index not found",
            {"cluster_results_root": str(resolved_root), "index_path": str(index_path)},
        )

    parquet_file = pq.ParquetFile(str(index_path))
    entry_count = int(parquet_file.metadata.num_rows)
    if entry_count == 0:
        return (
            "attention",
            "cluster run index is present but empty",
            {
                "cluster_results_root": str(resolved_root),
                "index_path": str(index_path),
                "entry_count": 0,
            },
        )

    table = parquet_file.read(columns=["kind", "run_slug", "created_utc", "status", "alias"])
    kind_values = [str(value or "unknown") for value in table.column("kind").to_pylist()]
    status_values = [str(value or "unknown") for value in table.column("status").to_pylist()]
    created_values = [str(value or "") for value in table.column("created_utc").to_pylist()]
    slug_values = [str(value or "<unknown>") for value in table.column("run_slug").to_pylist()]
    alias_values = table.column("alias").to_pylist()

    kind_counts = Counter(kind_values)
    status_counts = Counter(status_values)
    latest_index = max(range(entry_count), key=lambda index: (created_values[index], slug_values[index]))
    latest_kind = kind_values[latest_index]
    latest_slug = slug_values[latest_index]
    latest_status = status_values[latest_index]
    summary = f"{entry_count} cluster run-index entries; latest {latest_kind} {latest_slug} is {latest_status}"
    all_complete = set(status_counts.keys()) <= {"complete"}
    return (
        "ok" if all_complete else "attention",
        summary,
        {
            "cluster_results_root": str(resolved_root),
            "index_path": str(index_path),
            "entry_count": entry_count,
            "kind_counts": dict(sorted(kind_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "latest_entry": {
                "kind": latest_kind,
                "run_slug": latest_slug,
                "status": latest_status,
                "created_utc": created_values[latest_index] or None,
                "alias": alias_values[latest_index],
            },
        },
    )


__all__ = [
    "cluster_run_index_progress",
    "provide_cluster_run_index_status",
]
