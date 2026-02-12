"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/conftest.py

Shared helpers for baserender vNext tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def write_parquet(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def write_job(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def densegen_job_payload(
    *, parquet_path: Path, results_root: Path, outputs: list[dict], extra: dict | None = None
) -> dict:
    base = {
        "version": 3,
        "results_root": str(results_root),
        "input": {
            "kind": "parquet",
            "path": str(parquet_path),
            "adapter": {
                "kind": "densegen_tfbs",
                "columns": {
                    "sequence": "sequence",
                    "annotations": "densegen__used_tfbs_detail",
                    "id": "id",
                    "details": "details",
                },
                "policies": {},
            },
            "alphabet": "DNA",
        },
        "render": {
            "renderer": "sequence_rows",
            "style": {
                "preset": None,
                "overrides": {},
            },
        },
        "outputs": outputs,
        "run": {
            "strict": False,
            "fail_on_skips": False,
            "emit_report": False,
            "report_path": None,
        },
    }
    if extra:
        base.update(extra)
    return base
