"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_streaming_pipeline.py

Streaming orchestration tests for short-circuit row consumption and record counts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.baserender.src.api import run_job_v3

from .conftest import write_job, write_parquet


def _generic_job_payload(*, parquet_path: Path, results_root: Path, limit: int) -> dict:
    return {
        "version": 3,
        "results_root": str(results_root),
        "input": {
            "kind": "parquet",
            "path": str(parquet_path),
            "adapter": {
                "kind": "generic_features",
                "columns": {
                    "sequence": "sequence",
                    "features": "features",
                    "id": "id",
                },
                "policies": {},
            },
            "alphabet": "DNA",
            "limit": limit,
        },
        "render": {"renderer": "sequence_rows", "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "fmt": "png"}],
        "run": {"strict": False, "fail_on_skips": False, "emit_report": False, "report_path": None},
    }


def test_limit_short_circuits_row_iteration_when_selection_is_disabled(tmp_path: Path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "ACGT",
                "features": [
                    {
                        "id": "k1",
                        "kind": "kmer",
                        "span": {"start": 0, "end": 4, "strand": "fwd"},
                        "label": "ACGT",
                        "tags": ["demo"],
                    }
                ],
            },
            {
                "id": "r2",
                "sequence": "TGCA",
                "features": [
                    {
                        "id": "k2",
                        "kind": "kmer",
                        "span": {"start": 0, "end": 4, "strand": "fwd"},
                        "label": "TGCA",
                        "tags": ["demo"],
                    }
                ],
            },
            {
                "id": "r3",
                "sequence": "GCGC",
                "features": [
                    {
                        "id": "k3",
                        "kind": "kmer",
                        "span": {"start": 0, "end": 4, "strand": "fwd"},
                        "label": "GCGC",
                        "tags": ["demo"],
                    }
                ],
            },
        ],
    )

    payload = _generic_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        limit=1,
    )
    job_path = write_job(tmp_path / "job.yml", payload)

    report = run_job_v3(str(job_path))
    assert report.total_rows_seen == 1
    assert report.yielded_records == 1
