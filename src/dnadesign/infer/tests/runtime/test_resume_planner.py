"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_resume_planner.py

Contract tests for USR resume planning module boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.infer.src.errors import WriteBackError
from dnadesign.infer.src.runtime.resume_planner import plan_resume_for_usr


def test_plan_resume_for_usr_overwrite_short_circuits_scan() -> None:
    out = SimpleNamespace(id="ll_mean")
    todo_idx, existing = plan_resume_for_usr(
        ds=None,
        ids=["id-1", "id-2"],
        model_id="evo2_7b",
        job_id="job_a",
        outputs=[out],
        overwrite=True,
    )
    assert todo_idx == [0, 1]
    assert existing == {"ll_mean": [None, None]}


def test_plan_resume_for_usr_fails_fast_on_unreadable_records(tmp_path: Path) -> None:
    broken = tmp_path / "records.parquet"
    broken.write_text("not a parquet file", encoding="utf-8")
    ds = SimpleNamespace(records_path=broken)
    out = SimpleNamespace(id="ll_mean")

    with pytest.raises(WriteBackError, match="resume scan failed"):
        plan_resume_for_usr(
            ds=ds,
            ids=["id-1"],
            model_id="evo2_7b",
            job_id="job_a",
            outputs=[out],
            overwrite=False,
        )


def test_plan_resume_for_usr_reads_only_requested_ids_and_preserves_duplicate_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["id-1", "id-2", "id-3"],
                "infer__evo2_7b__job_a__ll_mean": [1.0, 2.0, None],
            }
        ),
        path,
    )
    ds = SimpleNamespace(records_path=path, list_overlays=lambda: [])
    out = SimpleNamespace(id="ll_mean")

    captured_filters: list[object] = []
    read_table_original = pq.read_table

    def _capture_read_table(*args, **kwargs):
        captured_filters.append(kwargs.get("filters"))
        return read_table_original(*args, **kwargs)

    monkeypatch.setattr("pyarrow.parquet.read_table", _capture_read_table)

    todo_idx, existing = plan_resume_for_usr(
        ds=ds,
        ids=["id-2", "id-2", "id-1"],
        model_id="evo2_7b",
        job_id="job_a",
        outputs=[out],
        overwrite=False,
    )

    assert todo_idx == []
    assert existing["ll_mean"] == [2.0, 2.0, 1.0]
    assert captured_filters
    assert captured_filters[0] == [("id", "in", ["id-2", "id-1"])]
