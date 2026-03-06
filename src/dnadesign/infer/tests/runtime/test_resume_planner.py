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
