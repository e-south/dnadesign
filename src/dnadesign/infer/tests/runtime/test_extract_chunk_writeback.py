"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_extract_chunk_writeback.py

Contract tests for extract USR chunk write-back callback construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.infer.src.errors import WriteBackError
from dnadesign.infer.src.runtime.extract_chunk_writeback import build_extract_chunk_write_back


def test_build_extract_chunk_write_back_returns_none_when_not_usr() -> None:
    callback = build_extract_chunk_write_back(
        source="records",
        write_back=True,
        ds=None,
        ids=None,
        model_id="evo2_7b",
        job_id="job_a",
        out_id="ll_mean",
        overwrite=False,
    )
    assert callback is None


def test_build_extract_chunk_write_back_returns_none_when_disabled() -> None:
    callback = build_extract_chunk_write_back(
        source="usr",
        write_back=False,
        ds=object(),
        ids=["id-1"],
        model_id="evo2_7b",
        job_id="job_a",
        out_id="ll_mean",
        overwrite=False,
    )
    assert callback is None


def test_build_extract_chunk_write_back_fails_fast_for_missing_usr_contracts() -> None:
    with pytest.raises(WriteBackError, match="requires ids and dataset handle"):
        build_extract_chunk_write_back(
            source="usr",
            write_back=True,
            ds=object(),
            ids=None,
            model_id="evo2_7b",
            job_id="job_a",
            out_id="ll_mean",
            overwrite=False,
        )


def test_build_extract_chunk_write_back_writes_chunk_values() -> None:
    calls: list[dict[str, object]] = []

    def _writer(ds, *, ids, model_id, job_id, columnar, overwrite):
        calls.append(
            {
                "ds": ds,
                "ids": list(ids),
                "model_id": model_id,
                "job_id": job_id,
                "columnar": dict(columnar),
                "overwrite": overwrite,
            }
        )

    ds = object()
    callback = build_extract_chunk_write_back(
        source="usr",
        write_back=True,
        ds=ds,
        ids=["id-1", "id-2", "id-3"],
        model_id="evo2_7b",
        job_id="job_a",
        out_id="ll_mean",
        overwrite=True,
        writer=_writer,
    )
    assert callback is not None
    callback([0, 2], [1.0, 3.0])

    assert calls == [
        {
            "ds": ds,
            "ids": ["id-1", "id-3"],
            "model_id": "evo2_7b",
            "job_id": "job_a",
            "columnar": {"ll_mean": [1.0, 3.0]},
            "overwrite": True,
        }
    ]
