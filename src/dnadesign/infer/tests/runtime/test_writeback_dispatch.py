"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_writeback_dispatch.py

Contract tests for extract final write-back dispatch.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.infer.src.errors import WriteBackError
from dnadesign.infer.src.runtime.writeback_dispatch import run_extract_write_back


def test_run_extract_write_back_records_routes_to_records_writer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _records(records, *, model_id, job_id, columnar, overwrite):
        captured["records"] = records
        captured["model_id"] = model_id
        captured["job_id"] = job_id
        captured["columnar"] = dict(columnar)
        captured["overwrite"] = overwrite

    monkeypatch.setattr("dnadesign.infer.src.runtime.writeback_dispatch.write_back_records", _records)

    run_extract_write_back(
        write_back=True,
        source="records",
        records=[{"sequence": "ACGT"}],
        pt_path=None,
        ds=None,
        ids=None,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0]},
        overwrite=False,
    )

    assert captured == {
        "records": [{"sequence": "ACGT"}],
        "model_id": "evo2_7b",
        "job_id": "job_a",
        "columnar": {"ll_mean": [1.0]},
        "overwrite": False,
    }


def test_run_extract_write_back_pt_file_routes_to_pt_writer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _pt(path, records, *, model_id, job_id, columnar, overwrite):
        captured["pt_path"] = path
        captured["records"] = records
        captured["model_id"] = model_id
        captured["job_id"] = job_id
        captured["columnar"] = dict(columnar)
        captured["overwrite"] = overwrite

    monkeypatch.setattr("dnadesign.infer.src.runtime.writeback_dispatch.write_back_pt_file", _pt)

    run_extract_write_back(
        write_back=True,
        source="pt_file",
        records=[{"sequence": "ACGT"}],
        pt_path="/tmp/in.pt",
        ds=None,
        ids=None,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0]},
        overwrite=True,
    )

    assert captured == {
        "pt_path": "/tmp/in.pt",
        "records": [{"sequence": "ACGT"}],
        "model_id": "evo2_7b",
        "job_id": "job_a",
        "columnar": {"ll_mean": [1.0]},
        "overwrite": True,
    }


def test_run_extract_write_back_usr_requires_ids_and_dataset() -> None:
    try:
        run_extract_write_back(
            write_back=True,
            source="usr",
            records=None,
            pt_path=None,
            ds=object(),
            ids=None,
            model_id="evo2_7b",
            job_id="job_a",
            columnar={"ll_mean": [1.0]},
            overwrite=False,
        )
        raise AssertionError("expected WriteBackError")
    except WriteBackError as exc:
        assert "requires ids and dataset handle" in str(exc)


def test_run_extract_write_back_usr_with_contract_inputs_is_noop() -> None:
    run_extract_write_back(
        write_back=True,
        source="usr",
        records=None,
        pt_path=None,
        ds=object(),
        ids=["id-1"],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0]},
        overwrite=False,
    )


def test_run_extract_write_back_rejects_unsupported_source() -> None:
    try:
        run_extract_write_back(
            write_back=True,
            source="sequences",
            records=None,
            pt_path=None,
            ds=None,
            ids=None,
            model_id="evo2_7b",
            job_id="job_a",
            columnar={"ll_mean": [1.0]},
            overwrite=False,
        )
        raise AssertionError("expected WriteBackError")
    except WriteBackError as exc:
        assert "write_back not supported" in str(exc)


def test_run_extract_write_back_ignores_source_when_write_back_is_false() -> None:
    run_extract_write_back(
        write_back=False,
        source="not-a-real-source",
        records=None,
        pt_path=None,
        ds=None,
        ids=None,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0]},
        overwrite=False,
    )
