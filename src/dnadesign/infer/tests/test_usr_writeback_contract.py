"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_usr_writeback_contract.py

USR write-back contract tests for Infer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.infer.config import JobConfig, ModelConfig
from dnadesign.infer.engine import _plan_resume_for_usr, run_extract_job
from dnadesign.infer.errors import WriteBackError
from dnadesign.infer.writers.usr import write_back_usr
from dnadesign.usr import Dataset
from dnadesign.usr.tests.registry_helpers import register_test_namespace


class _AttachCaptureDataset:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def attach(
        self,
        path: Path,
        namespace: str,
        *,
        key: str,
        key_col: str | None = None,
        columns=None,
        allow_overwrite: bool = False,
        allow_missing: bool = False,
        parse_json: bool = True,
        backend: str = "pyarrow",
        note: str = "",
    ) -> int:
        payload = pq.read_table(path)
        self.calls.append(
            {
                "path": Path(path),
                "payload_schema_names": list(payload.schema.names),
                "namespace": namespace,
                "key": key,
                "key_col": key_col,
                "columns": list(columns) if columns is not None else None,
                "allow_overwrite": allow_overwrite,
                "allow_missing": allow_missing,
                "parse_json": parse_json,
                "backend": backend,
                "note": note,
            }
        )
        return 1


def test_write_back_usr_uses_infer_prefixed_columns_and_key_attach_contract() -> None:
    ds = _AttachCaptureDataset()
    write_back_usr(
        ds,
        ids=["id-1", "id-2"],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.25, 2.5]},
        overwrite=False,
    )

    assert len(ds.calls) == 1
    call = ds.calls[0]
    assert call["namespace"] == "infer"
    assert call["key"] == "id"
    assert call["key_col"] == "id"
    assert call["columns"] == ["infer__evo2_7b__job_a__ll_mean"]
    assert call["allow_overwrite"] is True
    assert call["payload_schema_names"] == ["id", "infer__evo2_7b__job_a__ll_mean"]


def test_plan_resume_for_usr_fails_fast_on_unreadable_records(tmp_path: Path) -> None:
    broken = tmp_path / "records.parquet"
    broken.write_text("not a parquet file", encoding="utf-8")
    ds = SimpleNamespace(records_path=broken)
    out = SimpleNamespace(id="ll_mean")
    with pytest.raises(WriteBackError, match="resume scan failed"):
        _plan_resume_for_usr(
            ds=ds,
            ids=["id-1"],
            model_id="evo2_7b",
            job_id="job_a",
            outputs=[out],
            overwrite=False,
        )


def test_plan_resume_for_usr_uses_infer_prefixed_columns(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": ["id-1", "id-2", "id-3"],
            "infer__evo2_7b__job_a__ll_mean": [1.0, None, 3.0],
        }
    )
    pq.write_table(table, path)
    ds = SimpleNamespace(records_path=path)
    out = SimpleNamespace(id="ll_mean")
    todo_idx, existing = _plan_resume_for_usr(
        ds=ds,
        ids=["id-1", "id-2", "id-3"],
        model_id="evo2_7b",
        job_id="job_a",
        outputs=[out],
        overwrite=False,
    )
    assert todo_idx == [1]
    assert existing["ll_mean"] == [1.0, None, 3.0]


def test_run_extract_job_usr_write_back_does_not_duplicate_final_call(monkeypatch) -> None:
    seqs = ["ACGT", "TGCA", "GGGG"]
    ids = ["id-1", "id-2", "id-3"]
    ds = SimpleNamespace()

    monkeypatch.setattr(
        "dnadesign.infer.engine.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr("dnadesign.infer.engine._validate_alphabet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "dnadesign.infer.engine._get_adapter",
        lambda _model: SimpleNamespace(log_likelihood=lambda chunk, **_kwargs: [float(i) for i, _ in enumerate(chunk)]),
    )
    calls: list[dict[str, object]] = []

    def _capture_write_back(ds_obj, *, ids, model_id, job_id, columnar, overwrite):
        calls.append(
            {
                "ds": ds_obj,
                "ids": list(ids),
                "model_id": model_id,
                "job_id": job_id,
                "columnar": dict(columnar),
                "overwrite": overwrite,
            }
        )

    monkeypatch.setattr("dnadesign.infer.engine.write_back_usr", _capture_write_back)
    monkeypatch.setattr(
        "dnadesign.infer.engine._plan_resume_for_usr",
        lambda **_kwargs: (list(range(len(ids))), {"ll_mean": [None] * len(ids)}),
    )

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna", batch_size=2)
    job = JobConfig(
        id="job_a",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo"},
        outputs=[{"id": "ll_mean", "fn": "evo2.log_likelihood", "format": "float", "params": {}}],
        io={"write_back": True, "overwrite": False},
    )

    result = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert "ll_mean" in result
    assert len(calls) == 2
    assert [call["ids"] for call in calls] == [["id-1", "id-2"], ["id-3"]]


def test_usr_chunk_write_back_is_append_safe_and_resume_reads_overlay(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec="infer__evo2_7b__job_a__ll_mean:float64",
        overwrite=True,
    )
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "GGGG", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    ids = ds.head(3, columns=["id"])["id"].tolist()

    write_back_usr(
        ds,
        ids=ids[:2],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0, 2.0]},
        overwrite=False,
    )
    write_back_usr(
        ds,
        ids=ids[2:],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [3.0]},
        overwrite=False,
    )

    overlays = ds.list_overlays()
    infer_overlay = next(overlay for overlay in overlays if overlay.namespace == "infer")
    overlay_table = pq.read_table(infer_overlay.path)
    assert overlay_table.num_rows == 3

    out = SimpleNamespace(id="ll_mean")
    todo_idx, existing = _plan_resume_for_usr(
        ds=ds,
        ids=ids,
        model_id="evo2_7b",
        job_id="job_a",
        outputs=[out],
        overwrite=False,
    )
    assert todo_idx == []
    assert existing["ll_mean"] == [1.0, 2.0, 3.0]


def test_run_extract_job_usr_resume_skips_completed_rows_from_overlay(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec="infer__evo2_7b__job_a__ll_mean:float64",
        overwrite=True,
    )
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "GGGG", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    ids = ds.head(3, columns=["id"])["id"].tolist()
    seqs = ds.head(3, columns=["sequence"])["sequence"].tolist()

    monkeypatch.setattr(
        "dnadesign.infer.engine.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr("dnadesign.infer.engine._validate_alphabet", lambda *_args, **_kwargs: None)

    adapter_calls = {"count": 0}

    class _Adapter:
        @staticmethod
        def log_likelihood(chunk, **_kwargs):
            adapter_calls["count"] += 1
            return [float(i + 1) for i, _ in enumerate(chunk)]

    monkeypatch.setattr("dnadesign.infer.engine._get_adapter", lambda _model: _Adapter())

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna", batch_size=2)
    job = JobConfig(
        id="job_a",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "root": str(root)},
        outputs=[{"id": "ll_mean", "fn": "evo2.log_likelihood", "format": "float", "params": {}}],
        io={"write_back": True, "overwrite": False},
    )

    first = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert list(first["ll_mean"]) == [1.0, 2.0, 1.0]
    assert adapter_calls["count"] == 2

    class _FailAdapter:
        @staticmethod
        def log_likelihood(_chunk, **_kwargs):
            raise AssertionError("resume should not invoke adapter when all rows are complete")

    monkeypatch.setattr("dnadesign.infer.engine._get_adapter", lambda _model: _FailAdapter())
    second = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert list(second["ll_mean"]) == [1.0, 2.0, 1.0]
