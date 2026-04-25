"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_usr_writeback_contract.py

USR write-back contract tests for Infer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.infer.src.config import JobConfig, ModelConfig
from dnadesign.infer.src.engine import _plan_resume_for_usr, run_extract_job
from dnadesign.infer.src.errors import RuntimeOOMError, WriteBackError
from dnadesign.infer.src.runtime.resume_planner import read_usr_columns
from dnadesign.infer.src.writers.usr import write_back_usr
from dnadesign.testsupport.usr import register_test_namespace
from dnadesign.usr import Dataset


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
        actor: dict[str, object] | None = None,
        event_args: dict[str, object] | None = None,
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
                "actor": actor,
                "event_args": event_args,
            }
        )
        return 1


class _OverlayPartCaptureDataset:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def list_overlays(self):
        return []

    def write_overlay_part(
        self,
        namespace: str,
        table_or_batches,
        *,
        key: str = "id",
        key_col: str | None = None,
        allow_missing: bool = False,
        actor: dict[str, object] | None = None,
        event_args: dict[str, object] | None = None,
    ) -> int:
        self.calls.append(
            {
                "namespace": namespace,
                "key": key,
                "key_col": key_col,
                "columns": list(table_or_batches.columns),
                "payload": table_or_batches.to_dict(orient="list"),
                "allow_missing": allow_missing,
                "actor": actor,
                "event_args": event_args,
            }
        )
        return 1


class _InProcessAttachCaptureDataset:
    def __init__(self) -> None:
        self.attach_called = False
        self.dir = Path("/tmp/infer-inprocess-capture")
        self.records_path = self.dir / "records.parquet"

    def attach(self, *args, **kwargs) -> int:
        self.attach_called = True
        raise AssertionError("write_back_usr should use in-process attach for dataset-like objects")

    def _auto_freeze_registry(self, *, record_auto_event: bool = True):
        return self.dir / "_registry" / "registry.hash.yaml", "hash", False

    def _record_event(self, *args, **kwargs) -> None:
        return None

    def _registry(self, *, required: bool):
        return {}

    def _registry_hash(self, *, required: bool):
        return "hash"

    def _validate_registry_schema(self, *, namespace: str, schema, key: str) -> None:
        return None


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


def test_write_back_usr_uses_overlay_parts_when_dataset_supports_part_contract() -> None:
    ds = _OverlayPartCaptureDataset()
    write_back_usr(
        ds,
        ids=["id-1", "id-2"],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.25, 2.5]},
        overwrite=False,
        event_args={"infer_notify_suppress": True},
    )

    assert len(ds.calls) == 1
    call = ds.calls[0]
    assert call["namespace"] == "infer"
    assert call["key"] == "id"
    assert call["columns"] == ["id", "infer__evo2_7b__job_a__ll_mean"]
    assert call["payload"] == {
        "id": ["id-1", "id-2"],
        "infer__evo2_7b__job_a__ll_mean": [1.25, 2.5],
    }
    assert call["allow_missing"] is False
    assert call["actor"]["tool"] == "infer"
    assert call["event_args"]["infer_notify_suppress"] is True
    assert call["event_args"]["infer_output"]["id"] == "ll_mean"


def test_write_back_usr_uses_inprocess_attach_when_dataset_supports_internal_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = _InProcessAttachCaptureDataset()
    captured: dict[str, object] = {}

    def _capture_attach_frame(**kwargs) -> int:
        captured["incoming_columns"] = list(kwargs["incoming"].columns)
        captured["columns"] = list(kwargs["columns"])
        captured["actor"] = kwargs["actor"]
        captured["event_args"] = kwargs["event_args"]
        captured["fail_on_non_null_overwrite"] = kwargs["fail_on_non_null_overwrite"]
        return 1

    monkeypatch.setattr("dnadesign.infer.src.writers.usr._attach_frame_dataset", _capture_attach_frame)

    write_back_usr(
        ds,
        ids=["id-1"],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.25]},
        overwrite=False,
        event_args={"infer_notify_suppress": True},
    )

    assert ds.attach_called is False
    assert captured["incoming_columns"] == ["id", "infer__evo2_7b__job_a__ll_mean"]
    assert captured["columns"] == ["infer__evo2_7b__job_a__ll_mean"]
    assert captured["actor"]["tool"] == "infer"
    assert captured["event_args"]["infer_notify_suppress"] is True
    assert captured["event_args"]["infer_output"]["id"] == "ll_mean"
    assert captured["fail_on_non_null_overwrite"] is True


def test_write_back_usr_inprocess_path_skips_external_overwrite_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = _InProcessAttachCaptureDataset()

    def _capture_attach_frame(**_kwargs) -> int:
        return 1

    def _fail_guard(*args, **kwargs):
        raise AssertionError("in-process write_back_usr should not use external overwrite guard")

    monkeypatch.setattr("dnadesign.infer.src.writers.usr._attach_frame_dataset", _capture_attach_frame)
    monkeypatch.setattr("dnadesign.infer.src.writers.usr._guard_usr_overwrite", _fail_guard)

    write_back_usr(
        ds,
        ids=["id-1"],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.25]},
        overwrite=False,
    )


def test_write_back_usr_respects_overwrite_flag_true() -> None:
    ds = _AttachCaptureDataset()
    write_back_usr(
        ds,
        ids=["id-1"],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.25]},
        overwrite=True,
    )

    assert len(ds.calls) == 1
    call = ds.calls[0]
    assert call["allow_overwrite"] is True


def test_write_back_usr_tags_attach_with_infer_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _AttachCaptureDataset()
    monkeypatch.setenv("USR_ACTOR_RUN_ID", "infer-run-42")

    write_back_usr(
        ds,
        ids=["id-1"],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.25]},
        overwrite=False,
    )

    assert ds.calls[0]["actor"] == {
        "tool": "infer",
        "run_id": "infer-run-42",
        "host": ds.calls[0]["actor"]["host"],
        "pid": ds.calls[0]["actor"]["pid"],
    }


def test_write_back_usr_records_infer_actor_in_usr_events(tmp_path: Path) -> None:
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
        [{"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"}],
        source="unit",
    )
    row_id = ds.head(1, columns=["id"])["id"].tolist()

    write_back_usr(
        ds,
        ids=row_id,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0]},
        overwrite=False,
    )

    events = [json.loads(line) for line in ds.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    overlay_part_events = [event for event in events if event["action"] == "write_overlay_part"]

    assert len(overlay_part_events) == 1
    assert overlay_part_events[0]["actor"]["tool"] == "infer"
    assert overlay_part_events[0]["actor"]["run_id"] == "infer-job_a"
    assert overlay_part_events[0]["args"]["rows_matched"] == 1
    assert overlay_part_events[0]["args"]["infer_output"]["id"] == "ll_mean"


def test_write_back_usr_fails_fast_on_existing_values_when_overwrite_false(tmp_path: Path) -> None:
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
        ],
        source="unit",
    )
    one_id = ds.head(1, columns=["id"])["id"].tolist()

    write_back_usr(
        ds,
        ids=one_id,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0]},
        overwrite=False,
    )

    with pytest.raises(WriteBackError, match="Refusing overwrite"):
        write_back_usr(
            ds,
            ids=one_id,
            model_id="evo2_7b",
            job_id="job_a",
            columnar={"ll_mean": [2.0]},
            overwrite=False,
        )


def test_write_back_usr_overwrite_guard_allows_new_columns_missing_from_existing_overlay(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec="infer__evo2_7b__job_a__ll_mean:float64,infer__evo2_7b__job_a__logits_mean:float64",
        overwrite=True,
    )
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    one_id = ds.head(1, columns=["id"])["id"].tolist()

    write_back_usr(
        ds,
        ids=one_id,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0]},
        overwrite=False,
    )
    write_back_usr(
        ds,
        ids=one_id,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"logits_mean": [2.0]},
        overwrite=False,
    )

    outputs = [SimpleNamespace(id="ll_mean"), SimpleNamespace(id="logits_mean")]
    todo_idx, existing = _plan_resume_for_usr(
        ds=ds,
        ids=one_id,
        model_id="evo2_7b",
        job_id="job_a",
        outputs=outputs,
        overwrite=False,
    )
    assert todo_idx == []
    assert existing["ll_mean"] == [1.0]
    assert existing["logits_mean"] == [2.0]


def test_write_back_usr_overwrite_guard_detects_existing_values_in_later_overlay_parts(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec="infer__evo2_7b__job_a__ll_mean:float64,infer__evo2_7b__job_a__logits_mean:float64",
        overwrite=True,
    )
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    one_id = ds.head(1, columns=["id"])["id"].tolist()

    write_back_usr(
        ds,
        ids=one_id,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [1.0]},
        overwrite=False,
    )
    write_back_usr(
        ds,
        ids=one_id,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"logits_mean": [2.0]},
        overwrite=False,
    )

    with pytest.raises(WriteBackError, match="Refusing overwrite"):
        write_back_usr(
            ds,
            ids=one_id,
            model_id="evo2_7b",
            job_id="job_a",
            columnar={"logits_mean": [3.0]},
            overwrite=False,
        )


def test_write_back_usr_overwrite_false_allows_existing_null_values(tmp_path: Path) -> None:
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
        [{"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"}],
        source="unit",
    )
    one_id = ds.head(1, columns=["id"])["id"].tolist()

    write_back_usr(
        ds,
        ids=one_id,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [None]},
        overwrite=False,
    )
    write_back_usr(
        ds,
        ids=one_id,
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [2.0]},
        overwrite=False,
    )

    out = SimpleNamespace(id="ll_mean")
    todo_idx, existing = _plan_resume_for_usr(
        ds=ds,
        ids=one_id,
        model_id="evo2_7b",
        job_id="job_a",
        outputs=[out],
        overwrite=False,
    )
    assert todo_idx == []
    assert existing["ll_mean"] == [2.0]


def test_write_back_usr_overwrite_guard_reads_only_requested_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay_path = tmp_path / "infer.parquet"
    table = pa.table(
        {
            "id": ["id-1", "id-2", "id-3"],
            "infer__evo2_7b__job_a__ll_mean": [1.0, None, 3.0],
        }
    )
    pq.write_table(table, overlay_path)

    class _OverlayAttachDataset(_AttachCaptureDataset):
        def list_overlays(self):
            return [SimpleNamespace(namespace="infer", path=overlay_path)]

    ds = _OverlayAttachDataset()

    captured_filters: list[object] = []
    read_table_original = pq.read_table

    def _capture_read_table(*args, **kwargs):
        path_arg = kwargs.get("where") if "where" in kwargs else args[0]
        if Path(path_arg) == overlay_path:
            captured_filters.append(kwargs.get("filters"))
        return read_table_original(*args, **kwargs)

    monkeypatch.setattr("pyarrow.parquet.read_table", _capture_read_table)

    write_back_usr(
        ds,
        ids=["id-2"],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [2.0]},
        overwrite=False,
    )

    assert captured_filters
    assert captured_filters[0] == [("id", "in", ["id-2"])]
    assert len(ds.calls) == 1


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
        "dnadesign.infer.src.runtime.ingest_loading.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr("dnadesign.infer.src.engine._validate_alphabet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "dnadesign.infer.src.engine._get_adapter",
        lambda _model: SimpleNamespace(log_likelihood=lambda chunk, **_kwargs: [float(i) for i, _ in enumerate(chunk)]),
    )
    calls: list[dict[str, object]] = []

    def _capture_write_back(ds_obj, *, ids, model_id, job_id, columnar, overwrite, event_args=None):
        calls.append(
            {
                "ds": ds_obj,
                "ids": list(ids),
                "model_id": model_id,
                "job_id": job_id,
                "columnar": dict(columnar),
                "overwrite": overwrite,
                "event_args": event_args,
            }
        )

    monkeypatch.setattr("dnadesign.infer.src.engine.write_back_usr", _capture_write_back)
    monkeypatch.setattr(
        "dnadesign.infer.src.engine._plan_resume_for_usr",
        lambda **_kwargs: (list(range(len(ids))), {"ll_mean": [None] * len(ids)}),
    )

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna", batch_size=2)
    job = JobConfig(
        id="job_a",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "root": "/tmp/usr-root"},
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


def test_write_back_usr_promotes_existing_infer_overlay_file_to_parts_without_data_loss(tmp_path: Path) -> None:
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
        ],
        source="unit",
    )
    ids = ds.head(2, columns=["id"])["id"].tolist()

    initial_overlay = tmp_path / "initial_infer_overlay.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [ids[0]],
                "infer__evo2_7b__job_a__ll_mean": [10.0],
            }
        ),
        initial_overlay,
    )
    ds.attach(
        initial_overlay,
        namespace="infer",
        key="id",
        key_col="id",
        columns=["infer__evo2_7b__job_a__ll_mean"],
        allow_overwrite=True,
        parse_json=False,
    )

    write_back_usr(
        ds,
        ids=[ids[1]],
        model_id="evo2_7b",
        job_id="job_a",
        columnar={"ll_mean": [11.0]},
        overwrite=False,
    )

    infer_overlay = next(overlay for overlay in ds.list_overlays() if overlay.namespace == "infer")
    assert Path(infer_overlay.path).is_dir()

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
    assert existing["ll_mean"] == [10.0, 11.0]


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
        "dnadesign.infer.src.runtime.ingest_loading.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr("dnadesign.infer.src.engine._validate_alphabet", lambda *_args, **_kwargs: None)

    adapter_calls = {"count": 0}

    class _Adapter:
        @staticmethod
        def log_likelihood(chunk, **_kwargs):
            adapter_calls["count"] += 1
            return [float(i + 1) for i, _ in enumerate(chunk)]

    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: _Adapter())

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

    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: _FailAdapter())
    second = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert list(second["ll_mean"]) == [1.0, 2.0, 1.0]


def test_run_extract_job_usr_resume_reads_completed_rows_from_overlay_parts(tmp_path: Path, monkeypatch) -> None:
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

    ds.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": ids[:2],
                "infer__evo2_7b__job_a__ll_mean": [10.0, 11.0],
            }
        ),
        key="id",
    )

    monkeypatch.setattr(
        "dnadesign.infer.src.runtime.ingest_loading.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr("dnadesign.infer.src.engine._validate_alphabet", lambda *_args, **_kwargs: None)

    adapter_calls = {"count": 0}

    class _Adapter:
        @staticmethod
        def log_likelihood(chunk, **_kwargs):
            adapter_calls["count"] += 1
            return [1.0 for _seq in chunk]

    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: _Adapter())

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna", batch_size=2)
    job = JobConfig(
        id="job_a",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "root": str(root)},
        outputs=[{"id": "ll_mean", "fn": "evo2.log_likelihood", "format": "float", "params": {}}],
        io={"write_back": True, "overwrite": False},
    )

    out = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)

    assert list(out["ll_mean"]) == [10.0, 11.0, 1.0]
    assert adapter_calls["count"] == 1


def test_run_extract_job_usr_resume_does_not_load_adapter_when_all_rows_are_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        ],
        source="unit",
    )
    ids = ds.head(2, columns=["id"])["id"].tolist()
    seqs = ds.head(2, columns=["sequence"])["sequence"].tolist()

    monkeypatch.setattr(
        "dnadesign.infer.src.runtime.ingest_loading.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr("dnadesign.infer.src.engine._validate_alphabet", lambda *_args, **_kwargs: None)

    class _Adapter:
        @staticmethod
        def log_likelihood(chunk, **_kwargs):
            return [float(i + 1) for i, _ in enumerate(chunk)]

    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: _Adapter())

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna", batch_size=2)
    job = JobConfig(
        id="job_a",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "root": str(root)},
        outputs=[{"id": "ll_mean", "fn": "evo2.log_likelihood", "format": "float", "params": {}}],
        io={"write_back": True, "overwrite": False},
    )

    first = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert list(first["ll_mean"]) == [1.0, 2.0]

    def _raise_if_called(_model):
        raise AssertionError("resume should not load adapter when all rows are complete")

    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", _raise_if_called)
    second = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert list(second["ll_mean"]) == [1.0, 2.0]


def test_run_extract_job_usr_resume_recovers_after_interrupted_partial_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "dnadesign.infer.src.runtime.ingest_loading.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr("dnadesign.infer.src.engine._validate_alphabet", lambda *_args, **_kwargs: None)

    interrupted_calls = {"count": 0}

    class _InterruptingAdapter:
        @staticmethod
        def log_likelihood(chunk, **_kwargs):
            interrupted_calls["count"] += 1
            if interrupted_calls["count"] == 1:
                return [10.0, 11.0]
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: _InterruptingAdapter())

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna", batch_size=2)
    job = JobConfig(
        id="job_a",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "root": str(root)},
        outputs=[{"id": "ll_mean", "fn": "evo2.log_likelihood", "format": "float", "params": {}}],
        io={"write_back": True, "overwrite": False},
    )

    with pytest.raises(RuntimeOOMError, match="simulated interruption"):
        run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert interrupted_calls["count"] == 2

    resumed_calls = {"count": 0}

    class _ResumeAdapter:
        @staticmethod
        def log_likelihood(chunk, **_kwargs):
            resumed_calls["count"] += 1
            assert len(chunk) == 1
            return [99.0]

    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: _ResumeAdapter())
    resumed = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)

    assert resumed_calls["count"] == 1
    assert list(resumed["ll_mean"]) == [10.0, 11.0, 99.0]


def test_run_extract_job_usr_subset_outputs_backfill_only_missing_target_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec=(
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__total:float64,"
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token:float64,"
            "infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean:list<float64>,"
            "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean:list<float64>,"
            "infer__evo2_20b__anchor_only_20b_features__metadata__feature_request_digest:string"
        ),
        overwrite=True,
    )
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    ids = ds.head(2, columns=["id"])["id"].tolist()
    seqs = ds.head(2, columns=["sequence"])["sequence"].tolist()

    ds.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": ids,
                "infer__evo2_20b__anchor_only_20b_features__log_likelihood__total": pa.array(
                    [1.0, None],
                    type=pa.float64(),
                ),
                "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": pa.array(
                    [None, None],
                    type=pa.float64(),
                ),
                "infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean": pa.array(
                    [None, None],
                    type=pa.list_(pa.float64()),
                ),
                (
                    "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean"
                ): pa.array(
                    [[7.0, 8.0], [9.0, 10.0]],
                    type=pa.list_(pa.float64()),
                ),
                "infer__evo2_20b__anchor_only_20b_features__metadata__feature_request_digest": ["digest-a", "digest-b"],
            }
        ),
        key="id",
    )

    monkeypatch.setattr(
        "dnadesign.infer.src.runtime.ingest_loading.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr("dnadesign.infer.src.engine._validate_alphabet", lambda *_args, **_kwargs: None)

    observed: dict[str, list[list[str]]] = {"sum": [], "mean": [], "logits": []}

    class _Adapter:
        @staticmethod
        def log_likelihood(chunk, *, method="native", reduction="sum"):
            assert method == "native"
            if reduction == "sum":
                observed["sum"].append(list(chunk))
                return [200.0 + float(index) for index, _seq in enumerate(chunk)]
            if reduction == "mean":
                observed["mean"].append(list(chunk))
                return [0.5 + float(index) for index, _seq in enumerate(chunk)]
            raise AssertionError(f"unexpected reduction {reduction}")

        @staticmethod
        def logits(chunk, *, pool, fmt):
            observed["logits"].append(list(chunk))
            assert pool == {"method": "mean", "dim": 1}
            assert fmt == "list"
            return [[11.0 + float(index), 21.0 + float(index)] for index, _seq in enumerate(chunk)]

    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: _Adapter())

    model = ModelConfig(id="evo2_20b", device="cpu", precision="fp32", alphabet="dna", batch_size=2)
    job = JobConfig(
        id="anchor_only_20b_features",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "root": str(root)},
        outputs=[
            {
                "id": "log_likelihood__total",
                "fn": "evo2.log_likelihood",
                "format": "float",
                "params": {"method": "native", "reduction": "sum"},
            },
            {
                "id": "log_likelihood__mean_per_token",
                "fn": "evo2.log_likelihood",
                "format": "float",
                "params": {"method": "native", "reduction": "mean"},
            },
            {
                "id": "output_layer_mean__seq_mean",
                "fn": "evo2.logits",
                "format": "list",
                "params": {"pool": {"method": "mean", "dim": 1}},
            },
        ],
        io={"write_back": True, "overwrite": False},
    )

    result = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)

    assert observed["sum"] == [[seqs[1]]]
    assert observed["mean"] == [seqs]
    assert observed["logits"] == [seqs]
    assert list(result["log_likelihood__total"]) == [1.0, 200.0]
    assert list(result["log_likelihood__mean_per_token"]) == [0.5, 1.5]
    assert list(result["output_layer_mean__seq_mean"]) == [[11.0, 21.0], [12.0, 22.0]]

    repaired = read_usr_columns(
        ds=ds,
        ids=ids,
        column_names=[
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__total",
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token",
            "infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean",
            "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean",
            "infer__evo2_20b__anchor_only_20b_features__metadata__feature_request_digest",
        ],
    )

    assert repaired == {
        "infer__evo2_20b__anchor_only_20b_features__log_likelihood__total": [1.0, 200.0],
        "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": [0.5, 1.5],
        "infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean": [[11.0, 21.0], [12.0, 22.0]],
        "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean": [
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        "infer__evo2_20b__anchor_only_20b_features__metadata__feature_request_digest": ["digest-a", "digest-b"],
    }


def test_run_extract_job_usr_subset_overwrite_preserves_unrelated_infer_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec=(
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__total:float64,"
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token:float64,"
            "infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean:list<float64>,"
            "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean:list<float64>,"
            "infer__evo2_20b__anchor_only_20b_features__metadata__feature_request_digest:string"
        ),
        overwrite=True,
    )
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    ids = ds.head(2, columns=["id"])["id"].tolist()
    seqs = ds.head(2, columns=["sequence"])["sequence"].tolist()

    ds.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": ids,
                "infer__evo2_20b__anchor_only_20b_features__log_likelihood__total": [10.0, 11.0],
                "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": [0.1, 0.2],
                "infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean": pa.array(
                    [[1.0, 2.0], [3.0, 4.0]],
                    type=pa.list_(pa.float64()),
                ),
                (
                    "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean"
                ): pa.array(
                    [[7.0, 8.0], [9.0, 10.0]],
                    type=pa.list_(pa.float64()),
                ),
                "infer__evo2_20b__anchor_only_20b_features__metadata__feature_request_digest": ["digest-a", "digest-b"],
            }
        ),
        key="id",
    )

    monkeypatch.setattr(
        "dnadesign.infer.src.runtime.ingest_loading.load_usr_input",
        lambda **_kwargs: (seqs, ids, ds),
    )
    monkeypatch.setattr("dnadesign.infer.src.engine._validate_alphabet", lambda *_args, **_kwargs: None)

    class _Adapter:
        @staticmethod
        def log_likelihood(chunk, *, method="native", reduction="sum"):
            assert method == "native"
            if reduction == "sum":
                return [300.0 + float(index) for index, _seq in enumerate(chunk)]
            if reduction == "mean":
                return [0.9 + float(index) for index, _seq in enumerate(chunk)]
            raise AssertionError(f"unexpected reduction {reduction}")

        @staticmethod
        def logits(chunk, *, pool, fmt):
            assert pool == {"method": "mean", "dim": 1}
            assert fmt == "list"
            return [[31.0 + float(index), 41.0 + float(index)] for index, _seq in enumerate(chunk)]

    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: _Adapter())

    model = ModelConfig(id="evo2_20b", device="cpu", precision="fp32", alphabet="dna", batch_size=2)
    job = JobConfig(
        id="anchor_only_20b_features",
        operation="extract",
        ingest={"source": "usr", "dataset": "demo", "root": str(root)},
        outputs=[
            {
                "id": "log_likelihood__total",
                "fn": "evo2.log_likelihood",
                "format": "float",
                "params": {"method": "native", "reduction": "sum"},
            },
            {
                "id": "log_likelihood__mean_per_token",
                "fn": "evo2.log_likelihood",
                "format": "float",
                "params": {"method": "native", "reduction": "mean"},
            },
            {
                "id": "output_layer_mean__seq_mean",
                "fn": "evo2.logits",
                "format": "list",
                "params": {"pool": {"method": "mean", "dim": 1}},
            },
        ],
        io={"write_back": True, "overwrite": True},
    )

    result = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)

    assert list(result["log_likelihood__total"]) == [300.0, 301.0]
    assert list(result["log_likelihood__mean_per_token"]) == [0.9, 1.9]
    assert list(result["output_layer_mean__seq_mean"]) == [[31.0, 41.0], [32.0, 42.0]]

    repaired = read_usr_columns(
        ds=ds,
        ids=ids,
        column_names=[
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__total",
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token",
            "infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean",
            "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean",
            "infer__evo2_20b__anchor_only_20b_features__metadata__feature_request_digest",
        ],
    )

    assert repaired == {
        "infer__evo2_20b__anchor_only_20b_features__log_likelihood__total": [300.0, 301.0],
        "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": [0.9, 1.9],
        "infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean": [[31.0, 41.0], [32.0, 42.0]],
        "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean": [
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        "infer__evo2_20b__anchor_only_20b_features__metadata__feature_request_digest": ["digest-a", "digest-b"],
    }
