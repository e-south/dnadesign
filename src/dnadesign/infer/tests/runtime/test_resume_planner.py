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

from dnadesign.devtools.tests.support.usr import register_test_namespace
from dnadesign.infer.src.errors import WriteBackError
from dnadesign.infer.src.features.contracts import (
    SequenceFeatureBundleConfig,
    SequenceFeatureContextConfig,
    SequenceFeaturePoolingConfig,
)
from dnadesign.infer.src.features.execution import build_feature_bundle_outputs
from dnadesign.infer.src.runtime.resume_planner import plan_resume_for_usr, read_usr_column_values, read_usr_columns
from dnadesign.usr import Dataset


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


def test_plan_resume_for_usr_chunks_large_id_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["id-1", "id-2", "id-3", "id-4"],
                "infer__evo2_7b__job_a__ll_mean": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        path,
    )
    ds = SimpleNamespace(records_path=path, list_overlays=lambda: [])
    out = SimpleNamespace(id="ll_mean")

    monkeypatch.setenv("DNADESIGN_INFER_RESUME_FILTER_CHUNK", "2")

    captured_filters: list[object] = []
    read_table_original = pq.read_table

    def _capture_read_table(*args, **kwargs):
        captured_filters.append(kwargs.get("filters"))
        return read_table_original(*args, **kwargs)

    monkeypatch.setattr("pyarrow.parquet.read_table", _capture_read_table)

    todo_idx, existing = plan_resume_for_usr(
        ds=ds,
        ids=["id-1", "id-2", "id-3", "id-4"],
        model_id="evo2_7b",
        job_id="job_a",
        outputs=[out],
        overwrite=False,
    )

    assert todo_idx == []
    assert existing["ll_mean"] == [1.0, 2.0, 3.0, 4.0]
    assert captured_filters == [
        [("id", "in", ["id-1", "id-2"])],
        [("id", "in", ["id-3", "id-4"])],
    ]


def test_plan_resume_for_usr_fails_fast_on_invalid_resume_filter_chunk_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["id-1"],
                "infer__evo2_7b__job_a__ll_mean": [1.0],
            }
        ),
        path,
    )
    ds = SimpleNamespace(records_path=path, list_overlays=lambda: [])
    out = SimpleNamespace(id="ll_mean")

    monkeypatch.setenv("DNADESIGN_INFER_RESUME_FILTER_CHUNK", "0")

    with pytest.raises(WriteBackError, match="DNADESIGN_INFER_RESUME_FILTER_CHUNK"):
        plan_resume_for_usr(
            ds=ds,
            ids=["id-1"],
            model_id="evo2_7b",
            job_id="job_a",
            outputs=[out],
            overwrite=False,
        )


def test_plan_resume_for_usr_reads_values_from_infer_overlay_parts(tmp_path: Path) -> None:
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

    ds.write_overlay_part("infer", pa.table({"id": [ids[0]], "infer__evo2_7b__job_a__ll_mean": [1.0]}), key="id")
    ds.write_overlay_part("infer", pa.table({"id": [ids[2]], "infer__evo2_7b__job_a__ll_mean": [3.0]}), key="id")

    out = SimpleNamespace(id="ll_mean")
    todo_idx, existing = plan_resume_for_usr(
        ds=ds,
        ids=ids,
        model_id="evo2_7b",
        job_id="job_a",
        outputs=[out],
        overwrite=False,
    )

    assert todo_idx == [1]
    assert existing["ll_mean"] == [1.0, None, 3.0]


def test_read_usr_column_values_overlay_parts_keep_last_non_null_value(tmp_path: Path) -> None:
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
    row_id = ds.head(1, columns=["id"])["id"].tolist()[0]
    column_name = "infer__evo2_7b__job_a__ll_mean"

    ds.write_overlay_part("infer", pa.table({"id": [row_id], column_name: [1.0]}), key="id")
    ds.write_overlay_part(
        "infer",
        pa.table({"id": [row_id], column_name: pa.array([None], type=pa.float64())}),
        key="id",
    )
    ds.write_overlay_part("infer", pa.table({"id": [row_id], column_name: [2.0]}), key="id")

    values = read_usr_column_values(ds=ds, ids=[row_id], column_name=column_name)

    assert values == [2.0]


def test_read_usr_columns_overlay_parts_keep_last_non_null_values_across_multiple_columns(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec="infer__evo2_7b__job_a__ll_mean:float64,infer__evo2_7b__job_a__feature_request_digest:string",
        overwrite=True,
    )
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [{"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"}],
        source="unit",
    )
    row_id = ds.head(1, columns=["id"])["id"].tolist()[0]
    score_column = "infer__evo2_7b__job_a__ll_mean"
    digest_column = "infer__evo2_7b__job_a__feature_request_digest"

    ds.write_overlay_part(
        "infer",
        pa.table({"id": [row_id], score_column: [1.0], digest_column: ["digest-a"]}),
        key="id",
    )
    ds.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": [row_id],
                score_column: pa.array([None], type=pa.float64()),
                digest_column: ["digest-b"],
            }
        ),
        key="id",
    )
    ds.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": [row_id],
                score_column: [2.0],
                digest_column: pa.array([None], type=pa.string()),
            }
        ),
        key="id",
    )

    values = read_usr_columns(ds=ds, ids=[row_id], column_names=[score_column, digest_column])

    assert values == {
        score_column: [2.0],
        digest_column: ["digest-b"],
    }


def test_plan_resume_for_usr_preserves_feature_output_mapping_with_mixed_overlay_nulls(tmp_path: Path) -> None:
    bundle = SequenceFeatureBundleConfig(
        context=SequenceFeatureContextConfig(kind="template_1kb"),
        pooling=SequenceFeaturePoolingConfig(seq_mean=True, anchor_mean_for_templated=True),
        collect_log_likelihood=True,
        collect_output_layer_mean=True,
        collect_intermediate_embedding=True,
        intermediate_block=23,
    )
    outputs = [
        SimpleNamespace(id=payload["id"])
        for payload in build_feature_bundle_outputs(bundle=bundle, model_id="evo2_20b")
    ]
    infer_cols = {output.id: f"infer__evo2_20b__template_1kb_20b_features__{output.id}" for output in outputs}
    list_type = pa.list_(pa.float64())

    records_path = tmp_path / "records.parquet"
    row_ids = ["id-1", "id-2"]
    pq.write_table(
        pa.table(
            {
                "id": row_ids,
                infer_cols["log_likelihood__total"]: [1.0, 2.0],
                infer_cols["log_likelihood__mean_per_token"]: [0.1, 0.2],
                infer_cols["output_layer_mean__seq_mean"]: pa.array([[10.0, 11.0], [20.0, 21.0]], type=list_type),
                infer_cols["output_layer_mean__anchor_mean"]: pa.array([[12.0, 13.0], [22.0, 23.0]], type=list_type),
                infer_cols["intermediate_embedding__block23_mlp_out__seq_mean"]: pa.array(
                    [[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]],
                    type=list_type,
                ),
                infer_cols["intermediate_embedding__block23_mlp_out__anchor_mean"]: pa.array(
                    [[33.0, 34.0, 35.0], [43.0, 44.0, 45.0]],
                    type=list_type,
                ),
            }
        ),
        records_path,
    )

    overlay_dir = tmp_path / "infer.parts"
    overlay_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["id-1"],
                infer_cols["log_likelihood__total"]: [10.0],
                infer_cols["log_likelihood__mean_per_token"]: pa.array([None], type=pa.float64()),
                infer_cols["output_layer_mean__seq_mean"]: pa.array([[100.0, 101.0]], type=list_type),
                infer_cols["output_layer_mean__anchor_mean"]: pa.array([None], type=list_type),
                infer_cols["intermediate_embedding__block23_mlp_out__seq_mean"]: pa.array(
                    [[200.0, 201.0, 202.0]],
                    type=list_type,
                ),
                infer_cols["intermediate_embedding__block23_mlp_out__anchor_mean"]: pa.array([None], type=list_type),
            }
        ),
        overlay_dir / "part-0001.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": ["id-1"],
                infer_cols["log_likelihood__total"]: pa.array([None], type=pa.float64()),
                infer_cols["log_likelihood__mean_per_token"]: [1.5],
                infer_cols["output_layer_mean__seq_mean"]: pa.array([None], type=list_type),
                infer_cols["output_layer_mean__anchor_mean"]: pa.array([[110.0, 111.0]], type=list_type),
                infer_cols["intermediate_embedding__block23_mlp_out__seq_mean"]: pa.array([None], type=list_type),
                infer_cols["intermediate_embedding__block23_mlp_out__anchor_mean"]: pa.array(
                    [[210.0, 211.0, 212.0]],
                    type=list_type,
                ),
            }
        ),
        overlay_dir / "part-0002.parquet",
    )

    ds = SimpleNamespace(
        records_path=records_path,
        list_overlays=lambda: [SimpleNamespace(namespace="infer", path=overlay_dir)],
    )

    todo_idx, existing = plan_resume_for_usr(
        ds=ds,
        ids=["id-2", "id-1", "id-2"],
        model_id="evo2_20b",
        job_id="template_1kb_20b_features",
        outputs=outputs,
        overwrite=False,
    )

    assert todo_idx == []
    assert existing["log_likelihood__total"] == [2.0, 10.0, 2.0]
    assert existing["log_likelihood__mean_per_token"] == [0.2, 1.5, 0.2]
    assert existing["output_layer_mean__seq_mean"] == [[20.0, 21.0], [100.0, 101.0], [20.0, 21.0]]
    assert existing["output_layer_mean__anchor_mean"] == [[22.0, 23.0], [110.0, 111.0], [22.0, 23.0]]
    assert existing["intermediate_embedding__block23_mlp_out__seq_mean"] == [
        [40.0, 41.0, 42.0],
        [200.0, 201.0, 202.0],
        [40.0, 41.0, 42.0],
    ]
    assert existing["intermediate_embedding__block23_mlp_out__anchor_mean"] == [
        [43.0, 44.0, 45.0],
        [210.0, 211.0, 212.0],
        [43.0, 44.0, 45.0],
    ]
