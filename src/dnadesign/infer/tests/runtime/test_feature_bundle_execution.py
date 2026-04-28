"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_feature_bundle_execution.py

Runtime contract tests for Evo2 promoter feature bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from Bio.Seq import Seq
from pyarrow import parquet as pq

from dnadesign.infer import export_evo2_promoter_opal_matrix
from dnadesign.infer.src.config import JobConfig, ModelConfig
from dnadesign.infer.src.contracts import infer_usr_column_name
from dnadesign.infer.src.engine import run_extract_job
from dnadesign.infer.src.errors import CapabilityError, RuntimeOOMError
from dnadesign.infer.src.features.aliases import FEATURE_ALIAS_RELATIVE_PATH
from dnadesign.infer.src.features.context import resolve_sequence_contexts
from dnadesign.infer.src.features.execution import (
    _LOG_LIKELIHOOD_MEAN,
    _LOG_LIKELIHOOD_TOTAL,
    _OUTPUT_LAYER_SEQ_MEAN,
    _apply_digest_resume_guard,
    _existing_feature_metadata_values,
    build_feature_bundle_outputs,
    execute_feature_bundle,
    feature_metadata_output_ids,
)
from dnadesign.usr import Dataset, SequenceViewRecord, ensure_sequence_contract_namespaces, write_sequence_views


def _assert_list_close(observed: list[float], expected: list[float]) -> None:
    assert len(observed) == len(expected)
    for lhs, rhs in zip(observed, expected, strict=True):
        assert lhs == pytest.approx(rhs)


class _FeatureAdapter:
    def __init__(self) -> None:
        self.embedding_layers: list[str] = []
        self.log_likelihood_reductions: list[str] = []

    def log_likelihood(self, seqs, *, method: str = "native", reduction: str = "sum"):
        assert method == "native"
        self.log_likelihood_reductions.extend([reduction] * len(seqs))
        if reduction == "sum":
            return [float(len(seq) - 1) for seq in seqs]
        return [1.0 if len(seq) > 1 else float("nan") for seq in seqs]

    def logits(self, seqs, *, fmt: str):
        assert fmt == "tensor"
        return [torch.arange(len(seq) * 2, dtype=torch.float32).reshape(len(seq), 2) for seq in seqs]

    def embedding(self, seqs, *, layer: str, fmt: str):
        assert fmt == "tensor"
        self.embedding_layers.append(layer)
        return [torch.arange(len(seq) * 3, dtype=torch.float32).reshape(len(seq), 3) for seq in seqs]


class _CombinedFeatureAdapter(_FeatureAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.combined_forward_calls = 0

    def logits(self, seqs, *, fmt: str):
        raise AssertionError("feature bundle should use fused logits/embedding path when available")

    def embedding(self, seqs, *, layer: str, fmt: str):
        raise AssertionError("feature bundle should use fused logits/embedding path when available")

    def logits_and_embedding(self, seqs, *, layer: str, fmt: str):
        assert fmt == "tensor"
        self.combined_forward_calls += 1
        self.embedding_layers.append(layer)
        logits = [torch.arange(len(seq) * 2, dtype=torch.float32).reshape(len(seq), 2) for seq in seqs]
        embeddings = [torch.arange(len(seq) * 3, dtype=torch.float32).reshape(len(seq), 3) for seq in seqs]
        return logits, embeddings


class _CountingFeatureAdapter(_FeatureAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.logits_call_count = 0
        self.embedding_call_count = 0

    def logits(self, seqs, *, fmt: str):
        self.logits_call_count += 1
        return super().logits(seqs, fmt=fmt)

    def embedding(self, seqs, *, layer: str, fmt: str):
        self.embedding_call_count += 1
        return super().embedding(seqs, layer=layer, fmt=fmt)


class _BatchSensitiveFeatureAdapter(_FeatureAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.log_likelihood_batch_sizes: list[tuple[str, int]] = []
        self.logits_batch_sizes: list[int] = []
        self.embedding_batch_sizes: list[int] = []

    def log_likelihood(self, seqs, *, method: str = "native", reduction: str = "sum"):
        assert method == "native"
        self.log_likelihood_batch_sizes.append((reduction, len(seqs)))
        if reduction == "sum":
            return [float(len(seqs) * 10 + idx) for idx, _seq in enumerate(seqs)]
        return [float(len(seqs)) + float(idx) / 100.0 for idx, _seq in enumerate(seqs)]

    def logits(self, seqs, *, fmt: str):
        assert fmt == "tensor"
        self.logits_batch_sizes.append(len(seqs))
        tensors = []
        for idx, seq in enumerate(seqs):
            base = float(1000 + len(seqs) + idx)
            tensors.append(torch.tensor([[base, base + 0.5]], dtype=torch.float32).repeat(len(seq), 1))
        return tensors

    def embedding(self, seqs, *, layer: str, fmt: str):
        assert fmt == "tensor"
        self.embedding_layers.append(layer)
        self.embedding_batch_sizes.append(len(seqs))
        tensors = []
        for idx, seq in enumerate(seqs):
            base = float(2000 + len(seqs) + idx)
            tensors.append(torch.tensor([[base, base + 0.5, base + 1.0]], dtype=torch.float32).repeat(len(seq), 1))
        return tensors


class _OOMOnFixedBatchFeatureAdapter(_FeatureAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def log_likelihood(self, seqs, *, method: str = "native", reduction: str = "sum"):
        assert method == "native"
        self.batch_sizes.append(len(seqs))
        raise RuntimeError("CUDA out of memory")


def _anchor_only_bundle():
    bundle = JobConfig(
        id="anchor_only_bundle",
        operation="extract",
        ingest={"source": "sequences"},
        feature_bundle={"context": {"kind": "anchor_only"}},
    ).feature_bundle
    assert bundle is not None
    return bundle


def test_run_extract_job_feature_bundle_anchor_only_executes_expected_outputs(monkeypatch) -> None:
    adapter = _FeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="anchor_only_bundle",
        operation="extract",
        ingest={"source": "sequences"},
        feature_bundle={"context": {"kind": "anchor_only"}},
    )

    out = run_extract_job(inputs=["ACGT"], model=model, job=job, progress_factory=None)

    assert sorted(out) == sorted(
        [
            "log_likelihood__mean_per_token",
            "log_likelihood__total",
            "metadata__anchor_end",
            "metadata__anchor_id",
            "metadata__anchor_start",
            "metadata__construct_version",
            "metadata__context_id",
            "metadata__context_kind",
            "metadata__derivation_id",
            "metadata__feature_request_digest",
            "metadata__feature_schema_version",
            "metadata__feature_vector_key",
            "metadata__forward_pass_key",
            "metadata__intermediate_block",
            "metadata__intermediate_selector",
            "metadata__is_wildtype",
            "metadata__model_name",
            "metadata__orientation",
            "metadata__parent_sequence_id",
            "metadata__pooling_end_0",
            "metadata__pooling_operation",
            "metadata__pooling_modes",
            "metadata__pooling_start_0",
            "metadata__product_kind",
            "metadata__provider_name",
            "metadata__provider_version",
            "metadata__resolved_length",
            "metadata__sequence_id",
            "metadata__template_id",
            "metadata__timestamp",
            "metadata__view_id",
            "metadata__view_name",
            "output_layer_mean__seq_mean",
            "intermediate_embedding__block26_mlp_out__seq_mean",
        ]
    )
    assert "output_layer_mean__anchor_mean" not in out
    assert "intermediate_embedding__block26_mlp_out__anchor_mean" not in out
    assert adapter.embedding_layers == ["block26_mlp_out"]
    assert adapter.log_likelihood_reductions == ["sum", "mean"]
    assert out["log_likelihood__total"] == [3.0]
    assert out["log_likelihood__mean_per_token"] == pytest.approx([1.0])
    _assert_list_close(out["output_layer_mean__seq_mean"][0], [3.0, 4.0])
    _assert_list_close(out["intermediate_embedding__block26_mlp_out__seq_mean"][0], [4.5, 5.5, 6.5])
    assert out["metadata__context_kind"] == ["anchor_only"]
    assert out["metadata__pooling_modes"] == [["seq_mean"]]
    assert out["metadata__intermediate_selector"] == ["block26_mlp_out"]
    assert out["metadata__forward_pass_key"][0]
    assert out["metadata__feature_vector_key"][0]


def test_run_extract_job_feature_bundle_uses_fused_adapter_paths_when_available(monkeypatch) -> None:
    adapter = _CombinedFeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)

    model = ModelConfig(id="evo2_20b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="templated_bundle_20b",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={"context": {"kind": "template_1kb"}},
    )
    records = [
        {
            "id": "wt-promoter-1",
            "sequence": "AAAACGTTTT",
            "construct__context_id": "construct:template_1kb:wt-promoter-1",
            "construct__template_id": "default_1kb",
            "construct__anchor_id": "wt-promoter-1",
            "construct__anchor_start": 4,
            "construct__anchor_end": 8,
            "construct__anchor_orientation": "forward",
            "construct__resolved_length": 10,
            "construct__spec_id": "construct-spec-v1",
            "is_wildtype": True,
        }
    ]

    out = run_extract_job(inputs=records, model=model, job=job, progress_factory=None)

    assert adapter.log_likelihood_reductions == ["sum", "mean"]
    assert adapter.combined_forward_calls == 1
    assert adapter.embedding_layers == ["block23_mlp_out"]
    assert out["log_likelihood__total"] == [9.0]
    assert out["log_likelihood__mean_per_token"] == pytest.approx([1.0])
    _assert_list_close(out["output_layer_mean__anchor_mean"][0], [11.0, 12.0])
    _assert_list_close(out["intermediate_embedding__block23_mlp_out__anchor_mean"][0], [16.5, 17.5, 18.5])


def test_run_extract_job_feature_bundle_fused_outputs_match_unfused_exactly(monkeypatch) -> None:
    records = [
        {
            "id": "wt-promoter-1",
            "sequence": "AAAACGTTTT",
            "construct__context_id": "construct:template_1kb:wt-promoter-1",
            "construct__template_id": "default_1kb",
            "construct__anchor_id": "wt-promoter-1",
            "construct__anchor_start": 4,
            "construct__anchor_end": 8,
            "construct__anchor_orientation": "forward",
            "construct__resolved_length": 10,
            "construct__spec_id": "construct-spec-v1",
            "is_wildtype": True,
        },
        {
            "id": "designed-promoter-2",
            "sequence": "AACCGGTT",
            "construct__context_id": "construct:template_1kb:designed-promoter-2",
            "construct__template_id": "default_1kb",
            "construct__anchor_id": "wt-promoter-1",
            "construct__anchor_start": 2,
            "construct__anchor_end": 6,
            "construct__anchor_orientation": "forward",
            "construct__resolved_length": 8,
            "construct__spec_id": "construct-spec-v1",
            "is_wildtype": False,
        },
    ]
    model = ModelConfig(id="evo2_20b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="templated_bundle_20b",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={"context": {"kind": "template_1kb"}},
    )

    separate_adapter = _FeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: separate_adapter)
    expected = run_extract_job(inputs=records, model=model, job=job, progress_factory=None)

    fused_adapter = _CombinedFeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: fused_adapter)
    observed = run_extract_job(inputs=records, model=model, job=job, progress_factory=None)

    feature_keys = [
        "log_likelihood__total",
        "log_likelihood__mean_per_token",
        "output_layer_mean__seq_mean",
        "output_layer_mean__anchor_mean",
        "intermediate_embedding__block23_mlp_out__seq_mean",
        "intermediate_embedding__block23_mlp_out__anchor_mean",
    ]
    for out_id in feature_keys:
        assert observed[out_id] == expected[out_id]
    assert observed["metadata__feature_request_digest"] == expected["metadata__feature_request_digest"]
    assert fused_adapter.combined_forward_calls == 1
    assert fused_adapter.log_likelihood_reductions == ["sum", "sum", "mean", "mean"]
    assert separate_adapter.log_likelihood_reductions == ["sum", "sum", "mean", "mean"]


def test_run_extract_job_feature_bundle_anchor_only_20b_uses_model_specific_selector(monkeypatch) -> None:
    adapter = _FeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)

    model = ModelConfig(id="evo2_20b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="anchor_only_bundle_20b",
        operation="extract",
        ingest={"source": "sequences"},
        feature_bundle={"context": {"kind": "anchor_only"}},
    )

    out = run_extract_job(inputs=["ACGT"], model=model, job=job, progress_factory=None)

    assert "intermediate_embedding__block23_mlp_out__seq_mean" in out
    assert "intermediate_embedding__block26_mlp_out__seq_mean" not in out
    assert adapter.embedding_layers == ["block23_mlp_out"]
    assert adapter.log_likelihood_reductions == ["sum", "mean"]
    assert out["metadata__intermediate_selector"] == ["block23_mlp_out"]


def test_run_extract_job_feature_bundle_anchor_only_20b_preserves_configured_eval_batch_size_for_small_inputs(
    monkeypatch,
) -> None:
    adapter = _BatchSensitiveFeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)

    model = ModelConfig(id="evo2_20b", device="cpu", precision="fp32", alphabet="dna", batch_size=256)
    job = JobConfig(
        id="anchor_only_20b_features",
        operation="extract",
        ingest={"source": "sequences"},
        feature_bundle={"context": {"kind": "anchor_only"}},
    )

    out = run_extract_job(inputs=["ACGT", "GGGG", "TTTT"], model=model, job=job, progress_factory=None)

    assert adapter.log_likelihood_batch_sizes == [("sum", 256), ("mean", 256)]
    assert adapter.logits_batch_sizes == [256]
    assert adapter.embedding_batch_sizes == [256]
    assert out["log_likelihood__total"] == [2560.0, 2561.0, 2562.0]
    assert out["log_likelihood__mean_per_token"] == pytest.approx([256.0, 256.01, 256.02])
    _assert_list_close(out["output_layer_mean__seq_mean"][0], [1256.0, 1256.5])
    _assert_list_close(out["intermediate_embedding__block23_mlp_out__seq_mean"][0], [2256.0, 2256.5, 2257.0])


def test_existing_feature_metadata_values_reads_usr_metadata_columns_in_single_scan(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_read_usr_columns(*, ds, ids, column_names):
        captured["ds"] = ds
        captured["ids"] = list(ids)
        captured["column_names"] = list(column_names)
        return {column_name: [column_name] for column_name in column_names}

    monkeypatch.setattr("dnadesign.infer.src.features.execution.read_usr_columns", _fake_read_usr_columns)

    values = _existing_feature_metadata_values(
        ds="dataset-handle",
        ids=["row-1"],
        model_id="evo2_20b",
        job_id="template_1kb_20b_features",
    )

    expected_columns = [
        infer_usr_column_name(
            model_id="evo2_20b",
            job_id="template_1kb_20b_features",
            out_id=out_id,
        )
        for out_id in feature_metadata_output_ids()
    ]
    assert captured["ds"] == "dataset-handle"
    assert captured["ids"] == ["row-1"]
    assert captured["column_names"] == expected_columns
    assert values == {
        out_id: [column_name]
        for out_id, column_name in zip(feature_metadata_output_ids(), expected_columns, strict=True)
    }


def test_apply_digest_resume_guard_uses_prefetched_digests_without_usr_rescan(monkeypatch) -> None:
    def _fail_read(*args, **kwargs):
        raise AssertionError("prefetched digest path should not rescan USR columns")

    monkeypatch.setattr("dnadesign.infer.src.features.execution.read_usr_columns", _fail_read)
    feature_values = {"log_likelihood__total": [3.0]}

    stale_idx = _apply_digest_resume_guard(
        ds="dataset-handle",
        ids=["row-1"],
        model_id="evo2_20b",
        job_id="template_1kb_20b_features",
        feature_values=feature_values,
        metadata_columnar={"metadata__feature_request_digest": ["digest-expected"]},
        existing_digests=["digest-stale"],
    )

    assert stale_idx == [0]
    assert feature_values["log_likelihood__total"] == [None]


def test_run_extract_job_feature_bundle_templated_records_compute_anchor_mean(monkeypatch) -> None:
    adapter = _FeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)

    records = [
        {
            "id": "wt-promoter-1",
            "sequence": "AAAACGTTTT",
            "construct__context_id": "construct:template_1kb:wt-promoter-1",
            "construct__template_id": "default_1kb",
            "construct__anchor_id": "wt-promoter-1",
            "construct__anchor_start": 4,
            "construct__anchor_end": 8,
            "construct__anchor_orientation": "forward",
            "construct__resolved_length": 10,
            "construct__spec_id": "construct-spec-v1",
            "is_wildtype": True,
        }
    ]
    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="templated_bundle",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={"context": {"kind": "template_1kb"}},
    )

    out = run_extract_job(inputs=records, model=model, job=job, progress_factory=None)

    logits = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    embeddings = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    _assert_list_close(out["output_layer_mean__seq_mean"][0], logits.mean(dim=0).tolist())
    _assert_list_close(out["output_layer_mean__anchor_mean"][0], logits[4:8].mean(dim=0).tolist())
    _assert_list_close(out["intermediate_embedding__block26_mlp_out__seq_mean"][0], embeddings.mean(dim=0).tolist())
    _assert_list_close(
        out["intermediate_embedding__block26_mlp_out__anchor_mean"][0], embeddings[4:8].mean(dim=0).tolist()
    )
    assert out["metadata__is_wildtype"] == [True]
    assert out["metadata__context_kind"] == ["template_1kb"]
    assert out["metadata__template_id"] == ["default_1kb"]
    assert out["metadata__anchor_start"] == [4]
    assert out["metadata__anchor_end"] == [8]
    assert out["metadata__pooling_modes"] == [["seq_mean", "anchor_mean"]]


def test_run_extract_job_feature_bundle_templated_can_disable_seq_mean(monkeypatch) -> None:
    adapter = _FeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)

    records = [
        {
            "id": "wt-promoter-1",
            "sequence": "AAAACGTTTT",
            "construct__context_id": "construct:template_1kb:wt-promoter-1",
            "construct__template_id": "default_1kb",
            "construct__anchor_id": "wt-promoter-1",
            "construct__anchor_start": 4,
            "construct__anchor_end": 8,
            "construct__anchor_orientation": "forward",
            "construct__resolved_length": 10,
            "construct__spec_id": "construct-spec-v1",
            "is_wildtype": True,
        }
    ]
    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="templated_bundle_anchor_only_pool",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={
            "context": {"kind": "template_1kb"},
            "pooling": {"seq_mean": False, "anchor_mean_for_templated": True},
        },
    )

    out = run_extract_job(inputs=records, model=model, job=job, progress_factory=None)

    logits = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    embeddings = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    assert "output_layer_mean__seq_mean" not in out
    assert "intermediate_embedding__block26_mlp_out__seq_mean" not in out
    _assert_list_close(out["output_layer_mean__anchor_mean"][0], logits[4:8].mean(dim=0).tolist())
    _assert_list_close(
        out["intermediate_embedding__block26_mlp_out__anchor_mean"][0], embeddings[4:8].mean(dim=0).tolist()
    )
    assert out["metadata__pooling_modes"] == [["anchor_mean"]]


def test_resolve_sequence_contexts_usr_reads_filtered_columns_without_full_scan(monkeypatch) -> None:
    bundle = JobConfig(
        id="templated_usr_contexts",
        operation="extract",
        ingest={"source": "usr", "dataset": "promoter/test"},
        feature_bundle={"context": {"kind": "template_1kb"}},
    ).feature_bundle
    assert bundle is not None

    ids = ["row-2", "row-1"]
    seqs = ["CCCCAAAA", "AAAACCCC"]
    payload = {
        "id": ["row-2", "row-1"],
        "construct__context_id": ["context-2", "context-1"],
        "construct__template_id": ["template-b", "template-a"],
        "construct__anchor_start": [1, 2],
        "construct__anchor_end": [5, 6],
        "construct__anchor_id": ["anchor-2", "anchor-1"],
        "construct__input_id": [None, None],
        "construct__anchor_orientation": ["reverse", "forward"],
        "construct__resolved_length": [8, 8],
        "construct__spec_id": ["construct-v2", "construct-v1"],
        "is_wildtype": [False, True],
    }
    query_kwargs: dict[str, object] = {}

    class _FakeBatch:
        num_rows = 2

        def to_pydict(self):
            return payload

    class _FakeConnection:
        def __init__(self) -> None:
            self.closed = False
            self.executed_query: str | None = None
            self.executed_params: list[str] | None = None

        def execute(self, query, params):
            self.executed_query = str(query)
            self.executed_params = list(params)

        def fetch_record_batch(self, batch_size):
            query_kwargs["batch_size"] = int(batch_size)
            return [_FakeBatch()]

        def close(self):
            self.closed = True

    fake_con = _FakeConnection()
    fake_ds = SimpleNamespace()
    fake_ds.scan = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("full dataset scan should not be used"))

    def _fake_duckdb_query(*, columns, include_overlays, include_deleted, where, params, limit):
        query_kwargs["columns"] = list(columns)
        query_kwargs["include_overlays"] = bool(include_overlays)
        query_kwargs["include_deleted"] = bool(include_deleted)
        query_kwargs["where"] = str(where)
        query_kwargs["params"] = list(params)
        query_kwargs["limit"] = limit
        return fake_con, "SELECT * FROM mock", list(params)

    fake_ds._duckdb_query = _fake_duckdb_query

    contexts = resolve_sequence_contexts(
        seqs=seqs,
        source="usr",
        ids=ids,
        records=None,
        ds=fake_ds,
        bundle=bundle,
    )

    assert query_kwargs["columns"] == [
        "id",
        "construct__context_id",
        "construct__template_id",
        "construct__anchor_start",
        "construct__anchor_end",
        "construct__anchor_id",
        "construct__input_id",
        "construct__anchor_orientation",
        "construct__resolved_length",
        "construct__spec_id",
        "is_wildtype",
    ]
    assert query_kwargs["include_overlays"] is True
    assert query_kwargs["include_deleted"] is False
    assert query_kwargs["params"] == ids
    assert query_kwargs["limit"] == len(ids)
    assert query_kwargs["batch_size"] == len(ids)
    assert fake_con.executed_query == "SELECT * FROM mock"
    assert fake_con.executed_params == ids
    assert fake_con.closed is True
    assert [context.sequence_id for context in contexts] == ids
    assert [context.context_id for context in contexts] == ["context-2", "context-1"]
    assert [context.anchor_id for context in contexts] == ["anchor-2", "anchor-1"]
    assert [context.anchor_start for context in contexts] == [1, 2]
    assert [context.anchor_end for context in contexts] == [5, 6]
    assert [context.template_id for context in contexts] == ["template-b", "template-a"]
    assert [context.is_wildtype for context in contexts] == [False, True]


def test_run_extract_job_feature_bundle_templated_requires_construct_metadata(monkeypatch) -> None:
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: _FeatureAdapter())

    records = [
        {
            "id": "designed-promoter-1",
            "sequence": "AAAACGTTTT",
            "construct__context_id": "construct:template_1kb:designed-promoter-1",
            "construct__template_id": "default_1kb",
            "construct__anchor_start": 4,
        }
    ]
    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="templated_missing_metadata",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={"context": {"kind": "template_1kb"}},
    )

    with pytest.raises(CapabilityError, match="construct metadata columns"):
        run_extract_job(inputs=records, model=model, job=job, progress_factory=None)


def test_run_extract_job_feature_bundle_sequence_views_deduplicates_alias_equivalent_core60_and_seq_mean(
    monkeypatch,
    tmp_path,
) -> None:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "reference_views")
    dataset.init(source="test", notes="sequence-view infer test")
    sequence = "ACGT" * 15
    add_result = dataset.add_sequences([sequence], bio_type="dna", alphabet="dna_4", source="test")
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="core60_view",
                aliases=["core60_alias"],
                product_kind="analysis_window",
                context_kind="analysis_window",
                orientation="forward",
                analysis_only=True,
                source_dataset_id=dataset.name,
                parent_sequence_id=add_result.ids[0],
                parent_dataset_id=dataset.name,
                derivation_id="construct:core60",
                derivation_spec_id="construct:core60",
                source_interval_start_0=0,
                source_interval_end_0=60,
                anchor_start_0=0,
                anchor_end_0=60,
                forward_anchor_start_0=0,
                forward_anchor_end_0=60,
                recommended_pooling="core60_mean",
                created_at="2026-04-25T00:00:00+00:00",
                created_by="test",
            ),
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="construct_insert60_view",
                aliases=["construct_insert60_alias"],
                product_kind="construct_insert",
                context_kind="anchor_only",
                orientation="forward",
                analysis_only=False,
                source_dataset_id=dataset.name,
                source_interval_start_0=0,
                source_interval_end_0=60,
                anchor_start_0=0,
                anchor_end_0=60,
                forward_anchor_start_0=0,
                forward_anchor_end_0=60,
                recommended_pooling="seq_mean",
                created_at="2026-04-25T00:00:00+00:00",
                created_by="test",
            ),
        ],
        conflict_policy="error",
    )

    adapter = _CountingFeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)

    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="reference_view_bundle",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={
            "collect_log_likelihood": False,
            "sequence_view_inputs": [
                {
                    "dataset": "reference_views",
                    "root": usr_root.as_posix(),
                    "view_selector": {"product_kind": "analysis_window"},
                    "pooling": {"operation": "core60_mean"},
                },
                {
                    "dataset": "reference_views",
                    "root": usr_root.as_posix(),
                    "view_selector": {"view_name": "construct_insert60_view"},
                    "pooling": {"operation": "seq_mean"},
                },
            ],
        },
    )

    out = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)

    assert adapter.logits_call_count == 1
    assert adapter.embedding_call_count == 1
    assert len(out["metadata__view_id"]) == 2
    assert out["metadata__forward_pass_key"][0] == out["metadata__forward_pass_key"][1]
    assert out["metadata__feature_vector_key"][0] == out["metadata__feature_vector_key"][1]
    assert out["metadata__context_kind"] == ["analysis_window", "anchor_only"]
    assert out["metadata__product_kind"] == ["analysis_window", "construct_insert"]
    assert out["metadata__pooling_operation"] == ["core60_mean", "seq_mean"]
    assert out["output_layer_mean__core60_mean"][0] is not None
    assert out["output_layer_mean__core60_mean"][1] is None
    assert out["output_layer_mean__seq_mean"][0] is None
    assert out["output_layer_mean__seq_mean"][1] is not None
    assert out["intermediate_embedding__block26_mlp_out__core60_mean"][0] is not None
    assert out["intermediate_embedding__block26_mlp_out__seq_mean"][1] is not None

    alias_table = pq.read_table(dataset.dir / FEATURE_ALIAS_RELATIVE_PATH).to_pylist()
    assert len(alias_table) == 4
    assert {row["representation_kind"] for row in alias_table} == {"output_layer_mean", "intermediate_embedding"}
    assert len({row["feature_vector_key"] for row in alias_table}) == 2


def test_run_extract_job_feature_bundle_sequence_view_deduplicate_flags_control_execution_grouping(
    monkeypatch,
    tmp_path,
) -> None:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "reference_views")
    dataset.init(source="test", notes="sequence-view infer deduplicate flag test")
    sequence = "ACGT" * 15
    add_result = dataset.add_sequences([sequence], bio_type="dna", alphabet="dna_4", source="test")
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="core60_view_a",
                product_kind="analysis_window",
                context_kind="analysis_window",
                orientation="forward",
                analysis_only=True,
                source_dataset_id=dataset.name,
                parent_sequence_id=add_result.ids[0],
                parent_dataset_id=dataset.name,
                derivation_id="construct:core60:a",
                derivation_spec_id="construct:core60:a",
                source_interval_start_0=0,
                source_interval_end_0=60,
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="core60_mean",
                created_at="2026-04-25T00:00:00+00:00",
                created_by="test",
            ),
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="core60_view_b",
                product_kind="source_record",
                context_kind="native_reference",
                orientation="forward",
                analysis_only=False,
                source_dataset_id=dataset.name,
                source_interval_start_0=0,
                source_interval_end_0=60,
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="seq_mean",
                created_at="2026-04-25T00:00:00+00:00",
                created_by="test",
            ),
        ],
        conflict_policy="error",
    )

    adapter = _BatchSensitiveFeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)
    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="reference_view_bundle_no_dedup",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={
            "collect_log_likelihood": False,
            "sequence_view_inputs": [
                {
                    "dataset": "reference_views",
                    "root": usr_root.as_posix(),
                    "view_selector": {"view_name": "core60_view_a"},
                    "pooling": {"operation": "core60_mean"},
                },
                {
                    "dataset": "reference_views",
                    "root": usr_root.as_posix(),
                    "view_selector": {"view_name": "core60_view_b"},
                    "pooling": {"operation": "seq_mean"},
                },
            ],
            "deduplicate": {
                "by_forward_pass_key": False,
                "by_feature_vector_key": False,
                "write_alias_map": True,
            },
        },
    )

    run_extract_job(inputs=None, model=model, job=job, progress_factory=None)

    assert adapter.logits_batch_sizes == [2]
    assert adapter.embedding_batch_sizes == [2]


def test_run_extract_job_sequence_view_anchor_mean_uses_full_context_and_emitted_orientation_bounds(
    monkeypatch,
    tmp_path,
) -> None:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "context_views")
    dataset.init(source="test", notes="sequence-view reverse-complement pooling test")
    forward_sequence = "AAAATTTTAGTCGGGG"
    reverse_complement_sequence = str(Seq(forward_sequence).reverse_complement())
    assert reverse_complement_sequence == "CCCCGACTAAAATTTT"
    add_result = dataset.add_sequences(
        [forward_sequence, reverse_complement_sequence],
        bio_type="dna",
        alphabet="dna_4",
        source="test",
    )
    forward_id, reverse_complement_id = add_result.ids
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=forward_id,
                view_name="context_forward",
                product_kind="realized_context",
                context_kind="template_1kb",
                orientation="forward",
                analysis_only=False,
                source_dataset_id=dataset.name,
                anchor_start_0=8,
                anchor_end_0=12,
                forward_anchor_start_0=8,
                forward_anchor_end_0=12,
                recommended_pooling="anchor_mean",
                created_at="2026-04-27T00:00:00+00:00",
                created_by="test",
            ),
            SequenceViewRecord(
                sequence_id=reverse_complement_id,
                view_name="context_reverse_complement",
                product_kind="realized_context",
                context_kind="template_1kb",
                orientation="reverse_complement",
                analysis_only=False,
                source_dataset_id=dataset.name,
                anchor_start_0=4,
                anchor_end_0=8,
                forward_anchor_start_0=8,
                forward_anchor_end_0=12,
                recommended_pooling="anchor_mean",
                created_at="2026-04-27T00:00:00+00:00",
                created_by="test",
            ),
        ],
        conflict_policy="error",
    )

    adapter = _CountingFeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)
    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="context_view_anchor_mean",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={
            "collect_log_likelihood": True,
            "sequence_view_inputs": [
                {
                    "dataset": "context_views",
                    "root": usr_root.as_posix(),
                    "view_selector": {"product_kind": "realized_context", "orientation": "forward"},
                    "pooling": {"operation": "anchor_mean", "bounds_from": "sequence_view"},
                },
                {
                    "dataset": "context_views",
                    "root": usr_root.as_posix(),
                    "view_selector": {"product_kind": "realized_context", "orientation": "reverse_complement"},
                    "pooling": {"operation": "anchor_mean", "bounds_from": "sequence_view"},
                },
            ],
        },
    )

    out = run_extract_job(inputs=None, model=model, job=job, progress_factory=None)

    assert adapter.logits_call_count == 1
    assert adapter.embedding_call_count == 1
    assert out["metadata__orientation"] == ["forward", "reverse_complement"]
    assert out["metadata__resolved_length"] == [16, 16]
    assert out["metadata__pooling_operation"] == ["anchor_mean", "anchor_mean"]
    assert out["metadata__pooling_start_0"] == [8, 4]
    assert out["metadata__pooling_end_0"] == [12, 8]
    assert out["log_likelihood__total"] == [15.0, 15.0]
    _assert_list_close(out["output_layer_mean__anchor_mean"][0], [19.0, 20.0])
    _assert_list_close(out["output_layer_mean__anchor_mean"][1], [11.0, 12.0])
    _assert_list_close(out["intermediate_embedding__block26_mlp_out__anchor_mean"][0], [28.5, 29.5, 30.5])
    _assert_list_close(out["intermediate_embedding__block26_mlp_out__anchor_mean"][1], [16.5, 17.5, 18.5])


def test_run_extract_job_feature_bundle_sequence_view_alias_map_is_idempotent(
    monkeypatch,
    tmp_path,
) -> None:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "reference_views")
    dataset.init(source="test", notes="sequence-view infer idempotency test")
    sequence = "ACGT" * 15
    add_result = dataset.add_sequences([sequence], bio_type="dna", alphabet="dna_4", source="test")
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="core60_view",
                product_kind="analysis_window",
                context_kind="analysis_window",
                orientation="forward",
                analysis_only=True,
                source_dataset_id=dataset.name,
                parent_sequence_id=add_result.ids[0],
                parent_dataset_id=dataset.name,
                derivation_id="construct:core60",
                derivation_spec_id="construct:core60",
                source_interval_start_0=0,
                source_interval_end_0=60,
                anchor_start_0=0,
                anchor_end_0=60,
                forward_anchor_start_0=0,
                forward_anchor_end_0=60,
                recommended_pooling="core60_mean",
                created_at="2026-04-25T00:00:00+00:00",
                created_by="test",
            )
        ],
        conflict_policy="error",
    )

    adapter = _CountingFeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)
    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="reference_view_bundle",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={
            "collect_log_likelihood": False,
            "sequence_view_inputs": [
                {
                    "dataset": "reference_views",
                    "root": usr_root.as_posix(),
                    "view_selector": {"product_kind": "analysis_window"},
                    "pooling": {"operation": "core60_mean"},
                }
            ],
        },
    )

    run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert adapter.logits_call_count == 1
    assert adapter.embedding_call_count == 1

    run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    assert adapter.logits_call_count == 1
    assert adapter.embedding_call_count == 1

    alias_table = pq.read_table(dataset.dir / FEATURE_ALIAS_RELATIVE_PATH).to_pylist()
    assert len(alias_table) == 2


def test_run_extract_job_feature_bundle_sequence_view_alias_map_tolerates_view_name_drift(
    monkeypatch,
    tmp_path,
) -> None:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "reference_views")
    dataset.init(source="test", notes="sequence-view infer view-name drift test")
    sequence = "ACGT" * 15
    add_result = dataset.add_sequences([sequence], bio_type="dna", alphabet="dna_4", source="test")
    original_view = SequenceViewRecord(
        sequence_id=add_result.ids[0],
        view_name="core60_view",
        product_kind="analysis_window",
        context_kind="analysis_window",
        orientation="forward",
        analysis_only=True,
        source_dataset_id=dataset.name,
        parent_sequence_id=add_result.ids[0],
        parent_dataset_id=dataset.name,
        derivation_id="construct:core60",
        derivation_spec_id="construct:core60",
        source_interval_start_0=0,
        source_interval_end_0=60,
        anchor_start_0=0,
        anchor_end_0=60,
        forward_anchor_start_0=0,
        forward_anchor_end_0=60,
        recommended_pooling="core60_mean",
        created_at="2026-04-25T00:00:00+00:00",
        created_by="test",
    )
    write_sequence_views(dataset, [original_view], conflict_policy="error")

    adapter = _CountingFeatureAdapter()
    monkeypatch.setattr("dnadesign.infer.src.engine._get_adapter", lambda _model: adapter)
    model = ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")
    job = JobConfig(
        id="reference_view_bundle",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={
            "collect_log_likelihood": False,
            "sequence_view_inputs": [
                {
                    "dataset": "reference_views",
                    "root": usr_root.as_posix(),
                    "view_selector": {"product_kind": "analysis_window"},
                    "pooling": {"operation": "core60_mean"},
                }
            ],
        },
    )

    run_extract_job(inputs=None, model=model, job=job, progress_factory=None)
    renamed_view = original_view.model_copy(update={"view_name": "core60_view_renamed"})
    write_sequence_views(dataset, [renamed_view], conflict_policy="replace")
    run_extract_job(inputs=None, model=model, job=job, progress_factory=None)

    alias_table = pq.read_table(dataset.dir / FEATURE_ALIAS_RELATIVE_PATH).to_pylist()
    assert len(alias_table) == 2
    assert {row["view_name"] for row in alias_table} == {"core60_view_renamed"}
    assert adapter.logits_call_count == 1
    assert adapter.embedding_call_count == 1


def test_export_evo2_promoter_opal_matrix_keeps_deterministic_feature_order() -> None:
    payload = export_evo2_promoter_opal_matrix(
        row_ids=["row-1"],
        model_id="evo2_7b",
        bundle={"context": {"kind": "template_1kb"}},
        columnar={
            "log_likelihood__total": [1.5],
            "log_likelihood__mean_per_token": [0.5],
            "output_layer_mean__seq_mean": [[10.0, 11.0]],
            "output_layer_mean__anchor_mean": [[12.0, 13.0]],
            "intermediate_embedding__block26_mlp_out__seq_mean": [[20.0, 21.0, 22.0]],
            "intermediate_embedding__block26_mlp_out__anchor_mean": [[23.0, 24.0, 25.0]],
        },
    )

    assert payload["row_ids"] == ["row-1"]
    assert payload["feature_names"] == [
        "infer.evo2.evo2_7b.template_1kb.log_likelihood.total",
        "infer.evo2.evo2_7b.template_1kb.log_likelihood.mean_per_token",
        "infer.evo2.evo2_7b.template_1kb.output_layer_mean.seq_mean[0]",
        "infer.evo2.evo2_7b.template_1kb.output_layer_mean.seq_mean[1]",
        "infer.evo2.evo2_7b.template_1kb.output_layer_mean.anchor_mean[0]",
        "infer.evo2.evo2_7b.template_1kb.output_layer_mean.anchor_mean[1]",
        "infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.seq_mean[0]",
        "infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.seq_mean[1]",
        "infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.seq_mean[2]",
        "infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.anchor_mean[0]",
        "infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.anchor_mean[1]",
        "infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.anchor_mean[2]",
    ]
    assert payload["x"] == [[1.5, 0.5, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]]


def test_export_evo2_promoter_opal_matrix_skips_seq_mean_when_disabled() -> None:
    payload = export_evo2_promoter_opal_matrix(
        row_ids=["row-1"],
        model_id="evo2_7b",
        bundle={
            "context": {"kind": "template_1kb"},
            "pooling": {"seq_mean": False, "anchor_mean_for_templated": True},
        },
        columnar={
            "log_likelihood__total": [1.5],
            "log_likelihood__mean_per_token": [0.5],
            "output_layer_mean__anchor_mean": [[12.0, 13.0]],
            "intermediate_embedding__block26_mlp_out__anchor_mean": [[23.0, 24.0, 25.0]],
        },
    )

    assert payload["feature_names"] == [
        "infer.evo2.evo2_7b.template_1kb.log_likelihood.total",
        "infer.evo2.evo2_7b.template_1kb.log_likelihood.mean_per_token",
        "infer.evo2.evo2_7b.template_1kb.output_layer_mean.anchor_mean[0]",
        "infer.evo2.evo2_7b.template_1kb.output_layer_mean.anchor_mean[1]",
        "infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.anchor_mean[0]",
        "infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.anchor_mean[1]",
        "infer.evo2.evo2_7b.template_1kb.intermediate_embedding.block26_mlp_out.anchor_mean[2]",
    ]
    assert payload["x"] == [[1.5, 0.5, 12.0, 13.0, 23.0, 24.0, 25.0]]


def test_execute_feature_bundle_recomputes_usr_rows_when_digest_mismatches(monkeypatch) -> None:
    adapter = _FeatureAdapter()
    bundle = _anchor_only_bundle()
    digest_column = infer_usr_column_name(
        model_id="evo2_7b",
        job_id="digest_guard_bundle",
        out_id="metadata__feature_request_digest",
    )

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution.read_usr_columns",
        lambda **kwargs: {
            column_name: (["stale-digest"] if column_name == digest_column else [None])
            for column_name in kwargs["column_names"]
        },
    )

    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle)]
    existing = {out_id: [[999.0, 999.0]] if out_id.endswith("seq_mean") else [999.0] for out_id in feature_out_ids}
    existing["intermediate_embedding__block26_mlp_out__seq_mean"] = [[999.0, 999.0, 999.0]]

    progress: list[int] = []
    output_writers = {out_id: None for out_id in feature_out_ids}
    metadata_writers = {out_id: None for out_id in feature_metadata_output_ids()}

    columnar, metadata_rows = execute_feature_bundle(
        seqs=["ACGT"],
        source="usr",
        ids=["row-1"],
        records=None,
        ds=SimpleNamespace(records_path="unused"),
        model_id="evo2_7b",
        job_id="digest_guard_bundle",
        bundle=bundle,
        existing=existing,
        need_idx=[],
        adapter=adapter,
        micro_batch_size=1,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=progress.append,
        on_chunk_by_output=output_writers,
        on_chunk_by_metadata=metadata_writers,
    )

    assert progress == [1]
    assert metadata_rows[0]["feature_request_digest"] != "stale-digest"
    assert adapter.log_likelihood_reductions == ["sum", "mean"]
    assert columnar["log_likelihood__total"] == [3.0]
    assert columnar["log_likelihood__mean_per_token"] == pytest.approx([1.0])
    _assert_list_close(columnar["output_layer_mean__seq_mean"][0], [3.0, 4.0])
    _assert_list_close(columnar["intermediate_embedding__block26_mlp_out__seq_mean"][0], [4.5, 5.5, 6.5])


def test_execute_feature_bundle_resume_writes_only_missing_feature_columns(monkeypatch) -> None:
    adapter = _FeatureAdapter()
    bundle = _anchor_only_bundle()
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle)]
    existing = {out_id: [None] for out_id in feature_out_ids}
    existing[_LOG_LIKELIHOOD_TOTAL] = [3.0]

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._apply_digest_resume_guard",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._existing_feature_metadata_values",
        lambda **_kwargs: {out_id: ["present"] for out_id in feature_metadata_output_ids()},
    )

    output_calls: dict[str, list[tuple[list[int], list[object], bool | None]]] = {
        out_id: [] for out_id in feature_out_ids
    }

    def _writer_for(out_id: str):
        def _writer(idx_chunk, values, *, overwrite_override=None, progress=None):
            output_calls[out_id].append((list(idx_chunk), list(values), overwrite_override))

        return _writer

    columnar, _metadata_rows = execute_feature_bundle(
        seqs=["ACGT"],
        source="usr",
        ids=["row-1"],
        records=None,
        ds=SimpleNamespace(records_path="unused"),
        model_id="evo2_7b",
        job_id="resume_bundle",
        bundle=bundle,
        existing=existing,
        need_idx=[0],
        adapter=adapter,
        micro_batch_size=1,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=lambda _count: None,
        on_chunk_by_output={out_id: _writer_for(out_id) for out_id in feature_out_ids},
        on_chunk_by_metadata={out_id: None for out_id in feature_metadata_output_ids()},
    )

    assert output_calls[_LOG_LIKELIHOOD_TOTAL] == []
    assert adapter.log_likelihood_reductions == ["sum", "mean"]
    assert output_calls[_LOG_LIKELIHOOD_MEAN] == [([0], [1.0], None)]
    assert output_calls[_OUTPUT_LAYER_SEQ_MEAN][0][2] is None
    assert output_calls["intermediate_embedding__block26_mlp_out__seq_mean"][0][2] is None
    assert columnar[_LOG_LIKELIHOOD_TOTAL] == [3.0]


def test_execute_feature_bundle_stale_digest_rewrites_features_with_overwrite(monkeypatch) -> None:
    adapter = _FeatureAdapter()
    bundle = _anchor_only_bundle()
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle)]
    existing = {out_id: [[999.0, 999.0]] if out_id.endswith("seq_mean") else [999.0] for out_id in feature_out_ids}
    existing["intermediate_embedding__block26_mlp_out__seq_mean"] = [[999.0, 999.0, 999.0]]

    def _force_stale(*, feature_values, **_kwargs):
        for values in feature_values.values():
            values[0] = None
        return [0]

    monkeypatch.setattr("dnadesign.infer.src.features.execution._apply_digest_resume_guard", _force_stale)
    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._existing_feature_metadata_values",
        lambda **_kwargs: {out_id: ["present"] for out_id in feature_metadata_output_ids()},
    )

    output_calls: dict[str, list[tuple[list[int], list[object], bool | None]]] = {
        out_id: [] for out_id in feature_out_ids
    }

    def _writer_for(out_id: str):
        def _writer(idx_chunk, values, *, overwrite_override=None, progress=None):
            output_calls[out_id].append((list(idx_chunk), list(values), overwrite_override))

        return _writer

    execute_feature_bundle(
        seqs=["ACGT"],
        source="usr",
        ids=["row-1"],
        records=None,
        ds=SimpleNamespace(records_path="unused"),
        model_id="evo2_7b",
        job_id="stale_bundle",
        bundle=bundle,
        existing=existing,
        need_idx=[],
        adapter=adapter,
        micro_batch_size=1,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=lambda _count: None,
        on_chunk_by_output={out_id: _writer_for(out_id) for out_id in feature_out_ids},
        on_chunk_by_metadata={out_id: None for out_id in feature_metadata_output_ids()},
    )

    assert all(len(calls) == 1 for calls in output_calls.values())
    assert all(calls[0][0] == [0] for calls in output_calls.values())
    assert all(calls[0][2] is True for calls in output_calls.values())


def test_execute_feature_bundle_backfills_missing_metadata_without_recomputing(monkeypatch) -> None:
    bundle = _anchor_only_bundle()
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle)]
    existing = {out_id: [[3.0, 4.0]] if out_id.endswith("seq_mean") else [3.0] for out_id in feature_out_ids}
    existing["intermediate_embedding__block26_mlp_out__seq_mean"] = [[4.5, 5.5, 6.5]]

    metadata_existing = {out_id: ["present"] for out_id in feature_metadata_output_ids()}
    metadata_existing["metadata__provider_version"] = [None]
    metadata_calls: dict[str, list[tuple[list[int], list[object], bool | None]]] = {
        out_id: [] for out_id in feature_metadata_output_ids()
    }
    progress: list[int] = []

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._apply_digest_resume_guard",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._existing_feature_metadata_values",
        lambda **_kwargs: metadata_existing,
    )

    def _writer_for(out_id: str):
        def _writer(idx_chunk, values, *, overwrite_override=None, progress=None):
            metadata_calls[out_id].append((list(idx_chunk), list(values), overwrite_override))

        return _writer

    adapter = _FeatureAdapter()
    columnar, _metadata_rows = execute_feature_bundle(
        seqs=["ACGT"],
        source="usr",
        ids=["row-1"],
        records=None,
        ds=SimpleNamespace(records_path="unused"),
        model_id="evo2_7b",
        job_id="metadata_backfill_bundle",
        bundle=bundle,
        existing=existing,
        need_idx=[],
        adapter=adapter,
        micro_batch_size=1,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=progress.append,
        on_chunk_by_output={out_id: None for out_id in feature_out_ids},
        on_chunk_by_metadata={out_id: _writer_for(out_id) for out_id in feature_metadata_output_ids()},
    )

    assert progress == []
    assert metadata_calls["metadata__provider_version"] == [([0], [None], None)]
    assert all(not calls for out_id, calls in metadata_calls.items() if out_id != "metadata__provider_version")
    assert columnar["metadata__provider_version"] == [None]


def test_execute_feature_bundle_anchor_only_20b_fixed_eval_batch_fails_fast_on_oom(monkeypatch) -> None:
    adapter = _OOMOnFixedBatchFeatureAdapter()
    bundle = _anchor_only_bundle()
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle, model_id="evo2_20b")]
    existing = {out_id: [None, None, None] for out_id in feature_out_ids}

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._apply_digest_resume_guard",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._existing_feature_metadata_values",
        lambda **_kwargs: {out_id: ["present", "present", "present"] for out_id in feature_metadata_output_ids()},
    )

    with pytest.raises(RuntimeOOMError, match="fixed evaluation batch size 256"):
        execute_feature_bundle(
            seqs=["ACGT", "GGGG", "TTTT"],
            source="sequences",
            ids=None,
            records=None,
            ds=None,
            model_id="evo2_20b",
            job_id="anchor_only_20b_features",
            bundle=bundle,
            existing=existing,
            need_idx=[0, 1, 2],
            adapter=adapter,
            micro_batch_size=256,
            default_batch_size=64,
            auto_derate=True,
            is_oom=lambda exc: "out of memory" in str(exc).lower(),
            on_progress=lambda _count: None,
            on_chunk_by_output={out_id: None for out_id in feature_out_ids},
            on_chunk_by_metadata={out_id: None for out_id in feature_metadata_output_ids()},
        )

    assert adapter.batch_sizes == [256]


def test_execute_feature_bundle_groups_metadata_chunk_writes_when_group_writer_is_available(monkeypatch) -> None:
    adapter = _FeatureAdapter()
    bundle = _anchor_only_bundle()
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle)]
    existing = {out_id: [None] for out_id in feature_out_ids}

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._apply_digest_resume_guard",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._existing_feature_metadata_values",
        lambda **_kwargs: {out_id: [None] for out_id in feature_metadata_output_ids()},
    )

    metadata_group_calls: list[tuple[list[int], dict[str, list[object]], bool | None, dict[str, object] | None]] = []

    def _metadata_group_writer(idx_chunk, columnar, *, overwrite_override=None, event_args=None):
        metadata_group_calls.append((list(idx_chunk), dict(columnar), overwrite_override, dict(event_args or {})))

    execute_feature_bundle(
        seqs=["ACGT"],
        source="usr",
        ids=["row-1"],
        records=None,
        ds=SimpleNamespace(records_path="unused"),
        model_id="evo2_7b",
        job_id="metadata_group_bundle",
        bundle=bundle,
        existing=existing,
        need_idx=[0],
        adapter=adapter,
        micro_batch_size=1,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=lambda _count: None,
        on_chunk_by_output={out_id: None for out_id in feature_out_ids},
        on_chunk_by_metadata={out_id: None for out_id in feature_metadata_output_ids()},
        on_chunk_metadata_group=_metadata_group_writer,
    )

    assert len(metadata_group_calls) == 1
    row_indexes, grouped_columnar, overwrite_override, event_args = metadata_group_calls[0]
    assert row_indexes == [0]
    assert overwrite_override is None
    assert event_args == {"infer_notify_suppress": True}
    assert sorted(grouped_columnar) == sorted(feature_metadata_output_ids())


def test_execute_feature_bundle_groups_feature_chunk_writes_when_group_writer_is_available(monkeypatch) -> None:
    adapter = _CombinedFeatureAdapter()
    bundle = JobConfig(
        id="templated_group_bundle",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={"context": {"kind": "template_1kb"}},
    ).feature_bundle
    assert bundle is not None
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle, model_id="evo2_20b")]
    existing = {out_id: [None] for out_id in feature_out_ids}

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._apply_digest_resume_guard",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._existing_feature_metadata_values",
        lambda **_kwargs: {out_id: ["present"] for out_id in feature_metadata_output_ids()},
    )

    output_group_calls: list[tuple[list[int], dict[str, list[object]], bool | None, dict[str, object] | None]] = []

    def _output_group_writer(idx_chunk, columnar, *, overwrite_override=None, event_args=None):
        output_group_calls.append((list(idx_chunk), dict(columnar), overwrite_override, dict(event_args or {})))

    execute_feature_bundle(
        seqs=["AAAACGTTTT"],
        source="records",
        ids=["row-1"],
        records=[
            {
                "id": "row-1",
                "sequence": "AAAACGTTTT",
                "construct__context_id": "construct:template_1kb:row-1",
                "construct__template_id": "default_1kb",
                "construct__anchor_id": "row-1",
                "construct__anchor_start": 4,
                "construct__anchor_end": 8,
                "construct__anchor_orientation": "forward",
                "construct__resolved_length": 10,
                "construct__spec_id": "construct-spec-v1",
                "is_wildtype": True,
            }
        ],
        ds=None,
        model_id="evo2_20b",
        job_id="templated_group_bundle",
        bundle=bundle,
        existing=existing,
        need_idx=[0],
        adapter=adapter,
        micro_batch_size=1,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=lambda _count: None,
        on_chunk_by_output={out_id: None for out_id in feature_out_ids},
        on_chunk_by_metadata={out_id: None for out_id in feature_metadata_output_ids()},
        on_chunk_output_group=_output_group_writer,
    )

    assert len(output_group_calls) == 1
    row_indexes, grouped_columnar, overwrite_override, event_args = output_group_calls[0]
    assert row_indexes == [0]
    assert overwrite_override is None
    assert sorted(grouped_columnar) == sorted(feature_out_ids)
    assert event_args is not None
    infer_progress = event_args.get("infer_progress")
    assert isinstance(infer_progress, dict)
    assert infer_progress["overall_target_units"] == 6
    assert infer_progress["overall_completed_units"] == 6


def test_execute_feature_bundle_combines_feature_and_metadata_chunk_writes_when_group_writer_is_available(
    monkeypatch,
) -> None:
    adapter = _CombinedFeatureAdapter()
    bundle = JobConfig(
        id="templated_group_bundle_with_metadata",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={"context": {"kind": "template_1kb"}},
    ).feature_bundle
    assert bundle is not None
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle, model_id="evo2_20b")]
    existing = {out_id: [None] for out_id in feature_out_ids}

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._apply_digest_resume_guard",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._existing_feature_metadata_values",
        lambda **_kwargs: {out_id: [None] for out_id in feature_metadata_output_ids()},
    )

    output_group_calls: list[tuple[list[int], dict[str, list[object]], bool | None, dict[str, object] | None]] = []
    metadata_group_calls: list[tuple[list[int], dict[str, list[object]], bool | None, dict[str, object] | None]] = []

    def _output_group_writer(idx_chunk, columnar, *, overwrite_override=None, event_args=None):
        output_group_calls.append((list(idx_chunk), dict(columnar), overwrite_override, dict(event_args or {})))

    def _metadata_group_writer(idx_chunk, columnar, *, overwrite_override=None, event_args=None):
        metadata_group_calls.append((list(idx_chunk), dict(columnar), overwrite_override, dict(event_args or {})))

    execute_feature_bundle(
        seqs=["AAAACGTTTT"],
        source="records",
        ids=["row-1"],
        records=[
            {
                "id": "row-1",
                "sequence": "AAAACGTTTT",
                "construct__context_id": "construct:template_1kb:row-1",
                "construct__template_id": "default_1kb",
                "construct__anchor_id": "row-1",
                "construct__anchor_start": 4,
                "construct__anchor_end": 8,
                "construct__anchor_orientation": "forward",
                "construct__resolved_length": 10,
                "construct__spec_id": "construct-spec-v1",
                "is_wildtype": True,
            }
        ],
        ds=None,
        model_id="evo2_20b",
        job_id="templated_group_bundle_with_metadata",
        bundle=bundle,
        existing=existing,
        need_idx=[0],
        adapter=adapter,
        micro_batch_size=1,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=lambda _count: None,
        on_chunk_by_output={out_id: None for out_id in feature_out_ids},
        on_chunk_by_metadata={out_id: None for out_id in feature_metadata_output_ids()},
        on_chunk_output_group=_output_group_writer,
        on_chunk_metadata_group=_metadata_group_writer,
    )

    assert len(output_group_calls) == 1
    row_indexes, grouped_columnar, overwrite_override, event_args = output_group_calls[0]
    assert row_indexes == [0]
    assert overwrite_override is None
    assert sorted(grouped_columnar) == sorted([*feature_out_ids, *feature_metadata_output_ids()])
    assert event_args is not None
    infer_progress = event_args.get("infer_progress")
    assert isinstance(infer_progress, dict)
    assert infer_progress["overall_target_units"] == 6
    assert infer_progress["overall_completed_units"] == 6
    assert metadata_group_calls == []


def test_execute_feature_bundle_grouped_output_path_skips_per_column_metadata_writers(monkeypatch) -> None:
    adapter = _CombinedFeatureAdapter()
    bundle = JobConfig(
        id="templated_group_bundle_with_metadata",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={"context": {"kind": "template_1kb"}},
    ).feature_bundle
    assert bundle is not None
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle, model_id="evo2_20b")]
    existing = {out_id: [None] for out_id in feature_out_ids}

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._apply_digest_resume_guard",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._existing_feature_metadata_values",
        lambda **_kwargs: {out_id: [None] for out_id in feature_metadata_output_ids()},
    )

    output_group_calls: list[tuple[list[int], dict[str, list[object]], bool | None, dict[str, object] | None]] = []
    metadata_calls: dict[str, list[tuple[list[int], list[object], bool | None]]] = {
        out_id: [] for out_id in feature_metadata_output_ids()
    }

    def _output_group_writer(idx_chunk, columnar, *, overwrite_override=None, event_args=None):
        output_group_calls.append((list(idx_chunk), dict(columnar), overwrite_override, dict(event_args or {})))

    def _metadata_writer(out_id: str):
        def _writer(idx_chunk, values, *, overwrite_override=None, progress=None):
            metadata_calls[out_id].append((list(idx_chunk), list(values), overwrite_override))

        return _writer

    execute_feature_bundle(
        seqs=["AAAACGTTTT"],
        source="records",
        ids=["row-1"],
        records=[
            {
                "id": "row-1",
                "sequence": "AAAACGTTTT",
                "construct__context_id": "construct:template_1kb:row-1",
                "construct__template_id": "default_1kb",
                "construct__anchor_id": "row-1",
                "construct__anchor_start": 4,
                "construct__anchor_end": 8,
                "construct__anchor_orientation": "forward",
                "construct__resolved_length": 10,
                "construct__spec_id": "construct-spec-v1",
                "is_wildtype": True,
            }
        ],
        ds=None,
        model_id="evo2_20b",
        job_id="templated_group_bundle_with_metadata",
        bundle=bundle,
        existing=existing,
        need_idx=[0],
        adapter=adapter,
        micro_batch_size=1,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=lambda _count: None,
        on_chunk_by_output={out_id: None for out_id in feature_out_ids},
        on_chunk_by_metadata={out_id: _metadata_writer(out_id) for out_id in feature_metadata_output_ids()},
        on_chunk_output_group=_output_group_writer,
    )

    assert len(output_group_calls) == 1
    assert sorted(output_group_calls[0][1]) == sorted([*feature_out_ids, *feature_metadata_output_ids()])
    assert all(not calls for calls in metadata_calls.values())


def test_execute_feature_bundle_grouped_output_path_handles_later_micro_batches(monkeypatch) -> None:
    adapter = _CombinedFeatureAdapter()
    bundle = JobConfig(
        id="templated_group_bundle_with_metadata",
        operation="extract",
        ingest={"source": "records", "field": "sequence"},
        feature_bundle={"context": {"kind": "template_1kb"}},
    ).feature_bundle
    assert bundle is not None
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle, model_id="evo2_20b")]
    existing = {out_id: [None, None] for out_id in feature_out_ids}

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._apply_digest_resume_guard",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution._existing_feature_metadata_values",
        lambda **_kwargs: {out_id: [None, None] for out_id in feature_metadata_output_ids()},
    )

    output_group_calls: list[tuple[list[int], dict[str, list[object]], bool | None, dict[str, object] | None]] = []

    def _output_group_writer(idx_chunk, columnar, *, overwrite_override=None, event_args=None):
        output_group_calls.append((list(idx_chunk), dict(columnar), overwrite_override, dict(event_args or {})))

    execute_feature_bundle(
        seqs=["AAAACGTTTT", "AACCGGTTAA"],
        source="records",
        ids=["row-1", "row-2"],
        records=[
            {
                "id": "row-1",
                "sequence": "AAAACGTTTT",
                "construct__context_id": "construct:template_1kb:row-1",
                "construct__template_id": "default_1kb",
                "construct__anchor_id": "row-1",
                "construct__anchor_start": 4,
                "construct__anchor_end": 8,
                "construct__anchor_orientation": "forward",
                "construct__resolved_length": 10,
                "construct__spec_id": "construct-spec-v1",
                "is_wildtype": True,
            },
            {
                "id": "row-2",
                "sequence": "AACCGGTTAA",
                "construct__context_id": "construct:template_1kb:row-2",
                "construct__template_id": "default_1kb",
                "construct__anchor_id": "row-2",
                "construct__anchor_start": 3,
                "construct__anchor_end": 7,
                "construct__anchor_orientation": "forward",
                "construct__resolved_length": 10,
                "construct__spec_id": "construct-spec-v1",
                "is_wildtype": False,
            },
        ],
        ds=None,
        model_id="evo2_20b",
        job_id="templated_group_bundle_with_metadata",
        bundle=bundle,
        existing=existing,
        need_idx=[0, 1],
        adapter=adapter,
        micro_batch_size=1,
        default_batch_size=64,
        auto_derate=True,
        is_oom=lambda _exc: False,
        on_progress=lambda _count: None,
        on_chunk_by_output={out_id: None for out_id in feature_out_ids},
        on_chunk_by_metadata={out_id: None for out_id in feature_metadata_output_ids()},
        on_chunk_output_group=_output_group_writer,
    )

    assert [call[0] for call in output_group_calls] == [[0], [1]]
    assert all(
        sorted(call[1]) == sorted([*feature_out_ids, *feature_metadata_output_ids()]) for call in output_group_calls
    )
