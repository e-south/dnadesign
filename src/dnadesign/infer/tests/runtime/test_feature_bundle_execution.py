"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_feature_bundle_execution.py

Runtime contract tests for Evo2 promoter feature bundles.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dnadesign.infer import export_evo2_promoter_opal_matrix
from dnadesign.infer.src.config import JobConfig, ModelConfig
from dnadesign.infer.src.engine import run_extract_job
from dnadesign.infer.src.errors import CapabilityError
from dnadesign.infer.src.features.execution import (
    _LOG_LIKELIHOOD_MEAN,
    _LOG_LIKELIHOOD_TOTAL,
    _OUTPUT_LAYER_SEQ_MEAN,
    build_feature_bundle_outputs,
    execute_feature_bundle,
    feature_metadata_output_ids,
)


def _assert_list_close(observed: list[float], expected: list[float]) -> None:
    assert len(observed) == len(expected)
    for lhs, rhs in zip(observed, expected, strict=True):
        assert lhs == pytest.approx(rhs)


class _FeatureAdapter:
    def __init__(self) -> None:
        self.embedding_layers: list[str] = []

    def log_likelihood(self, seqs, *, method: str = "native", reduction: str = "sum"):
        assert method == "native"
        if reduction == "sum":
            return [float(len(seq)) for seq in seqs]
        return [float(len(seq)) / 10.0 for seq in seqs]

    def logits(self, seqs, *, fmt: str):
        assert fmt == "tensor"
        return [torch.arange(len(seq) * 2, dtype=torch.float32).reshape(len(seq), 2) for seq in seqs]

    def embedding(self, seqs, *, layer: str, fmt: str):
        assert fmt == "tensor"
        self.embedding_layers.append(layer)
        return [torch.arange(len(seq) * 3, dtype=torch.float32).reshape(len(seq), 3) for seq in seqs]


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
            "metadata__feature_request_digest",
            "metadata__feature_schema_version",
            "metadata__intermediate_block",
            "metadata__intermediate_selector",
            "metadata__is_wildtype",
            "metadata__model_name",
            "metadata__pooling_modes",
            "metadata__provider_name",
            "metadata__provider_version",
            "metadata__resolved_length",
            "metadata__sequence_id",
            "metadata__template_id",
            "metadata__timestamp",
            "output_layer_mean__seq_mean",
            "intermediate_embedding__block26_mlp_out__seq_mean",
        ]
    )
    assert "output_layer_mean__anchor_mean" not in out
    assert "intermediate_embedding__block26_mlp_out__anchor_mean" not in out
    assert adapter.embedding_layers == ["block26_mlp_out"]
    assert out["log_likelihood__total"] == [4.0]
    assert out["log_likelihood__mean_per_token"] == [0.4]
    _assert_list_close(out["output_layer_mean__seq_mean"][0], [3.0, 4.0])
    _assert_list_close(out["intermediate_embedding__block26_mlp_out__seq_mean"][0], [4.5, 5.5, 6.5])
    assert out["metadata__context_kind"] == ["anchor_only"]
    assert out["metadata__pooling_modes"] == [["seq_mean"]]
    assert out["metadata__intermediate_selector"] == ["block26_mlp_out"]


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

    monkeypatch.setattr(
        "dnadesign.infer.src.features.execution.read_usr_column_values",
        lambda **_kwargs: ["stale-digest"],
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
    assert columnar["log_likelihood__total"] == [4.0]
    assert columnar["log_likelihood__mean_per_token"] == [0.4]
    _assert_list_close(columnar["output_layer_mean__seq_mean"][0], [3.0, 4.0])
    _assert_list_close(columnar["intermediate_embedding__block26_mlp_out__seq_mean"][0], [4.5, 5.5, 6.5])


def test_execute_feature_bundle_resume_writes_only_missing_feature_columns(monkeypatch) -> None:
    adapter = _FeatureAdapter()
    bundle = _anchor_only_bundle()
    feature_out_ids = [payload["id"] for payload in build_feature_bundle_outputs(bundle=bundle)]
    existing = {out_id: [None] for out_id in feature_out_ids}
    existing[_LOG_LIKELIHOOD_TOTAL] = [4.0]

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
    assert output_calls[_LOG_LIKELIHOOD_MEAN] == [([0], [0.4], None)]
    assert output_calls[_OUTPUT_LAYER_SEQ_MEAN][0][2] is None
    assert output_calls["intermediate_embedding__block26_mlp_out__seq_mean"][0][2] is None
    assert columnar[_LOG_LIKELIHOOD_TOTAL] == [4.0]


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
    existing = {out_id: [[3.0, 4.0]] if out_id.endswith("seq_mean") else [4.0] for out_id in feature_out_ids}
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
