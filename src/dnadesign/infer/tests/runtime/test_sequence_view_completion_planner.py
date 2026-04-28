"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_sequence_view_completion_planner.py

Sequence-view feature completion planner tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.infer.src.features import completion_planner
from dnadesign.infer.src.features.aliases import persist_feature_scalar_rows, persist_feature_vector_rows
from dnadesign.infer.src.features.completion_planner import plan_sequence_view_feature_completion
from dnadesign.infer.src.features.contracts import PromoterFeatureBundleConfig
from dnadesign.infer.src.features.execution import (
    _sequence_view_feature_scalar_specs,
    _sequence_view_feature_vector_specs,
    build_feature_metadata_rows,
)
from dnadesign.infer.src.features.selectors import resolve_intermediate_selector
from dnadesign.infer.src.features.sequence_views import load_sequence_view_input_records, resolve_sequence_view_contexts
from dnadesign.usr import Dataset, SequenceViewRecord, ensure_sequence_contract_namespaces, write_sequence_views


def _dataset_with_sequence_view(tmp_path: Path) -> tuple[Path, str]:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "reference_views")
    dataset.init(source="test", notes="sequence-view completion planner test")
    add_result = dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test")
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
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="core60_mean",
                created_at="2026-04-28T00:00:00+00:00",
                created_by="test",
            )
        ],
        conflict_policy="error",
    )
    return usr_root, dataset.name


def _bundle(usr_root: Path, dataset: str) -> PromoterFeatureBundleConfig:
    return PromoterFeatureBundleConfig(
        collect_log_likelihood=False,
        sequence_view_inputs=[
            {
                "dataset": dataset,
                "root": usr_root.as_posix(),
                "view_selector": {"product_kind": "analysis_window"},
                "pooling": {"operation": "core60_mean"},
            }
        ],
    )


def test_sequence_view_completion_planner_reports_missing_vectors(tmp_path: Path) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = _bundle(usr_root, dataset)

    plan = plan_sequence_view_feature_completion(bundle=bundle, model_id="evo2_7b", job_id="reference_views")

    assert plan.required_views == 1
    assert plan.required_vectors == 2
    assert plan.reusable_vectors == 0
    assert plan.missing_vectors == 2
    assert plan.by_product_kind == {"analysis_window": 1}
    assert plan.by_pooling_operation == {"core60_mean": 1}


def test_sequence_view_completion_planner_reports_missing_scalars(tmp_path: Path) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = PromoterFeatureBundleConfig(
        collect_log_likelihood=True,
        collect_output_layer_mean=False,
        collect_intermediate_embedding=False,
        sequence_view_inputs=[
            {
                "dataset": dataset,
                "root": usr_root.as_posix(),
                "view_selector": {"product_kind": "analysis_window"},
                "pooling": {"operation": "core60_mean"},
            }
        ],
    )

    plan = plan_sequence_view_feature_completion(bundle=bundle, model_id="evo2_7b", job_id="reference_views")

    assert plan.required_vectors == 0
    assert plan.required_scalars == 2
    assert plan.reusable_scalars == 0
    assert plan.missing_scalars == 2


def test_sequence_view_completion_planner_reports_missing_products(tmp_path: Path) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = PromoterFeatureBundleConfig(
        collect_log_likelihood=False,
        sequence_view_inputs=[
            {
                "dataset": dataset,
                "root": usr_root.as_posix(),
                "view_selector": {"product_kind": "realized_context", "orientation": "reverse_complement"},
                "pooling": {"operation": "anchor_mean", "bounds_from": "sequence_view"},
            }
        ],
    )

    plan = plan_sequence_view_feature_completion(bundle=bundle, model_id="evo2_7b", job_id="reference_views")

    assert plan.required_views == 0
    assert plan.required_vectors == 0
    assert plan.dataset == dataset
    assert plan.reusable_vectors == 0
    assert plan.missing_vectors == 0
    assert plan.missing_products == 1
    assert plan.missing_product_selectors == [
        {
            "dataset": dataset,
            "root": usr_root.as_posix(),
            "product_kind": "realized_context",
            "view_name": None,
            "alias": None,
            "orientation": "reverse_complement",
            "pooling_operation": "anchor_mean",
        }
    ]
    assert plan.commands.construct_completion == [
        "complete sequence products dataset=reference_views product_kind=realized_context "
        "orientation=reverse_complement pooling=anchor_mean"
    ]


def test_sequence_view_completion_planner_reuses_persisted_feature_vectors(tmp_path: Path) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = _bundle(usr_root, dataset)
    records = load_sequence_view_input_records(bundle=bundle)
    contexts = resolve_sequence_view_contexts(records=records)
    metadata_rows = build_feature_metadata_rows(contexts=contexts, bundle=bundle, model_id="evo2_7b")
    selector = resolve_intermediate_selector(model_id="evo2_7b", intermediate_block=bundle.intermediate_block)
    specs = _sequence_view_feature_vector_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
    )
    persist_feature_vector_rows(
        [
            {
                "_dataset_root": spec["dataset_root"],
                "_dataset_id": spec["dataset_id"],
                "feature_vector_key": spec["feature_vector_key"],
                "value": [1.0, 2.0],
                "created_at": metadata_rows[int(spec["row_index"])]["timestamp"],
            }
            for spec in specs
        ]
    )

    plan = plan_sequence_view_feature_completion(bundle=bundle, model_id="evo2_7b", job_id="reference_views")

    assert plan.required_vectors == 2
    assert plan.reusable_vectors == 2
    assert plan.persisted_vector_reusable == 2
    assert plan.missing_vectors == 0


def test_sequence_view_completion_planner_reuses_persisted_feature_scalars(tmp_path: Path) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = PromoterFeatureBundleConfig(
        collect_log_likelihood=True,
        collect_output_layer_mean=False,
        collect_intermediate_embedding=False,
        sequence_view_inputs=[
            {
                "dataset": dataset,
                "root": usr_root.as_posix(),
                "view_selector": {"product_kind": "analysis_window"},
                "pooling": {"operation": "core60_mean"},
            }
        ],
    )
    records = load_sequence_view_input_records(bundle=bundle)
    contexts = resolve_sequence_view_contexts(records=records)
    metadata_rows = build_feature_metadata_rows(contexts=contexts, bundle=bundle, model_id="evo2_7b")
    specs = _sequence_view_feature_scalar_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
    )
    persist_feature_scalar_rows(
        [
            {
                "_dataset_root": spec["dataset_root"],
                "_dataset_id": spec["dataset_id"],
                "feature_scalar_key": spec["feature_scalar_key"],
                "value": 1.25,
                "created_at": metadata_rows[int(spec["row_index"])]["timestamp"],
            }
            for spec in specs
        ]
    )

    plan = plan_sequence_view_feature_completion(bundle=bundle, model_id="evo2_7b", job_id="reference_views")

    assert plan.required_scalars == 2
    assert plan.reusable_scalars == 2
    assert plan.persisted_scalar_reusable == 2
    assert plan.missing_scalars == 0


def test_sequence_view_completion_planner_uses_key_only_vector_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = _bundle(usr_root, dataset)
    records = load_sequence_view_input_records(bundle=bundle)
    contexts = resolve_sequence_view_contexts(records=records)
    metadata_rows = build_feature_metadata_rows(contexts=contexts, bundle=bundle, model_id="evo2_7b")
    selector = resolve_intermediate_selector(model_id="evo2_7b", intermediate_block=bundle.intermediate_block)
    specs = _sequence_view_feature_vector_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
    )
    persist_feature_vector_rows(
        [
            {
                "_dataset_root": spec["dataset_root"],
                "_dataset_id": spec["dataset_id"],
                "feature_vector_key": spec["feature_vector_key"],
                "value": [1.0, 2.0],
                "created_at": metadata_rows[int(spec["row_index"])]["timestamp"],
            }
            for spec in specs
        ]
    )

    key_only_calls = 0
    real_key_loader = completion_planner.load_feature_vector_keys

    def track_key_only_load(*args: object, **kwargs: object) -> set[str]:
        nonlocal key_only_calls
        key_only_calls += 1
        return real_key_loader(*args, **kwargs)

    monkeypatch.setattr(completion_planner, "load_feature_vector_keys", track_key_only_load)

    plan = plan_sequence_view_feature_completion(bundle=bundle, model_id="evo2_7b", job_id="reference_views")

    assert key_only_calls == 1
    assert plan.persisted_vector_reusable == 2
    assert plan.missing_vectors == 0
