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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.infer.src.features import aliases, completion_planner
from dnadesign.infer.src.features.aliases import (
    FEATURE_ALIAS_INVENTORY_COLUMNS,
    FEATURE_SCALAR_ALIAS_INVENTORY_COLUMNS,
    feature_alias_path,
    load_feature_alias_inventory_rows,
    load_feature_scalar_alias_inventory_rows,
    persist_feature_alias_rows,
    persist_feature_scalar_alias_rows,
    persist_feature_scalar_rows,
    persist_feature_vector_rows,
)
from dnadesign.infer.src.features.completion_planner import (
    plan_sequence_view_feature_completion,
    plan_sequence_view_feature_inventory_completion,
)
from dnadesign.infer.src.features.contracts import SequenceFeatureBundleConfig
from dnadesign.infer.src.features.execution import (
    _sequence_view_feature_alias_rows,
    _sequence_view_feature_scalar_alias_rows,
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


def _bundle(usr_root: Path, dataset: str) -> SequenceFeatureBundleConfig:
    return SequenceFeatureBundleConfig(
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
    assert plan.shard_plan.shard_count == 1
    assert plan.shard_plan.runtime_fingerprint_key
    assert plan.shard_plan.ledger_relative_path == "_derived/infer/checkpoints/reference_views/ledger.json"


def test_sequence_view_completion_planner_reports_missing_scalars(tmp_path: Path) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = SequenceFeatureBundleConfig(
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
    assert plan.shard_plan.pending_scalar_keys == 2


def test_sequence_view_completion_planner_reports_missing_products(tmp_path: Path) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = SequenceFeatureBundleConfig(
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


def test_sequence_view_input_records_preserve_pooling_slots_for_same_view(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "context_views")
    dataset.init(source="test", notes="sequence-view pooling slot fidelity test")
    add_result = dataset.add_sequences(["AAAACCCCGGGGTTTT"], bio_type="dna", alphabet="dna_4", source="test")
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="context_forward",
                product_kind="realized_context",
                context_kind="template_1kb",
                orientation="forward",
                analysis_only=False,
                source_dataset_id=dataset.name,
                anchor_start_0=4,
                anchor_end_0=8,
                recommended_pooling="anchor_mean",
                created_at="2026-04-28T00:00:00+00:00",
                created_by="test",
            )
        ],
        conflict_policy="error",
    )
    bundle = SequenceFeatureBundleConfig(
        collect_log_likelihood=True,
        collect_output_layer_mean=True,
        collect_intermediate_embedding=True,
        sequence_view_inputs=[
            {
                "dataset": dataset.name,
                "root": usr_root.as_posix(),
                "view_selector": {"product_kind": "realized_context", "orientation": "forward"},
                "pooling": {"operation": "seq_mean"},
            },
            {
                "dataset": dataset.name,
                "root": usr_root.as_posix(),
                "view_selector": {"product_kind": "realized_context", "orientation": "forward"},
                "pooling": {"operation": "anchor_mean", "bounds_from": "sequence_view"},
            },
        ],
    )

    records = load_sequence_view_input_records(bundle=bundle)
    contexts = resolve_sequence_view_contexts(records=records)
    metadata_rows = build_feature_metadata_rows(contexts=contexts, bundle=bundle, model_id="evo2_7b")
    planner_metadata_rows = build_feature_metadata_rows(
        contexts=contexts,
        bundle=bundle,
        model_id="evo2_7b",
        include_feature_request_digest=False,
    )
    selector = resolve_intermediate_selector(model_id="evo2_7b", intermediate_block=bundle.intermediate_block)
    vector_specs = _sequence_view_feature_vector_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
    )
    scalar_specs = _sequence_view_feature_scalar_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
    )
    planner_vector_specs = _sequence_view_feature_vector_specs(
        contexts=contexts,
        metadata_rows=planner_metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
    )
    planner_scalar_specs = _sequence_view_feature_scalar_specs(
        contexts=contexts,
        metadata_rows=planner_metadata_rows,
        bundle=bundle,
    )

    assert [record["_infer_pooling_operation"] for record in records] == ["seq_mean", "anchor_mean"]
    assert len({context.view_id for context in contexts}) == 1
    assert len({(context.view_id, context.pooling_operation) for context in contexts}) == 2
    assert [row["feature_request_digest"] for row in planner_metadata_rows] == [None, None]
    assert [row["forward_pass_key"] for row in planner_metadata_rows] == [
        row["forward_pass_key"] for row in metadata_rows
    ]
    assert {spec["feature_vector_key"] for spec in planner_vector_specs} == {
        spec["feature_vector_key"] for spec in vector_specs
    }
    assert {spec["feature_scalar_key"] for spec in planner_scalar_specs} == {
        spec["feature_scalar_key"] for spec in scalar_specs
    }
    assert len(vector_specs) == 4
    assert len({spec["feature_vector_key"] for spec in vector_specs}) == 4
    assert len(scalar_specs) == 4
    assert len({spec["feature_scalar_key"] for spec in scalar_specs}) == 2
    exact_plan = plan_sequence_view_feature_completion(bundle=bundle, model_id="evo2_7b", job_id="context_views")
    inventory_plan = plan_sequence_view_feature_inventory_completion(
        bundle=bundle,
        model_id="evo2_7b",
        job_id="context_views",
    )
    assert exact_plan.required_vectors == 4
    assert inventory_plan.required_vectors == 4
    assert exact_plan.required_scalars == 2
    assert inventory_plan.required_scalars == 2


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
    assert plan.shard_plan.shard_count == 0


def test_sequence_view_completion_planner_reuses_persisted_feature_scalars(tmp_path: Path) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = SequenceFeatureBundleConfig(
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


def test_sequence_view_completion_planner_counts_partial_vector_and_scalar_completion_independently(
    tmp_path: Path,
) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = SequenceFeatureBundleConfig(
        collect_log_likelihood=True,
        collect_output_layer_mean=True,
        collect_intermediate_embedding=True,
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
    selector = resolve_intermediate_selector(model_id="evo2_7b", intermediate_block=bundle.intermediate_block)
    vector_specs = _sequence_view_feature_vector_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
    )
    scalar_specs = _sequence_view_feature_scalar_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
    )
    persist_feature_vector_rows(
        [
            {
                "_dataset_root": vector_specs[0]["dataset_root"],
                "_dataset_id": vector_specs[0]["dataset_id"],
                "feature_vector_key": vector_specs[0]["feature_vector_key"],
                "value": [1.0, 2.0],
                "created_at": metadata_rows[int(vector_specs[0]["row_index"])]["timestamp"],
            }
        ]
    )
    persist_feature_scalar_rows(
        [
            {
                "_dataset_root": scalar_specs[0]["dataset_root"],
                "_dataset_id": scalar_specs[0]["dataset_id"],
                "feature_scalar_key": scalar_specs[0]["feature_scalar_key"],
                "value": 1.25,
                "created_at": metadata_rows[int(scalar_specs[0]["row_index"])]["timestamp"],
            }
        ]
    )

    plan = plan_sequence_view_feature_completion(
        bundle=bundle,
        model_id="evo2_7b",
        job_id="reference_views",
        infer_command="uv run infer run --config config.yaml --job reference_views",
    )

    assert plan.required_vectors == 2
    assert plan.reusable_vectors == 1
    assert plan.missing_vectors == 1
    assert plan.required_scalars == 2
    assert plan.reusable_scalars == 1
    assert plan.missing_scalars == 1
    assert plan.missing_products == 0
    assert plan.commands.infer_backfill == ["uv run infer run --config config.yaml --job reference_views"]


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


def test_feature_alias_inventory_loaders_project_narrow_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = SequenceFeatureBundleConfig(
        collect_log_likelihood=True,
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
    selector = resolve_intermediate_selector(model_id="evo2_7b", intermediate_block=bundle.intermediate_block)
    persist_feature_alias_rows(
        _sequence_view_feature_alias_rows(
            contexts=contexts,
            metadata_rows=metadata_rows,
            bundle=bundle,
            selector=selector.intermediate_selector,
            model_id="evo2_7b",
        )
    )
    persist_feature_scalar_alias_rows(
        _sequence_view_feature_scalar_alias_rows(
            contexts=contexts,
            metadata_rows=metadata_rows,
            bundle=bundle,
            model_id="evo2_7b",
        )
    )

    captured_columns: list[tuple[str, ...]] = []
    real_read_table = aliases.pq.read_table

    def track_read_table(*args: object, **kwargs: object):
        captured_columns.append(tuple(str(column) for column in (kwargs.get("columns") or ())))
        return real_read_table(*args, **kwargs)

    monkeypatch.setattr(aliases.pq, "read_table", track_read_table)

    vector_rows = load_feature_alias_inventory_rows(dataset_root=usr_root, dataset_id=dataset)
    scalar_rows = load_feature_scalar_alias_inventory_rows(dataset_root=usr_root, dataset_id=dataset)

    assert captured_columns == [
        FEATURE_ALIAS_INVENTORY_COLUMNS,
        FEATURE_SCALAR_ALIAS_INVENTORY_COLUMNS,
    ]
    assert set(vector_rows[0]) == set(FEATURE_ALIAS_INVENTORY_COLUMNS)
    assert set(scalar_rows[0]) == set(FEATURE_SCALAR_ALIAS_INVENTORY_COLUMNS)
    assert "sequence_id" not in vector_rows[0]
    assert "sequence_id" not in scalar_rows[0]


def test_sequence_view_inventory_completion_reuses_dataset_inventory_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usr_root, dataset = _dataset_with_sequence_view(tmp_path)
    bundle = SequenceFeatureBundleConfig(
        collect_log_likelihood=True,
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
    selector = resolve_intermediate_selector(model_id="evo2_7b", intermediate_block=bundle.intermediate_block)
    persist_feature_alias_rows(
        _sequence_view_feature_alias_rows(
            contexts=contexts,
            metadata_rows=metadata_rows,
            bundle=bundle,
            selector=selector.intermediate_selector,
            model_id="evo2_7b",
        )
    )
    persist_feature_scalar_alias_rows(
        _sequence_view_feature_scalar_alias_rows(
            contexts=contexts,
            metadata_rows=metadata_rows,
            bundle=bundle,
            model_id="evo2_7b",
        )
    )

    call_counts = {
        "vector_alias": 0,
        "vector_keys": 0,
        "scalar_alias": 0,
        "scalar_keys": 0,
    }
    real_vector_alias_loader = completion_planner.load_feature_alias_inventory_rows
    real_vector_key_loader = completion_planner._load_feature_vector_key_inventory
    real_scalar_alias_loader = completion_planner.load_feature_scalar_alias_inventory_rows
    real_scalar_key_loader = completion_planner._load_feature_scalar_key_inventory

    def track_vector_alias_loader(*args: object, **kwargs: object) -> list[dict[str, object]]:
        call_counts["vector_alias"] += 1
        return real_vector_alias_loader(*args, **kwargs)

    def track_vector_key_loader(*args: object, **kwargs: object) -> set[str]:
        call_counts["vector_keys"] += 1
        return real_vector_key_loader(*args, **kwargs)

    def track_scalar_alias_loader(*args: object, **kwargs: object) -> list[dict[str, object]]:
        call_counts["scalar_alias"] += 1
        return real_scalar_alias_loader(*args, **kwargs)

    def track_scalar_key_loader(*args: object, **kwargs: object) -> set[str]:
        call_counts["scalar_keys"] += 1
        return real_scalar_key_loader(*args, **kwargs)

    monkeypatch.setattr(completion_planner, "load_feature_alias_inventory_rows", track_vector_alias_loader)
    monkeypatch.setattr(completion_planner, "_load_feature_vector_key_inventory", track_vector_key_loader)
    monkeypatch.setattr(completion_planner, "load_feature_scalar_alias_inventory_rows", track_scalar_alias_loader)
    monkeypatch.setattr(completion_planner, "_load_feature_scalar_key_inventory", track_scalar_key_loader)

    inventory_cache: dict[tuple[str, ...], object] = {}
    for job_id in ("reference_views_forward", "reference_views_repeat"):
        plan_sequence_view_feature_inventory_completion(
            bundle=bundle,
            model_id="evo2_7b",
            job_id=job_id,
            inventory_cache=inventory_cache,
        )

    assert call_counts == {
        "vector_alias": 1,
        "vector_keys": 1,
        "scalar_alias": 1,
        "scalar_keys": 1,
    }


def test_sequence_view_inventory_completion_reuses_aliases_and_reports_stale_payloads(tmp_path: Path) -> None:
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
    persist_feature_alias_rows(
        _sequence_view_feature_alias_rows(
            contexts=contexts,
            metadata_rows=metadata_rows,
            bundle=bundle,
            selector=selector.intermediate_selector,
            model_id="evo2_7b",
        )
    )
    persist_feature_vector_rows(
        [
            {
                "_dataset_root": specs[0]["dataset_root"],
                "_dataset_id": specs[0]["dataset_id"],
                "feature_vector_key": specs[0]["feature_vector_key"],
                "value": [1.0, 2.0],
                "created_at": metadata_rows[int(specs[0]["row_index"])]["timestamp"],
            }
        ]
    )

    plan = plan_sequence_view_feature_inventory_completion(
        bundle=bundle,
        model_id="evo2_7b",
        job_id="reference_views",
    )

    assert plan.required_views == 1
    assert plan.required_vectors == 2
    assert plan.reusable_vectors == 1
    assert plan.stale_vectors == 1
    assert plan.missing_vectors == 0
    assert plan.existing_aliases == 2
    assert plan.by_pooling_operation == {"core60_mean": 1}
    assert plan.shard_plan.pending_vector_keys == 1
    assert plan.shard_plan.shard_count == 1
    assert plan.shard_plan.runtime_fingerprint_key


def test_sequence_view_inventory_completion_quarantines_legacy_aliases_without_runtime_fingerprint(
    tmp_path: Path,
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
    legacy_alias_rows = []
    for row in _sequence_view_feature_alias_rows(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
        model_id="evo2_7b",
    ):
        payload = dict(row)
        payload.pop("_dataset_root")
        payload.pop("_dataset_id")
        legacy_alias_rows.append(payload)
    legacy_schema = pa.schema(
        [
            pa.field("alias_id", pa.string()),
            pa.field("view_id", pa.string()),
            pa.field("view_name", pa.string()),
            pa.field("sequence_id", pa.string()),
            pa.field("feature_vector_key", pa.string()),
            pa.field("forward_pass_key", pa.string()),
            pa.field("provider", pa.string()),
            pa.field("model_name", pa.string()),
            pa.field("model_revision", pa.string()),
            pa.field("layer_name", pa.string()),
            pa.field("representation_kind", pa.string()),
            pa.field("pooling_operation", pa.string()),
            pa.field("pooling_start_0", pa.int64()),
            pa.field("pooling_end_0", pa.int64()),
            pa.field("orientation", pa.string()),
            pa.field("source_dataset_id", pa.string()),
            pa.field("feature_request_digest", pa.string()),
            pa.field("created_at", pa.string()),
        ]
    )
    alias_path = feature_alias_path(dataset_root=usr_root, dataset_id=dataset)
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                field.name: pa.array([row.get(field.name) for row in legacy_alias_rows], type=field.type)
                for field in legacy_schema
            },
            schema=legacy_schema,
        ),
        alias_path,
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

    plan = plan_sequence_view_feature_inventory_completion(
        bundle=bundle,
        model_id="evo2_7b",
        job_id="reference_views",
    )

    assert plan.required_vectors == 2
    assert plan.reusable_vectors == 0
    assert plan.stale_vectors == 2
    assert plan.missing_vectors == 0
