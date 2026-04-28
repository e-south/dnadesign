"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_legacy_alias_migration.py

Legacy row-overlay to sequence-view alias migration tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.devtools.testsupport.usr import register_test_namespace
from dnadesign.infer.src.contracts import infer_usr_column_name
from dnadesign.infer.src.features.aliases import FEATURE_ALIAS_RELATIVE_PATH, FEATURE_VECTOR_RELATIVE_PATH
from dnadesign.infer.src.features.contracts import PromoterFeatureBundleConfig
from dnadesign.infer.src.features.legacy_alias_migration import migrate_legacy_overlay_aliases
from dnadesign.infer.src.features.legacy_payload_retirement import (
    prune_stale_infer_overlay_columns,
    retire_legacy_overlay_payloads,
)
from dnadesign.usr import Dataset, SequenceViewRecord, ensure_sequence_contract_namespaces, write_sequence_views


def _dataset_with_forward_construct_insert(tmp_path: Path) -> tuple[Path, Dataset, str]:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "anchor_views")
    dataset.init(source="test", notes="legacy alias migration test")
    sequence = "ACGT" * 15
    add_result = dataset.add_sequences([sequence], bio_type="dna", alphabet="dna_4", source="test")
    sequence_id = add_result.ids[0]
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=sequence_id,
                view_name="construct_insert_forward",
                product_kind="construct_insert",
                context_kind="anchor_only",
                orientation="forward",
                analysis_only=False,
                source_dataset_id=dataset.name,
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="seq_mean",
                created_at="2026-04-28T00:00:00+00:00",
                created_by="test",
            )
        ],
        conflict_policy="error",
    )
    return usr_root, dataset, sequence_id


def _sequence_view_bundle(usr_root: Path, dataset: str, *, orientation: str = "forward") -> PromoterFeatureBundleConfig:
    return PromoterFeatureBundleConfig(
        collect_log_likelihood=False,
        collect_output_layer_mean=False,
        sequence_view_inputs=[
            {
                "dataset": dataset,
                "root": usr_root.as_posix(),
                "view_selector": {"product_kind": "construct_insert", "orientation": orientation},
                "pooling": {"operation": "seq_mean"},
            }
        ],
    )


def _write_legacy_anchor_overlay(dataset: Dataset, sequence_id: str) -> None:
    legacy_job_id = "anchor_only_7b_features"
    feature_column = infer_usr_column_name(
        model_id="evo2_7b",
        job_id=legacy_job_id,
        out_id="intermediate_embedding__block26_mlp_out__seq_mean",
    )
    metadata = {
        out_id: infer_usr_column_name(model_id="evo2_7b", job_id=legacy_job_id, out_id=out_id)
        for out_id in (
            "metadata__sequence_id",
            "metadata__context_kind",
            "metadata__resolved_length",
            "metadata__anchor_start",
            "metadata__anchor_end",
            "metadata__model_name",
            "metadata__intermediate_selector",
            "metadata__timestamp",
        )
    }
    register_test_namespace(
        dataset.root,
        namespace="infer",
        columns_spec=",".join(
            [
                f"{feature_column}:list<float64>",
                f"{metadata['metadata__sequence_id']}:string",
                f"{metadata['metadata__context_kind']}:string",
                f"{metadata['metadata__resolved_length']}:int64",
                f"{metadata['metadata__anchor_start']}:int64",
                f"{metadata['metadata__anchor_end']}:int64",
                f"{metadata['metadata__model_name']}:string",
                f"{metadata['metadata__intermediate_selector']}:string",
                f"{metadata['metadata__timestamp']}:string",
            ]
        ),
        overwrite=True,
    )
    dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": [sequence_id],
                feature_column: [[1.0, 2.0, 3.0]],
                metadata["metadata__sequence_id"]: [sequence_id],
                metadata["metadata__context_kind"]: ["anchor_only"],
                metadata["metadata__resolved_length"]: [60],
                metadata["metadata__anchor_start"]: [0],
                metadata["metadata__anchor_end"]: [60],
                metadata["metadata__model_name"]: ["evo2_7b"],
                metadata["metadata__intermediate_selector"]: ["block26_mlp_out"],
                metadata["metadata__timestamp"]: ["2026-04-01T00:00:00+00:00"],
            }
        ),
        key="id",
        actor={"tool": "test", "run_id": "legacy-overlay-fixture"},
    )


def _write_mixed_stale_infer_overlay(dataset: Dataset, sequence_id: str) -> tuple[str, str, str]:
    stale_payload = infer_usr_column_name(
        model_id="evo2_20b",
        job_id="anchor_only_20b_features",
        out_id="intermediate_embedding__block23_mlp_out__seq_mean",
    )
    stale_metadata = infer_usr_column_name(
        model_id="evo2_20b",
        job_id="anchor_only_20b_features",
        out_id="metadata__feature_request_digest",
    )
    retained_payload = infer_usr_column_name(
        model_id="evo2_7b",
        job_id="anchor_only_7b_features",
        out_id="output_layer_mean__seq_mean",
    )
    register_test_namespace(
        dataset.root,
        namespace="infer",
        columns_spec=",".join(
            [
                f"{stale_payload}:list<float64>",
                f"{stale_metadata}:string",
                f"{retained_payload}:list<float64>",
            ]
        ),
        overwrite=True,
    )
    dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": [sequence_id],
                stale_payload: [[20.0, 21.0]],
                stale_metadata: ["old-20b-digest"],
                retained_payload: [[7.0, 8.0]],
            }
        ),
        key="id",
        actor={"tool": "test", "run_id": "stale-infer-fixture"},
    )
    return stale_payload, stale_metadata, retained_payload


def test_legacy_overlay_alias_migration_copies_verified_forward_vectors(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _dataset_with_forward_construct_insert(tmp_path)
    _write_legacy_anchor_overlay(dataset, sequence_id)
    bundle = _sequence_view_bundle(usr_root, dataset.name)

    dry_run = migrate_legacy_overlay_aliases(
        bundle=bundle,
        model_id="evo2_7b",
        legacy_job_id="anchor_only_7b_features",
        write=False,
    )

    assert dry_run.required_vectors == 1
    assert dry_run.reusable_vectors == 1
    assert dry_run.payload_unverified_vectors == 1
    assert dry_run.vectors_written == 0
    assert not (dataset.dir / FEATURE_VECTOR_RELATIVE_PATH).exists()

    verified_dry_run = migrate_legacy_overlay_aliases(
        bundle=bundle,
        model_id="evo2_7b",
        legacy_job_id="anchor_only_7b_features",
        write=False,
        verify_payloads=True,
    )
    assert verified_dry_run.reusable_vectors == 1
    assert verified_dry_run.payload_unverified_vectors == 0

    written = migrate_legacy_overlay_aliases(
        bundle=bundle,
        model_id="evo2_7b",
        legacy_job_id="anchor_only_7b_features",
        write=True,
    )

    assert written.reusable_vectors == 1
    assert written.vectors_written == 1
    assert written.aliases_written == 1
    vector_rows = pq.read_table(dataset.dir / FEATURE_VECTOR_RELATIVE_PATH).to_pylist()
    assert vector_rows[0]["value"] == [1.0, 2.0, 3.0]
    alias_rows = pq.read_table(dataset.dir / FEATURE_ALIAS_RELATIVE_PATH).to_pylist()
    assert alias_rows[0]["sequence_id"] == sequence_id
    assert alias_rows[0]["orientation"] == "forward"
    events = [json.loads(line) for line in dataset.events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["action"] for event in events[-2:]] == [
        "infer_feature_vectors_write",
        "infer_feature_aliases_write",
    ]

    rerun = migrate_legacy_overlay_aliases(
        bundle=bundle,
        model_id="evo2_7b",
        legacy_job_id="anchor_only_7b_features",
        write=True,
    )
    assert rerun.vectors_written == 0
    assert rerun.aliases_written == 0


def test_legacy_overlay_alias_migration_refuses_reverse_complement_reuse(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _dataset_with_forward_construct_insert(tmp_path)
    _write_legacy_anchor_overlay(dataset, sequence_id)
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=sequence_id,
                view_name="construct_insert_reverse_complement",
                product_kind="construct_insert",
                context_kind="anchor_only",
                orientation="reverse_complement",
                analysis_only=False,
                source_dataset_id=dataset.name,
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="seq_mean",
                created_at="2026-04-28T00:00:00+00:00",
                created_by="test",
            )
        ],
        conflict_policy="error",
    )
    bundle = _sequence_view_bundle(usr_root, dataset.name, orientation="reverse_complement")

    result = migrate_legacy_overlay_aliases(
        bundle=bundle,
        model_id="evo2_7b",
        legacy_job_id="anchor_only_7b_features",
        write=True,
    )

    assert result.required_vectors == 1
    assert result.reusable_vectors == 0
    assert result.orientation_blocked_vectors == 1
    assert result.vectors_written == 0
    assert not (dataset.dir / FEATURE_VECTOR_RELATIVE_PATH).exists()


def test_legacy_payload_retirement_prunes_only_protected_payload_column(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _dataset_with_forward_construct_insert(tmp_path)
    _write_legacy_anchor_overlay(dataset, sequence_id)
    bundle = _sequence_view_bundle(usr_root, dataset.name)
    migrate_legacy_overlay_aliases(
        bundle=bundle,
        model_id="evo2_7b",
        legacy_job_id="anchor_only_7b_features",
        write=True,
    )

    dry_run = retire_legacy_overlay_payloads(
        bundle=bundle,
        model_id="evo2_7b",
        legacy_job_id="anchor_only_7b_features",
        write=False,
    )

    assert dry_run.required_vectors == 1
    assert dry_run.protected_vectors == 1
    assert dry_run.missing_modern_vectors == 0
    assert dry_run.legacy_parts_with_payload == 1
    assert dry_run.files_rewritten == 0

    written = retire_legacy_overlay_payloads(
        bundle=bundle,
        model_id="evo2_7b",
        legacy_job_id="anchor_only_7b_features",
        write=True,
    )

    assert written.files_rewritten == 1
    assert written.bytes_reclaimed > 0
    part_files = sorted((dataset.dir / "_derived" / "infer").glob("part-*.parquet"))
    assert len(part_files) == 1
    schema_names = pq.ParquetFile(part_files[0]).schema_arrow.names
    retired_column = infer_usr_column_name(
        model_id="evo2_7b",
        job_id="anchor_only_7b_features",
        out_id="intermediate_embedding__block26_mlp_out__seq_mean",
    )
    assert retired_column not in schema_names
    assert (
        infer_usr_column_name(
            model_id="evo2_7b",
            job_id="anchor_only_7b_features",
            out_id="metadata__sequence_id",
        )
        in schema_names
    )
    assert (dataset.dir / FEATURE_VECTOR_RELATIVE_PATH).exists()
    events = [json.loads(line) for line in dataset.events_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["action"] == "infer_legacy_payload_retirement"
    assert events[-1]["metrics"]["files_rewritten"] == 1


def test_legacy_payload_retirement_refuses_write_without_modern_vectors(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _dataset_with_forward_construct_insert(tmp_path)
    _write_legacy_anchor_overlay(dataset, sequence_id)
    bundle = _sequence_view_bundle(usr_root, dataset.name)

    dry_run = retire_legacy_overlay_payloads(
        bundle=bundle,
        model_id="evo2_7b",
        legacy_job_id="anchor_only_7b_features",
        write=False,
    )
    assert dry_run.required_vectors == 1
    assert dry_run.protected_vectors == 0
    assert dry_run.legacy_parts_with_payload == 1

    try:
        retire_legacy_overlay_payloads(
            bundle=bundle,
            model_id="evo2_7b",
            legacy_job_id="anchor_only_7b_features",
            write=True,
        )
    except ValueError as error:
        assert "canonical feature-vector protection" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("payload retirement should fail without canonical vectors")


def test_prune_stale_infer_overlay_columns_removes_explicit_prefix_only(tmp_path: Path) -> None:
    _, dataset, sequence_id = _dataset_with_forward_construct_insert(tmp_path)
    stale_payload, stale_metadata, retained_payload = _write_mixed_stale_infer_overlay(dataset, sequence_id)

    dry_run = prune_stale_infer_overlay_columns(
        dataset_root=dataset.root,
        dataset_id=dataset.name,
        column_prefixes=("infer__evo2_20b__",),
        reason="collapsed 20B lane no longer supported for this study",
        write=False,
    )

    assert dry_run.parts_scanned == 1
    assert dry_run.parts_with_columns == 1
    assert stale_payload in dry_run.removed_columns
    assert stale_metadata in dry_run.removed_columns
    assert retained_payload not in dry_run.removed_columns
    before_schema = pq.ParquetFile(sorted((dataset.dir / "_derived" / "infer").glob("part-*.parquet"))[0]).schema_arrow
    assert stale_payload in before_schema.names

    written = prune_stale_infer_overlay_columns(
        dataset_root=dataset.root,
        dataset_id=dataset.name,
        column_prefixes=("infer__evo2_20b__",),
        reason="collapsed 20B lane no longer supported for this study",
        write=True,
    )

    assert written.files_rewritten == 1
    part_path = sorted((dataset.dir / "_derived" / "infer").glob("part-*.parquet"))[0]
    schema_names = pq.ParquetFile(part_path).schema_arrow.names
    assert stale_payload not in schema_names
    assert stale_metadata not in schema_names
    assert retained_payload in schema_names
    events = [json.loads(line) for line in dataset.events_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["action"] == "infer_stale_overlay_column_prune"
    assert events[-1]["metrics"]["columns_removed"] == 2


def test_prune_stale_infer_overlay_columns_refuses_join_column(tmp_path: Path) -> None:
    _, dataset, _ = _dataset_with_forward_construct_insert(tmp_path)

    try:
        prune_stale_infer_overlay_columns(
            dataset_root=dataset.root,
            dataset_id=dataset.name,
            column_names=("id",),
            write=True,
        )
    except ValueError as error:
        assert "join column" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("stale-column pruning should refuse the join column")
