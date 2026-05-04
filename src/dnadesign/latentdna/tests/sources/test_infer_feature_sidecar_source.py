"""Contracts for LatentDNA Infer feature sidecar sources."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.devtools.tests.support.usr import register_test_namespace
from dnadesign.infer.src.features.aliases import (
    compute_feature_alias_id,
    compute_feature_scalar_alias_id,
    compute_feature_scalar_key,
    persist_feature_alias_rows,
    persist_feature_scalar_alias_rows,
    persist_feature_scalar_rows,
    persist_feature_vector_rows,
)
from dnadesign.latentdna.src.contracts.errors import SourceResolutionError
from dnadesign.latentdna.src.io.parquet_io import read_table
from dnadesign.latentdna.src.sources import infer_feature_scalar_sidecar_source, infer_feature_sidecar_source
from dnadesign.latentdna.src.sources.infer_feature_sidecar_source import _stable_batch_schema
from dnadesign.latentdna.src.views.materialize import materialize_view_artifact
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config
from dnadesign.usr import (
    Dataset,
    SequenceViewRecord,
    ViewSemanticsRecord,
    ensure_sequence_contract_namespaces,
    write_sequence_views,
    write_view_semantics,
)


def _planned_dataset(tmp_path: Path) -> tuple[Path, Dataset, str]:
    usr_root = tmp_path / "usr"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "planned_sidecar")
    dataset.init(source="test", notes="planned infer feature sidecar test")
    sequence_id = dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test").ids[0]
    return usr_root, dataset, sequence_id


def _standard_metadata_sidecar_dataset(tmp_path: Path) -> tuple[Path, Dataset]:
    usr_root = tmp_path / "usr_standard"
    ensure_sequence_contract_namespaces(usr_root)
    register_test_namespace(
        usr_root,
        namespace="promoter_standard",
        columns_spec=(
            "promoter_standard__collection_id:string,"
            "promoter_standard__promoter_id:string,"
            "promoter_standard__display_name:string,"
            "promoter_standard__strength_metric:string,"
            "promoter_standard__strength_value:string,"
            "promoter_standard__strength_value_numeric:float64,"
            "promoter_standard__strength_reference:string"
        ),
    )
    dataset = Dataset(usr_root, "reference_core60")
    dataset.init(source="test", notes="reference-standard sidecar metadata test")
    sequence_id = dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test").ids[0]
    created_at = "2026-04-28T00:00:00+00:00"
    dataset.write_overlay_part(
        "promoter_standard",
        pa.table(
            {
                "id": pa.array([sequence_id], type=pa.string()),
                "promoter_standard__collection_id": pa.array(["anderson_igem"], type=pa.string()),
                "promoter_standard__promoter_id": pa.array(["BBa_J23105"], type=pa.string()),
                "promoter_standard__display_name": pa.array(["J23105"], type=pa.string()),
                "promoter_standard__strength_metric": pa.array(
                    ["relative_fluorescence_to_BBa_J23100"],
                    type=pa.string(),
                ),
                "promoter_standard__strength_value": pa.array(["0.24"], type=pa.string()),
                "promoter_standard__strength_value_numeric": pa.array([0.24], type=pa.float64()),
                "promoter_standard__strength_reference": pa.array(["BBa_J23100=1.0"], type=pa.string()),
            }
        ),
        key="id",
    )
    view = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="J23105_core60",
        product_kind="analysis_window",
        context_kind="analysis_window",
        orientation="forward",
        analysis_only=True,
        source_dataset_id=dataset.name,
        source_label="J23105",
        anchor_start_0=0,
        anchor_end_0=60,
        recommended_pooling="core60_mean",
        created_at=created_at,
        created_by="test",
    )
    write_sequence_views(dataset, [view], conflict_policy="error")
    write_view_semantics(
        dataset,
        [
            ViewSemanticsRecord(
                view_id=str(view.view_id),
                sequence_id=sequence_id,
                source_family="synthetic_reference_standard",
                selection_basis="reference_core60",
                view_collections=["reference", "core60"],
                role_tags=["reference_standard", "annotate_core60"],
                study_id="fixture_study",
                created_at=created_at,
                created_by="test",
            )
        ],
        conflict_policy="error",
    )

    feature_vector_key = "fv_j23105_core60_7b"
    forward_pass_key = "fp_j23105_core60_7b"
    alias_id = compute_feature_alias_id(
        view_id=view.view_id,
        sequence_id=sequence_id,
        feature_vector_key=feature_vector_key,
        representation_kind="intermediate_embedding",
    )
    persist_feature_alias_rows(
        [
            {
                "_dataset_root": usr_root.as_posix(),
                "_dataset_id": dataset.name,
                "alias_id": alias_id,
                "view_id": view.view_id,
                "view_name": view.view_name,
                "sequence_id": sequence_id,
                "feature_vector_key": feature_vector_key,
                "forward_pass_key": forward_pass_key,
                "provider": "evo2",
                "model_name": "evo2_7b",
                "model_revision": None,
                "layer_name": "block26_mlp_out",
                "representation_kind": "intermediate_embedding",
                "pooling_operation": "core60_mean",
                "pooling_start_0": 0,
                "pooling_end_0": 60,
                "orientation": "forward",
                "source_dataset_id": dataset.name,
                "feature_request_digest": "digest_j23105_core60",
                "created_at": created_at,
            }
        ]
    )
    scalar_key = compute_feature_scalar_key(
        forward_pass_key=forward_pass_key,
        scalar_kind="log_likelihood__mean_per_token",
    )
    scalar_alias_id = compute_feature_scalar_alias_id(
        view_id=view.view_id,
        sequence_id=sequence_id,
        feature_scalar_key=scalar_key,
        scalar_kind="log_likelihood__mean_per_token",
    )
    persist_feature_scalar_alias_rows(
        [
            {
                "_dataset_root": usr_root.as_posix(),
                "_dataset_id": dataset.name,
                "alias_id": scalar_alias_id,
                "view_id": view.view_id,
                "view_name": view.view_name,
                "sequence_id": sequence_id,
                "feature_scalar_key": scalar_key,
                "forward_pass_key": forward_pass_key,
                "provider": "evo2",
                "model_name": "evo2_7b",
                "model_revision": None,
                "scalar_kind": "log_likelihood__mean_per_token",
                "orientation": "forward",
                "source_dataset_id": dataset.name,
                "feature_request_digest": "digest_j23105_core60",
                "created_at": created_at,
            }
        ]
    )
    persist_feature_vector_rows(
        [
            {
                "_dataset_root": usr_root.as_posix(),
                "_dataset_id": dataset.name,
                "feature_vector_key": feature_vector_key,
                "value": [0.1, 0.2, 0.3],
                "created_at": created_at,
            }
        ]
    )
    persist_feature_scalar_rows(
        [
            {
                "_dataset_root": usr_root.as_posix(),
                "_dataset_id": dataset.name,
                "feature_scalar_key": scalar_key,
                "value": -2.75,
                "created_at": created_at,
            }
        ]
    )
    return usr_root, dataset


def test_stable_batch_schema_uses_later_non_null_metadata_values() -> None:
    schema = _stable_batch_schema(
        ["alias_id", "usr_label__primary", "value"],
        {
            "fv_a": {"alias_id": "alias_a", "usr_label__primary": None},
            "fv_b": {"alias_id": "alias_b", "usr_label__primary": "spyP"},
        },
    )

    assert schema.field("usr_label__primary").type == pa.string()
    assert schema.field("value").type == pa.list_(pa.float64())


def test_missing_feature_sidecar_files_expose_empty_planned_source(tmp_path: Path) -> None:
    usr_root, dataset, _sequence_id = _planned_dataset(tmp_path)

    schema = infer_feature_sidecar_source.inspect_schema(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where={
            "model_name": "evo2_7b",
            "representation_kind": "intermediate_embedding",
            "pooling_operation": "seq_mean",
            "orientation": "forward",
        },
    )
    table = infer_feature_sidecar_source.read_table(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where=None,
        columns=["alias_id", "id", "value"],
    )

    assert schema["row_count"] == 0
    assert schema["vector_columns"] == ["value"]
    assert {"alias_id", "id", "value"}.issubset(set(schema["columns"]))
    assert table.num_rows == 0
    assert table.schema.field("value").type == pa.list_(pa.float64())


def test_alias_rows_without_feature_vectors_fail_fast(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _planned_dataset(tmp_path)
    derived_dir = dataset.dir / "_derived" / "infer"
    derived_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "alias_id": "alias_missing",
                    "view_id": "view_missing",
                    "view_name": "fixture",
                    "sequence_id": sequence_id,
                    "feature_vector_key": "fv_missing",
                    "forward_pass_key": "fp_missing",
                    "provider": "evo2",
                    "model_name": "evo2_7b",
                    "model_revision": None,
                    "layer_name": "block26_mlp_out",
                    "representation_kind": "intermediate_embedding",
                    "pooling_operation": "seq_mean",
                    "pooling_start_0": None,
                    "pooling_end_0": None,
                    "orientation": "forward",
                    "source_dataset_id": dataset.name,
                    "feature_request_digest": "digest",
                    "created_at": "2026-04-28T00:00:00+00:00",
                }
            ]
        ),
        derived_dir / "feature_aliases.parquet",
    )

    with pytest.raises(SourceResolutionError, match="missing feature vectors"):
        infer_feature_sidecar_source.inspect_schema(
            usr_root.as_posix(),
            dataset.name,
            workspace_dir=tmp_path,
            where=None,
        )


def test_feature_sidecar_source_preserves_alias_rows_that_share_vector_keys(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _planned_dataset(tmp_path)
    created_at = "2026-04-28T00:00:00+00:00"
    vector_key = "fv_shared"
    derived_dir = dataset.dir / "_derived" / "infer"
    derived_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "alias_id": "alias_a",
                    "view_id": "view_a",
                    "view_name": "fixture_a",
                    "sequence_id": sequence_id,
                    "feature_vector_key": vector_key,
                    "forward_pass_key": "fp_shared",
                    "provider": "evo2",
                    "model_name": "evo2_7b",
                    "model_revision": None,
                    "layer_name": "block26_mlp_out",
                    "representation_kind": "intermediate_embedding",
                    "pooling_operation": "core60_mean",
                    "pooling_start_0": 0,
                    "pooling_end_0": 60,
                    "orientation": "forward",
                    "source_dataset_id": dataset.name,
                    "feature_request_digest": "digest_a",
                    "created_at": created_at,
                },
                {
                    "alias_id": "alias_b",
                    "view_id": "view_b",
                    "view_name": "fixture_b",
                    "sequence_id": sequence_id,
                    "feature_vector_key": vector_key,
                    "forward_pass_key": "fp_shared",
                    "provider": "evo2",
                    "model_name": "evo2_7b",
                    "model_revision": None,
                    "layer_name": "block26_mlp_out",
                    "representation_kind": "intermediate_embedding",
                    "pooling_operation": "core60_mean",
                    "pooling_start_0": 0,
                    "pooling_end_0": 60,
                    "orientation": "forward",
                    "source_dataset_id": dataset.name,
                    "feature_request_digest": "digest_b",
                    "created_at": created_at,
                },
            ],
            schema=infer_feature_sidecar_source._ALIAS_SCHEMA,
        ),
        derived_dir / "feature_aliases.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"feature_vector_key": vector_key, "value": [0.1, 0.2], "created_at": created_at}],
            schema=infer_feature_sidecar_source._VECTOR_SCHEMA,
        ),
        derived_dir / "feature_vectors.parquet",
    )

    schema = infer_feature_sidecar_source.inspect_schema(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where={"pooling_operation": "core60_mean"},
    )
    table = infer_feature_sidecar_source.read_table(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where={"pooling_operation": "core60_mean"},
        columns=["feature_vector_key", "id", "value"],
    )

    assert schema["row_count"] == 2
    assert table.num_rows == 2
    assert table["feature_vector_key"].to_pylist() == [vector_key, vector_key]


def test_feature_sidecar_inspect_schema_reuses_alias_table_scan(tmp_path: Path, monkeypatch) -> None:
    usr_root, dataset = _standard_metadata_sidecar_dataset(tmp_path)
    calls: list[str] = []
    real_read_alias_table = infer_feature_sidecar_source._read_alias_table

    def counted_read_alias_table(root, dataset_id, *, workspace_dir, where):
        calls.append(str(dataset_id))
        return real_read_alias_table(root, dataset_id, workspace_dir=workspace_dir, where=where)

    monkeypatch.setattr(infer_feature_sidecar_source, "_read_alias_table", counted_read_alias_table)

    schema = infer_feature_sidecar_source.inspect_schema(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where={"pooling_operation": "core60_mean"},
    )

    assert schema["row_count"] == 1
    assert calls == [dataset.name]


def test_feature_sidecar_alias_rows_without_vector_keys_fail_fast(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _planned_dataset(tmp_path)
    derived_dir = dataset.dir / "_derived" / "infer"
    derived_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "alias_id": "alias_null_key",
                    "view_id": "view_null_key",
                    "view_name": "fixture",
                    "sequence_id": sequence_id,
                    "feature_vector_key": None,
                    "forward_pass_key": "fp_null_key",
                    "provider": "evo2",
                    "model_name": "evo2_7b",
                    "model_revision": None,
                    "layer_name": "block26_mlp_out",
                    "representation_kind": "intermediate_embedding",
                    "pooling_operation": "seq_mean",
                    "pooling_start_0": None,
                    "pooling_end_0": None,
                    "orientation": "forward",
                    "source_dataset_id": dataset.name,
                    "feature_request_digest": "digest",
                    "created_at": "2026-04-28T00:00:00+00:00",
                }
            ],
            schema=infer_feature_sidecar_source._ALIAS_SCHEMA,
        ),
        derived_dir / "feature_aliases.parquet",
    )

    with pytest.raises(SourceResolutionError, match="has no feature_vector_key"):
        infer_feature_sidecar_source.inspect_schema(
            usr_root.as_posix(),
            dataset.name,
            workspace_dir=tmp_path,
            where=None,
        )


def test_feature_sidecar_source_carries_reference_strength_metadata_into_materialized_rows(
    tmp_path: Path,
) -> None:
    usr_root, dataset = _standard_metadata_sidecar_dataset(tmp_path)
    workspace_dir = tmp_path / "latentdna_workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "reference_strength_metadata", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "reference_core60_features": {
                        "kind": "infer_feature_sidecar",
                        "root": usr_root.as_posix(),
                        "dataset": dataset.name,
                        "record_key": "alias_id",
                        "subject_key": "id",
                        "where": {
                            "model_name": "evo2_7b",
                            "representation_kind": "intermediate_embedding",
                            "pooling_operation": "core60_mean",
                            "orientation": "forward",
                        },
                    }
                },
                "metadata": {
                    "include": [
                        "source_family",
                        "product_kind",
                        "view_name",
                        "role_tags",
                        "promoter_standard__collection_id",
                        "promoter_standard__display_name",
                        "promoter_standard__strength_value_numeric",
                    ]
                },
                "views": {
                    "z_reference_core60": {
                        "source": "reference_core60_features",
                        "vector": {"kind": "column", "name": "value"},
                        "coordinate_space_id": "evo2_7b_core60",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "reference_core60"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    source_rows = infer_feature_sidecar_source.read_table(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=workspace_dir,
        where={"pooling_operation": "core60_mean"},
        columns=[
            "alias_id",
            "id",
            "value",
            "source_family",
            "product_kind",
            "role_tags",
            "promoter_standard__collection_id",
            "promoter_standard__strength_value_numeric",
        ],
    ).to_pylist()
    context = load_workspace_config(workspace_dir)
    artifact_dir, row_count, dims, *_ = materialize_view_artifact(context, view_id="z_reference_core60")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert row_count == 1
    assert dims == 3
    assert len(source_rows) == 1
    assert len(rows) == 1
    assert source_rows[0]["promoter_standard__collection_id"] == "anderson_igem"
    assert source_rows[0]["promoter_standard__strength_value_numeric"] == 0.24
    assert source_rows[0]["product_kind"] == "analysis_window"
    assert source_rows[0]["role_tags"] == ["reference_standard", "annotate_core60"]
    assert rows == [
        {
            "alias_id": source_rows[0]["alias_id"],
            "id": source_rows[0]["id"],
            "source_family": "synthetic_reference_standard",
            "product_kind": "analysis_window",
            "view_name": "J23105_core60",
            "role_tags": ["reference_standard", "annotate_core60"],
            "promoter_standard__collection_id": "anderson_igem",
            "promoter_standard__display_name": "J23105",
            "promoter_standard__strength_value_numeric": 0.24,
        }
    ]


def test_missing_feature_scalar_sidecar_files_expose_empty_planned_source(tmp_path: Path) -> None:
    usr_root, dataset, _sequence_id = _planned_dataset(tmp_path)

    schema = infer_feature_scalar_sidecar_source.inspect_schema(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where={
            "model_name": "evo2_7b",
            "scalar_kind": "log_likelihood_mean",
            "orientation": "forward",
        },
    )
    table = infer_feature_scalar_sidecar_source.read_table(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where=None,
        columns=["alias_id", "id", "value"],
    )

    assert schema["row_count"] == 0
    assert schema["vector_columns"] == []
    assert {"alias_id", "id", "value"}.issubset(set(schema["columns"]))
    assert table.num_rows == 0
    assert table.schema.field("value").type == pa.float64()


def test_invalid_feature_scalar_where_fails_fast_for_planned_source(tmp_path: Path) -> None:
    usr_root, dataset, _sequence_id = _planned_dataset(tmp_path)

    with pytest.raises(SourceResolutionError, match="where column is missing: absent"):
        infer_feature_scalar_sidecar_source.inspect_schema(
            usr_root.as_posix(),
            dataset.name,
            workspace_dir=tmp_path,
            where={"absent": "value"},
        )


def test_alias_rows_without_feature_scalars_fail_fast(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _planned_dataset(tmp_path)
    derived_dir = dataset.dir / "_derived" / "infer"
    derived_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "alias_id": "alias_missing",
                    "view_id": "view_missing",
                    "view_name": "fixture",
                    "sequence_id": sequence_id,
                    "feature_scalar_key": "fs_missing",
                    "forward_pass_key": "fp_missing",
                    "provider": "evo2",
                    "model_name": "evo2_7b",
                    "model_revision": None,
                    "scalar_kind": "log_likelihood_mean",
                    "orientation": "forward",
                    "source_dataset_id": dataset.name,
                    "feature_request_digest": "digest",
                    "created_at": "2026-04-28T00:00:00+00:00",
                }
            ],
            schema=infer_feature_scalar_sidecar_source._ALIAS_SCHEMA,
        ),
        derived_dir / "feature_scalar_aliases.parquet",
    )

    with pytest.raises(SourceResolutionError, match="missing feature scalars"):
        infer_feature_scalar_sidecar_source.inspect_schema(
            usr_root.as_posix(),
            dataset.name,
            workspace_dir=tmp_path,
            where=None,
        )


def test_feature_scalar_sidecar_source_preserves_alias_rows_that_share_scalar_keys(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _planned_dataset(tmp_path)
    created_at = "2026-04-28T00:00:00+00:00"
    scalar_key = "fs_shared"
    derived_dir = dataset.dir / "_derived" / "infer"
    derived_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "alias_id": "scalar_alias_a",
                    "view_id": "view_a",
                    "view_name": "fixture_a",
                    "sequence_id": sequence_id,
                    "feature_scalar_key": scalar_key,
                    "forward_pass_key": "fp_shared",
                    "provider": "evo2",
                    "model_name": "evo2_7b",
                    "model_revision": None,
                    "scalar_kind": "log_likelihood__mean_per_token",
                    "orientation": "forward",
                    "source_dataset_id": dataset.name,
                    "feature_request_digest": "digest_a",
                    "created_at": created_at,
                },
                {
                    "alias_id": "scalar_alias_b",
                    "view_id": "view_b",
                    "view_name": "fixture_b",
                    "sequence_id": sequence_id,
                    "feature_scalar_key": scalar_key,
                    "forward_pass_key": "fp_shared",
                    "provider": "evo2",
                    "model_name": "evo2_7b",
                    "model_revision": None,
                    "scalar_kind": "log_likelihood__mean_per_token",
                    "orientation": "forward",
                    "source_dataset_id": dataset.name,
                    "feature_request_digest": "digest_b",
                    "created_at": created_at,
                },
            ],
            schema=infer_feature_scalar_sidecar_source._ALIAS_SCHEMA,
        ),
        derived_dir / "feature_scalar_aliases.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"feature_scalar_key": scalar_key, "value": -2.5, "created_at": created_at}],
            schema=infer_feature_scalar_sidecar_source._SCALAR_SCHEMA,
        ),
        derived_dir / "feature_scalars.parquet",
    )

    schema = infer_feature_scalar_sidecar_source.inspect_schema(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where={"scalar_kind": "log_likelihood__mean_per_token"},
    )
    table = infer_feature_scalar_sidecar_source.read_table(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where={"scalar_kind": "log_likelihood__mean_per_token"},
        columns=["feature_scalar_key", "id", "value"],
    )

    assert schema["row_count"] == 2
    assert table.num_rows == 2
    assert table["feature_scalar_key"].to_pylist() == [scalar_key, scalar_key]


def test_feature_scalar_sidecar_inspect_schema_reuses_alias_table_scan(tmp_path: Path, monkeypatch) -> None:
    usr_root, dataset = _standard_metadata_sidecar_dataset(tmp_path)
    calls: list[str] = []
    real_read_alias_table = infer_feature_scalar_sidecar_source._read_alias_table

    def counted_read_alias_table(root, dataset_id, *, workspace_dir, where):
        calls.append(str(dataset_id))
        return real_read_alias_table(root, dataset_id, workspace_dir=workspace_dir, where=where)

    monkeypatch.setattr(infer_feature_scalar_sidecar_source, "_read_alias_table", counted_read_alias_table)

    schema = infer_feature_scalar_sidecar_source.inspect_schema(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where={"scalar_kind": "log_likelihood__mean_per_token"},
    )

    assert schema["row_count"] == 1
    assert calls == [dataset.name]


def test_feature_scalar_sidecar_alias_rows_without_scalar_keys_fail_fast(tmp_path: Path) -> None:
    usr_root, dataset, sequence_id = _planned_dataset(tmp_path)
    derived_dir = dataset.dir / "_derived" / "infer"
    derived_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "alias_id": "scalar_alias_null_key",
                    "view_id": "view_null_key",
                    "view_name": "fixture",
                    "sequence_id": sequence_id,
                    "feature_scalar_key": None,
                    "forward_pass_key": "fp_null_key",
                    "provider": "evo2",
                    "model_name": "evo2_7b",
                    "model_revision": None,
                    "scalar_kind": "log_likelihood__mean_per_token",
                    "orientation": "forward",
                    "source_dataset_id": dataset.name,
                    "feature_request_digest": "digest",
                    "created_at": "2026-04-28T00:00:00+00:00",
                }
            ],
            schema=infer_feature_scalar_sidecar_source._ALIAS_SCHEMA,
        ),
        derived_dir / "feature_scalar_aliases.parquet",
    )

    with pytest.raises(SourceResolutionError, match="has no feature_scalar_key"):
        infer_feature_scalar_sidecar_source.inspect_schema(
            usr_root.as_posix(),
            dataset.name,
            workspace_dir=tmp_path,
            where=None,
        )


def test_feature_scalar_sidecar_source_carries_reference_strength_metadata(tmp_path: Path) -> None:
    usr_root, dataset = _standard_metadata_sidecar_dataset(tmp_path)

    rows = infer_feature_scalar_sidecar_source.read_table(
        usr_root.as_posix(),
        dataset.name,
        workspace_dir=tmp_path,
        where={
            "model_name": "evo2_7b",
            "scalar_kind": "log_likelihood__mean_per_token",
            "orientation": "forward",
        },
        columns=[
            "alias_id",
            "id",
            "value",
            "source_family",
            "product_kind",
            "promoter_standard__collection_id",
            "promoter_standard__strength_value_numeric",
        ],
    ).to_pylist()

    assert len(rows) == 1
    assert rows == [
        {
            "alias_id": rows[0]["alias_id"],
            "id": rows[0]["id"],
            "value": -2.75,
            "source_family": "synthetic_reference_standard",
            "product_kind": "analysis_window",
            "promoter_standard__collection_id": "anderson_igem",
            "promoter_standard__strength_value_numeric": 0.24,
        }
    ]
