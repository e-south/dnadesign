"""Contracts for LatentDNA Infer feature sidecar sources."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.latentdna.src.contracts.errors import SourceResolutionError
from dnadesign.latentdna.src.sources import infer_feature_scalar_sidecar_source, infer_feature_sidecar_source
from dnadesign.latentdna.src.sources.infer_feature_sidecar_source import _stable_batch_schema
from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces


def _planned_dataset(tmp_path: Path) -> tuple[Path, Dataset, str]:
    usr_root = tmp_path / "usr"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "planned_sidecar")
    dataset.init(source="test", notes="planned infer feature sidecar test")
    sequence_id = dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test").ids[0]
    return usr_root, dataset, sequence_id


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
