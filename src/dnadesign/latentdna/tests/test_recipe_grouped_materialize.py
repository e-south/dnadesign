"""Recipe-level contracts for grouped Infer sidecar view materialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.infer.src.features.aliases import (
    compute_feature_alias_id,
    persist_feature_alias_rows,
    persist_feature_vector_rows,
)
from dnadesign.infer.src.features.cache_keys import DNA_SEQUENCE_CASE_POLICY
from dnadesign.latentdna.src.io.matrix_io import read_matrix
from dnadesign.latentdna.src.services.recipe_service import run_recipe
from dnadesign.latentdna.src.sources import infer_sidecar_join
from dnadesign.usr import Dataset, SequenceViewRecord, ensure_sequence_contract_namespaces, write_sequence_views


def _write_two_view_sidecar_workspace(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "grouped_sidecar")
    dataset.init(source="test", notes="grouped sidecar recipe test")
    sequence_ids = dataset.add_sequences(["ACGT" * 15, "TGCA" * 15], bio_type="dna", alphabet="dna_4").ids
    created_at = "2026-05-07T00:00:00+00:00"
    sequence_views = [
        SequenceViewRecord(
            sequence_id=sequence_ids[0],
            view_name="seq_mean_view",
            product_kind="analysis_window",
            context_kind="analysis_window",
            orientation="forward",
            analysis_only=True,
            source_dataset_id=dataset.name,
            anchor_start_0=0,
            anchor_end_0=60,
            recommended_pooling="seq_mean",
            created_at=created_at,
            created_by="test",
        ),
        SequenceViewRecord(
            sequence_id=sequence_ids[1],
            view_name="anchor_mean_view",
            product_kind="analysis_window",
            context_kind="analysis_window",
            orientation="forward",
            analysis_only=True,
            source_dataset_id=dataset.name,
            anchor_start_0=0,
            anchor_end_0=60,
            recommended_pooling="anchor_mean",
            created_at=created_at,
            created_by="test",
        ),
    ]
    write_sequence_views(dataset, sequence_views, conflict_policy="error")

    alias_rows = []
    vector_rows = []
    for index, sequence_view in enumerate(sequence_views):
        pooling = str(sequence_view.recommended_pooling)
        feature_vector_key = f"fv_{pooling}"
        alias_rows.append(
            {
                "_dataset_root": usr_root.as_posix(),
                "_dataset_id": dataset.name,
                "alias_id": compute_feature_alias_id(
                    view_id=sequence_view.view_id,
                    sequence_id=sequence_view.sequence_id,
                    feature_vector_key=feature_vector_key,
                    representation_kind="intermediate_embedding",
                ),
                "view_id": sequence_view.view_id,
                "view_name": sequence_view.view_name,
                "sequence_id": sequence_view.sequence_id,
                "feature_vector_key": feature_vector_key,
                "forward_pass_key": f"fp_{pooling}",
                "provider": "evo2",
                "model_name": "evo2_7b",
                "model_revision": None,
                "layer_name": "block26_mlp_out",
                "representation_kind": "intermediate_embedding",
                "pooling_operation": pooling,
                "pooling_start_0": 0,
                "pooling_end_0": 60,
                "orientation": "forward",
                "source_dataset_id": dataset.name,
                "feature_request_digest": f"digest_{pooling}",
                "runtime_fingerprint_key": "runtime_fingerprint_fixture",
                "sequence_case_policy": DNA_SEQUENCE_CASE_POLICY,
                "created_at": created_at,
            }
        )
        vector_rows.append(
            {
                "_dataset_root": usr_root.as_posix(),
                "_dataset_id": dataset.name,
                "feature_vector_key": feature_vector_key,
                "value": [float(index), float(index + 1), float(index + 2)],
                "created_at": created_at,
            }
        )
    persist_feature_alias_rows(alias_rows)
    persist_feature_vector_rows(vector_rows)

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    workspace_config = {
        "schema_version": "latentdna.workspace.v1",
        "workspace": {"id": "grouped_sidecar_recipe", "output_root": "./outputs"},
        "defaults": {
            "analysis_dtype": "float32",
            "metric": "cosine",
            "random_seed": 17,
            "plot_formats": ["svg"],
            "neighbor_backend": "auto",
        },
        "sources": {
            "seq_mean_features": {
                "kind": "infer_feature_sidecar",
                "root": usr_root.as_posix(),
                "dataset": dataset.name,
                "record_key": "alias_id",
                "subject_key": "id",
                "where": {
                    "model_name": "evo2_7b",
                    "representation_kind": "intermediate_embedding",
                    "pooling_operation": "seq_mean",
                    "orientation": "forward",
                },
            },
            "anchor_mean_features": {
                "kind": "infer_feature_sidecar",
                "root": usr_root.as_posix(),
                "dataset": dataset.name,
                "record_key": "alias_id",
                "subject_key": "id",
                "where": {
                    "model_name": "evo2_7b",
                    "representation_kind": "intermediate_embedding",
                    "pooling_operation": "anchor_mean",
                    "orientation": "forward",
                },
            },
        },
        "metadata": {"include": ["view_name", "product_kind"]},
        "views": {
            "seq_mean_view": {
                "source": "seq_mean_features",
                "vector": {"kind": "column", "name": "value"},
                "coordinate_space_id": "evo2_7b_fixture",
            },
            "anchor_mean_view": {
                "source": "anchor_mean_features",
                "vector": {"kind": "column", "name": "value"},
                "coordinate_space_id": "evo2_7b_fixture",
            },
        },
        "recipes": {
            "two_view_recipe": {
                "steps": [
                    {"id": "materialize_seq_mean", "op": "view.materialize", "params": {"view": "seq_mean_view"}},
                    {
                        "id": "materialize_anchor_mean",
                        "op": "view.materialize",
                        "params": {"view": "anchor_mean_view"},
                    },
                ]
            }
        },
    }
    (workspace_dir / "config.yaml").write_text(yaml.safe_dump(workspace_config, sort_keys=False), encoding="utf-8")
    return workspace_dir


def test_recipe_groups_compatible_infer_sidecar_materializations_after_one_shared_dim_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_dir = _write_two_view_sidecar_workspace(tmp_path)
    real_parquet_file = infer_sidecar_join.pq.ParquetFile
    payload_scans = 0

    class CountingParquetFile:
        def __init__(self, path, *args, **kwargs):
            self.path = Path(path)
            self.inner = real_parquet_file(path, *args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

        def iter_batches(self, *args, **kwargs):
            nonlocal payload_scans
            columns = list(kwargs.get("columns") or [])
            if self.path.name == "feature_vectors.parquet" and "value" in columns:
                payload_scans += 1
            return self.inner.iter_batches(*args, **kwargs)

    def counting_parquet_file(path, *args, **kwargs):
        return CountingParquetFile(path, *args, **kwargs)

    monkeypatch.setattr(infer_sidecar_join.pq, "ParquetFile", counting_parquet_file)

    result = run_recipe(workspace_dir, "two_view_recipe", refresh_catalog=False)

    assert result.status == "ok"
    assert payload_scans == 2
    assert read_matrix(workspace_dir / "outputs" / "views" / "seq_mean_view" / "matrix.npy").tolist() == [
        [0.0, 1.0, 2.0]
    ]
    assert read_matrix(workspace_dir / "outputs" / "views" / "anchor_mean_view" / "matrix.npy").tolist() == [
        [1.0, 2.0, 3.0]
    ]
