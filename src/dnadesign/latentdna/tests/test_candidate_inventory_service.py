"""Candidate X inventory contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.latentdna.src.services import view_shape_cache
from dnadesign.latentdna.src.services.candidate_inventory_service import build_candidate_inventory
from dnadesign.latentdna.src.services.catalog_service import workspace_catalog
from dnadesign.latentdna.src.services.notebook_controls_service import build_workspace_notebook_controls_payload
from dnadesign.latentdna.src.services.view_service import materialize_view
from dnadesign.latentdna.src.services.workspace_snapshot_service import workspace_snapshot
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _write_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["row_01", "row_02"],
                "subject_id": ["subj_01", "subj_02"],
                "embedding": pa.array([[1.0, 0.0], [0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "features.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "candidate_ledger_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "materialized_features": {
                        "kind": "parquet",
                        "path": "inputs/features.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "where": {
                            "model_name": "evo2_7b",
                            "representation_kind": "intermediate_embedding",
                            "pooling_operation": "seq_mean",
                            "orientation": "forward",
                        },
                    },
                    "planned_output_features": {
                        "kind": "parquet",
                        "path": "inputs/not_yet_materialized.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "role": "planned",
                        "where": {
                            "model_name": "evo2_7b",
                            "representation_kind": "output_layer_mean",
                            "pooling_operation": "seq_mean",
                            "orientation": "forward",
                        },
                    },
                },
                "metadata": {"include": []},
                "views": {
                    "embedding_anchor": {
                        "source": "materialized_features",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "evo2_7b_anchor_seq_mean",
                        "tags": {
                            "model": "7b",
                            "family": "intermediate_embedding",
                            "scope": "anchor_60bp",
                            "pooling": "seq_mean",
                            "orientation": "forward",
                        },
                        "role": "primary",
                    },
                    "output_anchor": {
                        "source": "planned_output_features",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "evo2_7b_output_layer",
                        "tags": {
                            "model": "7b",
                            "family": "output_layer_mean",
                            "scope": "anchor_60bp",
                            "pooling": "seq_mean",
                            "orientation": "forward",
                        },
                        "role": "planned",
                    },
                },
                "candidate_sets": {
                    "all_x": {
                        "label": "All candidate X",
                        "views": ["embedding_anchor", "output_anchor"],
                    }
                },
                "notebooks": {
                    "latent_geometry_browser": {
                        "kind": "workspace",
                        "title": "Browser",
                        "default_deliverable": "browser",
                        "candidate_sets": ["all_x"],
                        "default_candidate_set": "all_x",
                    }
                },
                "deliverables": {
                    "browser": {
                        "title": "Browser",
                        "section": "Browser",
                        "question": "Can candidate X be inspected?",
                        "summary": "Candidate inventory fixture.",
                        "recipe": "noop",
                        "requires": {"views": ["embedding_anchor"]},
                        "outputs": {"views": ["embedding_anchor"]},
                        "docs_refs": [],
                        "acceptance_checks": [],
                    }
                },
                "recipes": {
                    "noop": {
                        "steps": [
                            {
                                "id": "materialize_embedding",
                                "op": "view.materialize",
                                "params": {"view": "embedding_anchor"},
                            }
                        ]
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workspace_dir


def test_candidate_inventory_exposes_machine_readable_x_ledger(tmp_path) -> None:
    workspace_dir = _write_workspace(tmp_path)
    materialize_view(workspace_dir, "embedding_anchor")
    context = load_workspace_config(workspace_dir)

    rows = build_candidate_inventory(context)
    by_view = {row["view_id"]: row for row in rows}

    materialized = by_view["embedding_anchor"]
    assert materialized["study_id"] == "candidate_ledger_demo"
    assert materialized["candidate_set_ids"] == ["all_x"]
    assert materialized["source_id"] == "materialized_features"
    assert materialized["dataset"] == "inputs/features.parquet"
    assert materialized["row_basis"] == "subject_id"
    assert materialized["model_name"] == "evo2_7b"
    assert materialized["feature_family"] == "intermediate_embedding"
    assert materialized["modality"] == "vector"
    assert materialized["sequence_scope"] == "anchor_60bp"
    assert materialized["pooling_operation"] == "seq_mean"
    assert materialized["orientation"] == "forward"
    assert materialized["coordinate_space_id"] == "evo2_7b_anchor_seq_mean"
    assert materialized["n_rows"] == 2
    assert materialized["n_dims"] == 2
    assert materialized["materialization_status"] == "materialized"
    assert materialized["freshness_status"] == "ok"

    planned = by_view["output_anchor"]
    assert planned["candidate_set_ids"] == ["all_x"]
    assert planned["source_id"] == "planned_output_features"
    assert planned["model_name"] == "evo2_7b"
    assert planned["feature_family"] == "output_layer_mean"
    assert planned["materialization_status"] == "planned"
    assert planned["freshness_status"] == "planned"
    assert planned["n_rows"] is None
    assert planned["n_dims"] is None


def test_candidate_inventory_is_published_in_snapshot_and_catalog(tmp_path) -> None:
    workspace_dir = _write_workspace(tmp_path)
    materialize_view(workspace_dir, "embedding_anchor")

    snapshot = workspace_snapshot(workspace_dir)
    catalog = workspace_catalog(workspace_dir)

    assert [row["view_id"] for row in snapshot["candidate_inventory"]] == ["embedding_anchor", "output_anchor"]
    assert [row["view_id"] for row in catalog["candidate_inventory"]] == ["embedding_anchor", "output_anchor"]


def test_candidate_inventory_is_published_in_notebook_controls(tmp_path) -> None:
    workspace_dir = _write_workspace(tmp_path)
    materialize_view(workspace_dir, "embedding_anchor")
    context = load_workspace_config(workspace_dir)

    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert [row.view_id for row in controls.candidate_inventory] == ["embedding_anchor", "output_anchor"]
    assert controls.candidate_inventory[0].n_rows == 2
    assert controls.candidate_inventory[0].n_dims == 2
    assert controls.candidate_inventory[1].materialization_status == "planned"


def test_candidate_inventory_uses_view_manifest_shape_without_loading_matrix(tmp_path, monkeypatch) -> None:
    workspace_dir = _write_workspace(tmp_path)
    materialize_view(workspace_dir, "embedding_anchor")
    context = load_workspace_config(workspace_dir)
    calls: list[Path] = []
    real_load = np.load

    def counted_load(path, *args, **kwargs):
        calls.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(view_shape_cache.np, "load", counted_load)

    rows = build_candidate_inventory(context)

    assert {row["view_id"] for row in rows} == {"embedding_anchor", "output_anchor"}
    assert {row["view_id"]: row["n_dims"] for row in rows}["embedding_anchor"] == 2
    assert calls == []
