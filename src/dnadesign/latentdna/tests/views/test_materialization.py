"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/views/test_materialization.py

Contracts for latentdna view metadata materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.io.parquet_io import read_table
from dnadesign.latentdna.src.views.materialize import _derived_metadata_array, materialize_view_artifact
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def test_materialize_view_uses_lookup_derivation_for_parent_metadata(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["parent_a", "parent_b"], type=pa.string()),
                "sigma_factor": pa.array(["sigma70", "sigma38"], type=pa.string()),
            }
        ),
        inputs_dir / "parents.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["child_a", "child_b"], type=pa.string()),
                "subject_id": pa.array(["child_a", "child_b"], type=pa.string()),
                "parent_id": pa.array(["parent_a", "parent_b"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "children.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "lookup_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "parents": {
                        "kind": "parquet",
                        "path": "inputs/parents.parquet",
                        "record_key": "id",
                        "subject_key": "id",
                    },
                    "children": {
                        "kind": "parquet",
                        "path": "inputs/children.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                },
                "metadata": {
                    "include": ["sigma_factor"],
                    "derivations": {
                        "sigma_factor": {
                            "kind": "lookup",
                            "source": "parents",
                            "left_key": "parent_id",
                            "right_key": "id",
                            "value_column": "sigma_factor",
                        }
                    },
                },
                "views": {
                    "child_embedding": {
                        "source": "children",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="child_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["sigma_factor"] for row in rows] == ["sigma70", "sigma38"]


def test_materialize_view_lookup_derivation_fails_on_missing_parent_match(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["parent_a"], type=pa.string()),
                "sigma_factor": pa.array(["sigma70"], type=pa.string()),
            }
        ),
        inputs_dir / "parents.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["child_a", "child_b"], type=pa.string()),
                "subject_id": pa.array(["child_a", "child_b"], type=pa.string()),
                "parent_id": pa.array(["parent_a", "missing_parent"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "children.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "lookup_missing_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "parents": {
                        "kind": "parquet",
                        "path": "inputs/parents.parquet",
                        "record_key": "id",
                        "subject_key": "id",
                    },
                    "children": {
                        "kind": "parquet",
                        "path": "inputs/children.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                },
                "metadata": {
                    "include": ["sigma_factor"],
                    "derivations": {
                        "sigma_factor": {
                            "kind": "lookup",
                            "source": "parents",
                            "left_key": "parent_id",
                            "right_key": "id",
                            "value_column": "sigma_factor",
                        }
                    },
                },
                "views": {
                    "child_embedding": {
                        "source": "children",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)

    with pytest.raises(ContractViolationError, match="missing lookup matches"):
        materialize_view_artifact(context, view_id="child_embedding")


def test_materialize_view_lookup_derivation_can_fill_missing_matches_with_default(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["parent_a"], type=pa.string()),
                "label_status": pa.array(["observed"], type=pa.string()),
            }
        ),
        inputs_dir / "parents.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["child_a", "child_b"], type=pa.string()),
                "subject_id": pa.array(["child_a", "child_b"], type=pa.string()),
                "parent_id": pa.array(["parent_a", "missing_parent"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "children.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "lookup_default_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "parents": {
                        "kind": "parquet",
                        "path": "inputs/parents.parquet",
                        "record_key": "id",
                        "subject_key": "id",
                    },
                    "children": {
                        "kind": "parquet",
                        "path": "inputs/children.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                },
                "metadata": {
                    "include": ["label_status"],
                    "derivations": {
                        "label_status": {
                            "kind": "lookup",
                            "source": "parents",
                            "left_key": "parent_id",
                            "right_key": "id",
                            "value_column": "label_status",
                            "missing_policy": "null",
                            "default": "missing",
                        }
                    },
                },
                "views": {
                    "child_embedding": {
                        "source": "children",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="child_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["label_status"] for row in rows] == ["observed", "missing"]


def test_construct_template_id_derivation_uses_stable_string_type_for_all_null_batch(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace: {id: stable_construct_template_type_demo, output_root: ./outputs}
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources: {}
metadata:
  derivations:
    construct_template_id:
      kind: coalesce
      sources: [construct_template_id, template_id, construct__template_id]
      value_type: string
views: {}
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    context = load_workspace_config(workspace_dir)

    array = _derived_metadata_array(context, [{"construct__template_id": None}], column_name="construct_template_id")

    assert array.type == pa.string()
    assert array.to_pylist() == [None]


def test_materialize_view_populates_coalesced_metadata_from_source_columns(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a", "row_b"], type=pa.string()),
                "subject_id": pa.array(["row_a", "row_b"], type=pa.string()),
                "usr_label__primary": pa.array(["row_a", "row_b"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
                "score_anchor": pa.array([-1.2, None], type=pa.float32()),
                "score_context": pa.array([None, -6.3], type=pa.float32()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "coalesce_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_60bp": {
                        "kind": "parquet",
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["llr_demo"],
                    "derivations": {
                        "llr_demo": {
                            "kind": "coalesce",
                            "sources": ["score_anchor", "score_context"],
                        }
                    },
                },
                "views": {
                    "demo_embedding": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="demo_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["llr_demo"] for row in rows] == pytest.approx([-1.2, -6.3])


def test_materialize_view_allows_coalesce_derivation_when_one_input_column_is_available(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
                "score_context": pa.array([-6.3], type=pa.float32()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "missing_derivation_input_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_60bp": {
                        "kind": "parquet",
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["llr_demo"],
                    "derivations": {
                        "llr_demo": {
                            "kind": "coalesce",
                            "sources": ["score_anchor", "score_context"],
                        }
                    },
                },
                "views": {
                    "demo_embedding": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="demo_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["llr_demo"] for row in rows] == pytest.approx([-6.3])


def test_materialize_view_fails_fast_when_all_coalesce_inputs_are_missing(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "missing_all_derivation_inputs_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_60bp": {
                        "kind": "parquet",
                        "path": "inputs/records.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {
                    "include": ["llr_demo"],
                    "derivations": {
                        "llr_demo": {
                            "kind": "coalesce",
                            "sources": ["score_anchor", "score_context"],
                        }
                    },
                },
                "views": {
                    "demo_embedding": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)

    with pytest.raises(ContractViolationError, match="metadata derivation inputs are missing"):
        materialize_view_artifact(context, view_id="demo_embedding")
