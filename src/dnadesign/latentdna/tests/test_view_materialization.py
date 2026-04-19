"""Contracts for latentdna view metadata materialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.io.parquet_io import read_table
from dnadesign.latentdna.src.views.materialize import _spacer_length, materialize_view_artifact
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def test_materialize_view_emits_sig35_variant_without_sigma70_alias(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["synthetic_a", "control_a"], type=pa.string()),
                "subject_id": pa.array(["synthetic_a", "control_a"], type=pa.string()),
                "usr_label__primary": pa.array(["synthetic_a", "J23105"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol__sig35=b", None], type=pa.string()),
                "densegen__required_regulators": pa.array(["cpxR", None], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32())),
                "template_id": pa.array(["tpl_a", "wt"], type=pa.string()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "sig35_demo", "output_root": "./outputs"},
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
                    "include": [
                        "usr_label__primary",
                        "construct_template_id",
                        "design_family",
                        "sig35_variant",
                    ]
                },
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {
                    "design_family": {
                        "kind": "promoter_metadata",
                        "source": "anchor_60bp",
                        "derive": "design_family",
                    },
                    "sig35_variant": {
                        "kind": "promoter_metadata",
                        "source": "anchor_60bp",
                        "derive": "sig35_variant",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="intermediate_embedding_20b_anchor_60bp")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert [row["sig35_variant"] for row in rows] == ["b", "control"]
    assert [row["construct_template_id"] for row in rows] == ["tpl_a", "wt"]
    assert "sigma70_variant" not in rows[0]


def test_materialize_view_rejects_synthetic_rows_without_sig35_token(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["synthetic_a"], type=pa.string()),
                "subject_id": pa.array(["synthetic_a"], type=pa.string()),
                "usr_label__primary": pa.array(["synthetic_a"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol"], type=pa.string()),
                "densegen__required_regulators": pa.array(["cpxR"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "sig35_required_demo", "output_root": "./outputs"},
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
                "metadata": {"include": ["sig35_variant"]},
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {
                    "sig35_variant": {
                        "kind": "promoter_metadata",
                        "source": "anchor_60bp",
                        "derive": "sig35_variant",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)

    with pytest.raises(ContractViolationError, match="sig35_variant"):
        materialize_view_artifact(context, view_id="intermediate_embedding_20b_anchor_60bp")


def test_materialize_view_includes_source_scoped_metadata_columns(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
                "usr_label__primary": pa.array(["row_a"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol__sig35=b"], type=pa.string()),
                "densegen__required_regulators": pa.array(["cpxR"], type=pa.string()),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
                "anchor_logp": pa.array([-1.5], type=pa.float64()),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "source_metadata_include_demo", "output_root": "./outputs"},
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
                        "metadata_include": ["anchor_logp"],
                    }
                },
                "metadata": {"include": ["sig35_variant"]},
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {
                    "sig35_variant": {
                        "kind": "promoter_metadata",
                        "source": "anchor_60bp",
                        "derive": "sig35_variant",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="intermediate_embedding_20b_anchor_60bp")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert rows == [{"id": "row_a", "subject_id": "row_a", "anchor_logp": -1.5, "sig35_variant": "b"}]


def test_materialize_view_includes_source_promoter_cohorts_without_metadata_include(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["row_a"], type=pa.string()),
                "subject_id": pa.array(["row_a"], type=pa.string()),
                "usr_label__primary": pa.array(["row_a"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol__sig35=b"], type=pa.string()),
                "densegen__used_tfbs_detail": pa.array(
                    ['[{"part_kind":"fixed_element","spacer_length":17}]'], type=pa.string()
                ),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "promoter_cohort_auto_include_demo", "output_root": "./outputs"},
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
                "metadata": {"include": []},
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {
                    "spacer_length": {
                        "kind": "promoter_metadata",
                        "source": "anchor_60bp",
                        "derive": "spacer_length",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="intermediate_embedding_20b_anchor_60bp")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert rows == [{"id": "row_a", "subject_id": "row_a", "spacer_length": 17}]


def test_materialize_view_ignores_promoter_metadata_cohorts_from_other_sources(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["anchor_a"], type=pa.string()),
                "subject_id": pa.array(["anchor_a"], type=pa.string()),
                "usr_label__primary": pa.array(["anchor_a"], type=pa.string()),
                "densegen__plan": pa.array(["ethanol__sig35=b"], type=pa.string()),
                "densegen__used_tfbs_detail": pa.array(
                    ['[{"part_kind":"sigma35","variant_id":"b","spacer_length":17}]']
                ),
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "anchor.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["control_a"], type=pa.string()),
                "subject_id": pa.array(["control_a"], type=pa.string()),
                "usr_label__primary": pa.array(["control_a"], type=pa.string()),
                "embedding": pa.array([[1.0, 0.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "control.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "source_scoped_cohort_demo", "output_root": "./outputs"},
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
                        "path": "inputs/anchor.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                    "controls": {
                        "kind": "parquet",
                        "path": "inputs/control.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                },
                "metadata": {"include": []},
                "views": {
                    "control_embedding": {
                        "source": "controls",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
                "cohorts": {
                    "spacer_length": {
                        "kind": "promoter_metadata",
                        "source": "anchor_60bp",
                        "derive": "spacer_length",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    artifact_dir, *_ = materialize_view_artifact(context, view_id="control_embedding")
    rows = read_table(artifact_dir / "rows.parquet").to_pylist()

    assert rows == [{"id": "control_a", "subject_id": "control_a"}]


def test_spacer_length_accepts_numpy_object_array_entries() -> None:
    row = {
        "usr_label__primary": "synthetic_a",
        "densegen__plan": "ethanol__sig35=f",
        "densegen__used_tfbs_detail": np.array(
            [
                {
                    "part_kind": "tfbs",
                    "spacer_length": None,
                },
                {
                    "part_kind": "fixed_element",
                    "role": "upstream",
                    "spacer_length": 17,
                },
                {
                    "part_kind": "fixed_element",
                    "role": "downstream",
                    "spacer_length": 17,
                },
            ],
            dtype=object,
        ),
    }

    assert _spacer_length(row) == 17


def test_spacer_length_returns_none_when_synthetic_detail_is_missing() -> None:
    row = {
        "usr_label__primary": "synthetic_a",
        "densegen__plan": "ethanol__sig35=f",
        "densegen__used_tfbs_detail": None,
    }

    assert _spacer_length(row) is None
