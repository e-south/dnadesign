"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_scalar_builder_landmark_fallback.py

Integration coverage for same-source landmark selection without selector columns in view rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.latentdna.src.io.parquet_io import read_table
from dnadesign.latentdna.src.services.alignment_service import build_alignment
from dnadesign.latentdna.src.services.scalar_service import build_scalar
from dnadesign.latentdna.src.services.view_service import materialize_view


def _write_workspace(workspace_dir: Path) -> None:
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": ["spyP_row", "j23105_row", "sample_row"],
                "subject_id": ["spyP_row", "j23105_row", "sample_row"],
                "usr_label__primary": ["spyP", "J23105", "sample"],
                "embedding": [[1.0, 0.0], [0.0, 1.0], [0.75, 0.25]],
            }
        ),
        inputs_dir / "anchor.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "scalar_landmark_fallback_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_60bp": {
                        "kind": "parquet",
                        "path": "inputs/anchor.parquet",
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
                        "tags": {
                            "model": "20b",
                            "family": "intermediate_embedding",
                            "scope": "anchor_60bp",
                        },
                    }
                },
                "landmarks": {
                    "spyp": {
                        "source": "anchor_60bp",
                        "where": {"column": "usr_label__primary", "equals": "spyP"},
                        "representation": {"mode": "centroid"},
                    },
                    "j23105": {
                        "source": "anchor_60bp",
                        "where": {"column": "usr_label__primary", "equals": "J23105"},
                        "representation": {"mode": "centroid"},
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_alignment_workspace(workspace_dir: Path) -> None:
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": ["spyP_row", "j23105_row", "sample_row"],
                "subject_id": ["spyP_row", "j23105_row", "sample_row"],
                "usr_label__primary": ["spyp", "J23105", "sample"],
                "embedding": [[1.0, 0.0], [0.0, 1.0], [0.75, 0.25]],
            }
        ),
        inputs_dir / "anchor.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": ["ctx_spyp", "ctx_j23105", "ctx_sample"],
                "construct__anchor_id": ["spyP_row", "j23105_row", "sample_row"],
                "embedding": [[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]],
            }
        ),
        inputs_dir / "full_context.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "scalar_alignment_landmark_fallback_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_60bp": {
                        "kind": "parquet",
                        "path": "inputs/anchor.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                    "full_context_1kb": {
                        "kind": "parquet",
                        "path": "inputs/full_context.parquet",
                        "record_key": "id",
                        "subject_key": "construct__anchor_id",
                    },
                },
                "metadata": {"include": []},
                "alignments": {
                    "anchor_to_full_context": {
                        "left": "intermediate_embedding_20b_full_context_1kb",
                        "right": "intermediate_embedding_20b_anchor_60bp",
                        "left_on": ["construct__anchor_id"],
                        "right_on": ["id"],
                    }
                },
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {
                            "model": "20b",
                            "family": "intermediate_embedding",
                            "scope": "anchor_60bp",
                        },
                    },
                    "intermediate_embedding_20b_full_context_1kb": {
                        "source": "full_context_1kb",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {
                            "model": "20b",
                            "family": "intermediate_embedding",
                            "scope": "full_context_1kb",
                        },
                    },
                },
                "landmarks": {
                    "spyp": {
                        "source": "anchor_60bp",
                        "where": {"column": "usr_label__primary", "equals": "spyp"},
                        "representation": {"mode": "centroid"},
                    },
                    "j23105": {
                        "source": "anchor_60bp",
                        "where": {"column": "usr_label__primary", "equals": "J23105"},
                        "representation": {"mode": "centroid"},
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_similarity_margin_builder_projects_landmarks_from_source_when_view_rows_omit_selector(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace(workspace_dir)

    materialize_view(workspace_dir, "intermediate_embedding_20b_anchor_60bp", force=True)

    view_rows = read_table(
        workspace_dir / "outputs" / "views" / "intermediate_embedding_20b_anchor_60bp" / "rows.parquet"
    )
    assert "usr_label__primary" not in view_rows.column_names

    build_scalar(
        workspace_dir,
        "wildtype_reference_margins_intermediate_embedding_20b_anchor_60bp",
        builder_kind="similarity_margin",
        params={
            "view_id": "intermediate_embedding_20b_anchor_60bp",
            "margin_pairs": [
                {
                    "target_landmark": "spyp",
                    "control_landmark": "j23105",
                    "output_column": "wildtype_margin_ethanol_vs_control",
                }
            ],
        },
        force=True,
    )

    scalar_table = read_table(
        workspace_dir
        / "outputs"
        / "scalars"
        / "wildtype_reference_margins_intermediate_embedding_20b_anchor_60bp"
        / "table.parquet"
    )
    assert "wildtype_margin_ethanol_vs_control" in scalar_table.column_names
    assert "usr_label__primary" in scalar_table.column_names
    rows = {row["id"]: row["wildtype_margin_ethanol_vs_control"] for row in scalar_table.to_pylist()}
    assert rows["spyP_row"] > rows["j23105_row"]


def test_similarity_margin_builder_projects_alignment_landmarks_from_source_when_rows_omit_selector(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_alignment_workspace(workspace_dir)

    materialize_view(workspace_dir, "intermediate_embedding_20b_anchor_60bp", force=True)
    materialize_view(workspace_dir, "intermediate_embedding_20b_full_context_1kb", force=True)
    build_alignment(workspace_dir, "anchor_to_full_context", force=True)

    build_scalar(
        workspace_dir,
        "wildtype_reference_margins_intermediate_embedding_20b_full_context_1kb",
        builder_kind="similarity_margin",
        params={
            "view_id": "intermediate_embedding_20b_full_context_1kb",
            "alignment_id": "anchor_to_full_context",
            "margin_pairs": [
                {
                    "target_landmark": "spyp",
                    "control_landmark": "j23105",
                    "output_column": "wildtype_margin_ethanol_vs_control",
                }
            ],
        },
        force=True,
    )

    scalar_table = read_table(
        workspace_dir
        / "outputs"
        / "scalars"
        / "wildtype_reference_margins_intermediate_embedding_20b_full_context_1kb"
        / "table.parquet"
    )
    assert "usr_label__primary" in scalar_table.column_names
    rows = {row["construct__anchor_id"]: row["wildtype_margin_ethanol_vs_control"] for row in scalar_table.to_pylist()}
    assert rows["spyP_row"] > rows["j23105_row"]
