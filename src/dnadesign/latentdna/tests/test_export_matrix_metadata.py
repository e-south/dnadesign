"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_export_matrix_metadata.py

Regression tests for export matrix metadata LatentDNA.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.latentdna.src.exports.matrix import build_export_matrix_artifact
from dnadesign.latentdna.src.io.parquet_io import read_table
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def test_export_matrix_preserves_metadata_columns_after_alignment_projection(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    outputs_dir = workspace_dir / "outputs"
    (outputs_dir / "alignments" / "anchor_alignment").mkdir(parents=True)
    (outputs_dir / "reduced_views" / "demo_reduced").mkdir(parents=True)
    (outputs_dir / "reduced_views" / "demo_reduced_nulls").mkdir(parents=True)

    pq.write_table(
        pa.table(
            {
                "anchor_id": pa.array(["anchor_a", "anchor_b"], type=pa.string()),
                "left_count": pa.array([1, 1], type=pa.int64()),
                "right_count": pa.array([1, 1], type=pa.int64()),
            }
        ),
        outputs_dir / "alignments" / "anchor_alignment" / "rows.parquet",
    )
    (outputs_dir / "alignments" / "anchor_alignment" / "manifest.json").write_text(
        json.dumps({"params": {"key_columns": ["anchor_id"], "right_key_columns": ["id"]}}),
        encoding="utf-8",
    )

    np.save(outputs_dir / "reduced_views" / "demo_reduced" / "matrix.npy", np.asarray([[1.0], [2.0]], dtype=np.float32))
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["anchor_a", "anchor_b"], type=pa.string()),
                "sig35_variant": pa.array(["b", "c"], type=pa.string()),
                "source_class": pa.array(["densegen", "densegen"], type=pa.string()),
                "construct_template_id": pa.array(["tpl_a", "tpl_b"], type=pa.string()),
            }
        ),
        outputs_dir / "reduced_views" / "demo_reduced" / "rows.parquet",
    )
    np.save(
        outputs_dir / "reduced_views" / "demo_reduced_nulls" / "matrix.npy",
        np.asarray([[3.0], [4.0]], dtype=np.float32),
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["anchor_a", "anchor_b"], type=pa.string()),
                "sig35_variant": pa.array(["b", "c"], type=pa.string()),
                "source_class": pa.array(["densegen", "densegen"], type=pa.string()),
                "construct_template_id": pa.array([None, None], type=pa.string()),
            }
        ),
        outputs_dir / "reduced_views" / "demo_reduced_nulls" / "rows.parquet",
    )

    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "export_metadata_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "left_source": {
                        "kind": "parquet",
                        "path": "inputs/left.parquet",
                        "record_key": "anchor_id",
                        "subject_key": "anchor_id",
                    },
                    "right_source": {
                        "kind": "parquet",
                        "path": "inputs/right.parquet",
                        "record_key": "id",
                        "subject_key": "id",
                    },
                },
                "alignments": {
                    "anchor_alignment": {
                        "left": "left_rows",
                        "right": "right_rows",
                        "left_on": ["anchor_id"],
                        "right_on": ["id"],
                    }
                },
                "exports": {
                    "candidate_feature_bundle": {
                        "row_basis": "anchor_alignment",
                        "metadata_columns": ["sig35_variant", "source_class", "construct_template_id"],
                        "blocks": [
                            {
                                "kind": "reduced_view",
                                "block_id": "demo",
                                "source": "demo_reduced",
                                "feature_prefix": "demo",
                                "alignment": "anchor_alignment",
                            },
                            {
                                "kind": "reduced_view",
                                "block_id": "demo_nulls",
                                "source": "demo_reduced_nulls",
                                "feature_prefix": "demo_nulls",
                                "alignment": "anchor_alignment",
                            },
                        ],
                    }
                },
                "views": {
                    "left_rows": {
                        "source": "left_source",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                    },
                    "right_rows": {
                        "source": "right_source",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    export_dir, *_ = build_export_matrix_artifact(context, export_id="candidate_feature_bundle")
    rows = read_table(export_dir / "rows.parquet").to_pylist()

    assert rows == [
        {
            "anchor_id": "anchor_a",
            "left_count": 1,
            "right_count": 1,
            "sig35_variant": "b",
            "source_class": "densegen",
            "construct_template_id": "tpl_a",
        },
        {
            "anchor_id": "anchor_b",
            "left_count": 1,
            "right_count": 1,
            "sig35_variant": "c",
            "source_class": "densegen",
            "construct_template_id": "tpl_b",
        },
    ]
