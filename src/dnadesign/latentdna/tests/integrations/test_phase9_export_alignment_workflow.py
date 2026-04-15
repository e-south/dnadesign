"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase9_export_alignment_workflow.py

Phase 9 workflow tests for alignment-backed export bundles.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app

_RUNNER = CliRunner()


def _write_usr_dataset(root: Path, dataset: str, rows: list[dict[str, object]]) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dataset_dir / "records.parquet")


def _write_workspace_config(workspace_dir: Path, usr_root: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "stress_ethanol_cipro_latent_atlas", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_anchor_set",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                    "ctx1k": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_context_set",
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "context_key": "context_id",
                    },
                },
                "metadata": {"include": ["usr_label__primary", "densegen__plan", "context_id"]},
                "alignments": {
                    "anchor_ctx_20b": {
                        "left": "z20_1k_anchor",
                        "right": "z20_60",
                        "on": "subject_key",
                        "support": "intersection",
                    },
                    "anchor_ctx_7b": {
                        "left": "z7_1k_anchor",
                        "right": "z7_60",
                        "on": "subject_key",
                        "support": "intersection",
                    },
                },
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding_anchor_20b"},
                        "coordinate_space_id": "demo_space_20b",
                        "tags": {"model": "20b", "context": "anchor_only"},
                        "role": "primary",
                    },
                    "z20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "embedding_context_20b"},
                        "coordinate_space_id": "demo_space_20b",
                        "tags": {"model": "20b", "context": "template_1kb"},
                        "role": "primary",
                    },
                    "delta20": {
                        "derive": {
                            "kind": "vector_difference",
                            "left": "z20_1k_anchor",
                            "right": "z20_60",
                            "alignment": "anchor_ctx_20b",
                        },
                        "coordinate_space_id": "demo_space_20b",
                        "tags": {"operation": "difference"},
                        "role": "primary",
                    },
                    "z7_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding_anchor_7b"},
                        "coordinate_space_id": "demo_space_7b",
                        "tags": {"model": "7b", "context": "anchor_only"},
                        "role": "committee_member",
                    },
                    "z7_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "embedding_context_7b"},
                        "coordinate_space_id": "demo_space_7b",
                        "tags": {"model": "7b", "context": "template_1kb"},
                        "role": "committee_member",
                    },
                    "delta7": {
                        "derive": {
                            "kind": "vector_difference",
                            "left": "z7_1k_anchor",
                            "right": "z7_60",
                            "alignment": "anchor_ctx_7b",
                        },
                        "coordinate_space_id": "demo_space_7b",
                        "tags": {"operation": "difference"},
                        "role": "committee_member",
                    },
                },
                "scalars": {
                    "delta20_norm": {"derive": {"kind": "vector_norm", "view": "delta20", "norm": "l2"}},
                },
                "landmarks": {
                    "spy_p": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "spyP"},
                        "representation": {"mode": "centroid"},
                    },
                    "sul_ap": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "sulAp"},
                        "representation": {"mode": "centroid"},
                    },
                },
                "exports": {
                    "x2_primary_20b": {
                        "row_basis": "anchor_ctx_20b",
                        "blocks": [
                            {
                                "kind": "reduced_view",
                                "block_id": "z20_60_pc",
                                "source": "z20_60_pc2",
                                "feature_prefix": "z20_60",
                                "alignment": "anchor_ctx_20b",
                            },
                            {
                                "kind": "reduced_view",
                                "block_id": "delta20_pc",
                                "source": "delta20_pc2",
                                "feature_prefix": "delta20",
                            },
                            {
                                "kind": "table_columns",
                                "block_id": "landmark_distances",
                                "source": "primary_landmark_distances",
                                "columns": ["d_spy_p", "d_sul_ap"],
                                "alignment": "anchor_ctx_20b",
                            },
                            {
                                "kind": "table_columns",
                                "block_id": "delta20_scalars",
                                "source": "delta20_norm",
                                "columns": ["delta20_norm"],
                            },
                        ],
                    },
                    "x3_ablation_7b": {
                        "row_basis": "anchor_ctx_7b",
                        "blocks": [
                            {
                                "kind": "reduced_view",
                                "block_id": "z7_60_pc",
                                "source": "z7_60_pc2",
                                "feature_prefix": "z7_60",
                                "alignment": "anchor_ctx_7b",
                            },
                            {
                                "kind": "reduced_view",
                                "block_id": "delta7_pc",
                                "source": "delta7_pc2",
                                "feature_prefix": "delta7",
                            },
                        ],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_workspace_config_explicit_alignment_projection(workspace_dir: Path, usr_root: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "stress_ethanol_cipro_latent_atlas", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_anchor_set",
                        "record_key": "id",
                        "subject_key": "id",
                    },
                    "ctx1k": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_context_set",
                        "record_key": "id",
                        "subject_key": "construct__anchor_id",
                        "context_key": "context_id",
                    },
                },
                "metadata": {"include": ["usr_label__primary", "construct__anchor_id"]},
                "alignments": {
                    "anchor_ctx_explicit": {
                        "left": "z20_1k_anchor",
                        "right": "z20_60",
                        "left_on": ["construct__anchor_id"],
                        "right_on": ["id"],
                        "support": "intersection",
                    }
                },
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding_anchor_20b"},
                        "coordinate_space_id": "demo_space_20b",
                        "tags": {"model": "20b", "context": "anchor_only"},
                        "role": "primary",
                    },
                    "z20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "embedding_context_20b"},
                        "coordinate_space_id": "demo_space_20b",
                        "tags": {"model": "20b", "context": "template_1kb"},
                        "role": "primary",
                    },
                },
                "scalars": {
                    "anchor20_norm": {
                        "derive": {
                            "kind": "vector_norm",
                            "view": "z20_60",
                            "norm": "l2",
                            "output_column": "anchor20_norm",
                        }
                    }
                },
                "exports": {
                    "x0_alignment_projection": {
                        "row_basis": "anchor_ctx_explicit",
                        "blocks": [
                            {
                                "kind": "reduced_view",
                                "block_id": "z20_60_pc",
                                "source": "z20_60_pc2",
                                "feature_prefix": "z20_60",
                                "alignment": "anchor_ctx_explicit",
                            },
                            {
                                "kind": "table_columns",
                                "block_id": "anchor20_scalars",
                                "source": "anchor20_norm",
                                "columns": ["anchor20_norm"],
                                "alignment": "anchor_ctx_explicit",
                            },
                        ],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _subject_ids(table: pa.Table) -> list[str]:
    return [str(value) for value in table["subject_id"].to_pylist()]


def test_phase9_alignment_backed_export_flow(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(
        usr_root,
        "promoter/demo_anchor_set",
        [
            {
                "id": "anchor_04",
                "subject_id": "subject_04",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor_20b": [4.0, 0.0, 1.0],
                "embedding_anchor_7b": [0.0, 4.0, 1.0],
            },
            {
                "id": "anchor_01",
                "subject_id": "subject_01",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor_20b": [1.0, 0.0, 0.0],
                "embedding_anchor_7b": [0.0, 1.0, 0.0],
            },
            {
                "id": "anchor_03",
                "subject_id": "subject_03",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor_20b": [3.0, 1.0, 1.0],
                "embedding_anchor_7b": [1.0, 3.0, 1.0],
            },
            {
                "id": "anchor_02",
                "subject_id": "subject_02",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor_20b": [2.0, 0.0, 1.0],
                "embedding_anchor_7b": [0.0, 2.0, 1.0],
            },
        ],
    )
    _write_usr_dataset(
        usr_root,
        "promoter/demo_context_set",
        [
            {
                "id": "ctx_02",
                "subject_id": "subject_02",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context_20b": [3.0, 1.0, 1.0],
                "embedding_context_7b": [1.0, 3.0, 1.0],
            },
            {
                "id": "ctx_04",
                "subject_id": "subject_04",
                "context_id": "c1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context_20b": [5.0, 2.0, 1.0],
                "embedding_context_7b": [2.0, 5.0, 1.0],
            },
            {
                "id": "ctx_01",
                "subject_id": "subject_01",
                "context_id": "c1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context_20b": [2.0, 1.0, 0.0],
                "embedding_context_7b": [1.0, 2.0, 0.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)

    for view_id in ["z20_60", "z20_1k_anchor", "z7_60", "z7_1k_anchor"]:
        result = _RUNNER.invoke(
            app,
            ["view", "materialize", view_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    for alignment_id in ["anchor_ctx_20b", "anchor_ctx_7b"]:
        result = _RUNNER.invoke(
            app,
            ["alignment", "build", alignment_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    for command in [
        ["view", "derive", "delta20"],
        ["view", "derive", "delta7"],
        ["scalar", "derive", "delta20_norm"],
        [
            "distance",
            "score",
            "primary_landmark_distances",
            "--view",
            "z20_60",
            "--landmark",
            "spy_p",
            "--landmark",
            "sul_ap",
        ],
        ["view", "reduce", "z20_60", "--run-id", "z20_60_pca", "--dims", "2", "--reduced-view-id", "z20_60_pc2"],
        ["view", "reduce", "delta20", "--run-id", "delta20_pca", "--dims", "2", "--reduced-view-id", "delta20_pc2"],
        ["view", "reduce", "z7_60", "--run-id", "z7_60_pca", "--dims", "2", "--reduced-view-id", "z7_60_pc2"],
        ["view", "reduce", "delta7", "--run-id", "delta7_pca", "--dims", "2", "--reduced-view-id", "delta7_pc2"],
    ]:
        result = _RUNNER.invoke(app, [*command, "--workspace", workspace_dir.as_posix(), "--json"])
        assert result.exit_code == 0, result.stdout

    x2_result = _RUNNER.invoke(
        app,
        ["export", "matrix", "x2_primary_20b", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert x2_result.exit_code == 0, x2_result.stdout
    x2_payload = json.loads(x2_result.stdout)
    assert x2_payload["artifact_kind"] == "export_bundle"

    x3_result = _RUNNER.invoke(
        app,
        ["export", "matrix", "x3_ablation_7b", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert x3_result.exit_code == 0, x3_result.stdout

    output_root = workspace_dir / "outputs"
    basis_20b = pq.read_table(output_root / "alignments" / "anchor_ctx_20b" / "rows.parquet")
    basis_7b = pq.read_table(output_root / "alignments" / "anchor_ctx_7b" / "rows.parquet")
    assert _subject_ids(basis_20b) == ["subject_01", "subject_02", "subject_04"]
    assert _subject_ids(basis_7b) == ["subject_01", "subject_02", "subject_04"]

    x2_dir = output_root / "exports" / "x2_primary_20b"
    x2_matrix = np.load(x2_dir / "matrix.npy")
    x2_rows = pq.read_table(x2_dir / "rows.parquet")
    x2_features = pq.read_table(x2_dir / "features.parquet").to_pylist()
    assert x2_matrix.shape == (3, 7)
    assert _subject_ids(x2_rows) == _subject_ids(basis_20b)
    assert [row["feature_name"] for row in x2_features] == [
        "z20_60_pc_001",
        "z20_60_pc_002",
        "delta20_pc_001",
        "delta20_pc_002",
        "d_spy_p",
        "d_sul_ap",
        "delta20_norm",
    ]

    z20_anchor_rows = pq.read_table(output_root / "reduced_views" / "z20_60_pc2" / "rows.parquet")
    z20_anchor_matrix = np.load(output_root / "reduced_views" / "z20_60_pc2" / "matrix.npy")
    anchor_index = {subject_id: index for index, subject_id in enumerate(_subject_ids(z20_anchor_rows))}
    expected_anchor_block = z20_anchor_matrix[[anchor_index[subject_id] for subject_id in _subject_ids(basis_20b)]]
    np.testing.assert_allclose(x2_matrix[:, :2], expected_anchor_block)

    x2_manifest = json.loads((x2_dir / "manifest.json").read_text(encoding="utf-8"))
    assert x2_manifest["params"]["row_basis"] == "anchor_ctx_20b"
    assert x2_manifest["params"]["blocks"][0]["alignment_id"] == "anchor_ctx_20b"

    x3_dir = output_root / "exports" / "x3_ablation_7b"
    x3_matrix = np.load(x3_dir / "matrix.npy")
    x3_rows = pq.read_table(x3_dir / "rows.parquet")
    x3_features = pq.read_table(x3_dir / "features.parquet").to_pylist()
    assert x3_matrix.shape == (3, 4)
    assert _subject_ids(x3_rows) == _subject_ids(basis_7b)
    assert [row["feature_name"] for row in x3_features] == [
        "z7_60_pc_001",
        "z7_60_pc_002",
        "delta7_pc_001",
        "delta7_pc_002",
    ]

    z7_anchor_rows = pq.read_table(output_root / "reduced_views" / "z7_60_pc2" / "rows.parquet")
    z7_anchor_matrix = np.load(output_root / "reduced_views" / "z7_60_pc2" / "matrix.npy")
    z7_anchor_index = {subject_id: index for index, subject_id in enumerate(_subject_ids(z7_anchor_rows))}
    expected_z7_anchor_block = z7_anchor_matrix[[z7_anchor_index[subject_id] for subject_id in _subject_ids(basis_7b)]]
    np.testing.assert_allclose(x3_matrix[:, :2], expected_z7_anchor_block)


def test_phase9_export_projects_blocks_with_explicit_alignment_side_keys(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(
        usr_root,
        "promoter/demo_anchor_set",
        [
            {
                "id": "anchor_01",
                "usr_label__primary": "spyP",
                "embedding_anchor_20b": [1.0, 0.0, 0.0],
            },
            {
                "id": "anchor_02",
                "usr_label__primary": "sulAp",
                "embedding_anchor_20b": [2.0, 1.0, 0.0],
            },
            {
                "id": "anchor_03",
                "usr_label__primary": "soxSp",
                "embedding_anchor_20b": [3.0, 1.0, 1.0],
            },
        ],
    )
    _write_usr_dataset(
        usr_root,
        "promoter/demo_context_set",
        [
            {
                "id": "ctx_03",
                "construct__anchor_id": "anchor_03",
                "context_id": "c1",
                "usr_label__primary": "soxSp",
                "embedding_context_20b": [3.1, 1.1, 1.0],
            },
            {
                "id": "ctx_01",
                "construct__anchor_id": "anchor_01",
                "context_id": "c1",
                "usr_label__primary": "spyP",
                "embedding_context_20b": [1.1, 0.0, 0.0],
            },
            {
                "id": "ctx_02",
                "construct__anchor_id": "anchor_02",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "embedding_context_20b": [2.1, 1.0, 0.0],
            },
        ],
    )
    _write_workspace_config_explicit_alignment_projection(workspace_dir, usr_root)

    for command in [
        ["view", "materialize", "z20_60"],
        ["view", "materialize", "z20_1k_anchor"],
        ["alignment", "build", "anchor_ctx_explicit"],
        ["scalar", "derive", "anchor20_norm"],
        ["view", "reduce", "z20_60", "--run-id", "z20_60_pca", "--dims", "2", "--reduced-view-id", "z20_60_pc2"],
    ]:
        result = _RUNNER.invoke(app, [*command, "--workspace", workspace_dir.as_posix(), "--json"])
        assert result.exit_code == 0, result.stdout

    export_result = _RUNNER.invoke(
        app,
        ["export", "matrix", "x0_alignment_projection", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert export_result.exit_code == 0, export_result.stdout

    output_root = workspace_dir / "outputs"
    basis_rows = pq.read_table(output_root / "alignments" / "anchor_ctx_explicit" / "rows.parquet")
    export_dir = output_root / "exports" / "x0_alignment_projection"
    export_rows = pq.read_table(export_dir / "rows.parquet")
    export_matrix = np.load(export_dir / "matrix.npy")
    export_features = pq.read_table(export_dir / "features.parquet").to_pylist()

    assert basis_rows.column_names == ["construct__anchor_id", "left_count", "right_count"]
    assert export_rows.column_names == ["construct__anchor_id", "left_count", "right_count"]
    assert [row["feature_name"] for row in export_features] == [
        "z20_60_pc_001",
        "z20_60_pc_002",
        "anchor20_norm",
    ]
    assert export_matrix.shape == (3, 3)
    assert export_rows["construct__anchor_id"].to_pylist() == ["anchor_01", "anchor_02", "anchor_03"]

    reduced_rows = pq.read_table(output_root / "reduced_views" / "z20_60_pc2" / "rows.parquet")
    reduced_matrix = np.load(output_root / "reduced_views" / "z20_60_pc2" / "matrix.npy")
    reduced_index = {str(row["id"]): index for index, row in enumerate(reduced_rows.to_pylist())}
    expected_block = reduced_matrix[
        [reduced_index[anchor_id] for anchor_id in export_rows["construct__anchor_id"].to_pylist()]
    ]
    np.testing.assert_allclose(export_matrix[:, :2], expected_block)
