"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase3_reduce_export_workflow.py

Phase 3 workflow tests for reduction and deterministic export bundles.

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
                    "anchor_ctx": {
                        "left": "z20_1k_anchor",
                        "right": "z20_60",
                        "on": "subject_key",
                        "support": "intersection",
                    }
                },
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding_anchor"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo", "context": "anchor_only"},
                        "role": "primary",
                    },
                    "z20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "embedding_context"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo", "context": "template_1kb"},
                        "role": "primary",
                    },
                    "delta20": {
                        "derive": {
                            "kind": "vector_difference",
                            "left": "z20_1k_anchor",
                            "right": "z20_60",
                            "alignment": "anchor_ctx",
                        },
                        "coordinate_space_id": "demo_space",
                        "tags": {"operation": "difference"},
                        "role": "primary",
                    },
                },
                "scalars": {
                    "delta20_norm": {"derive": {"kind": "vector_norm", "view": "delta20", "norm": "l2"}},
                },
                "exports": {
                    "x_demo": {
                        "row_basis": "delta20_pc2",
                        "blocks": [
                            {
                                "kind": "reduced_view",
                                "block_id": "delta20_pc",
                                "source": "delta20_pc2",
                                "feature_prefix": "delta20",
                            },
                            {
                                "kind": "table_columns",
                                "block_id": "delta20_scalars",
                                "source": "delta20_norm",
                                "columns": ["delta20_norm"],
                            },
                        ],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_phase3_reduce_and_export_flow(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(
        usr_root,
        "promoter/demo_anchor_set",
        [
            {
                "id": "anchor_01",
                "subject_id": "subject_01",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor": [0.0, 0.0],
            },
            {
                "id": "anchor_02",
                "subject_id": "subject_02",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor": [1.0, 0.0],
            },
            {
                "id": "anchor_03",
                "subject_id": "subject_03",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [0.0, 2.0],
            },
            {
                "id": "anchor_04",
                "subject_id": "subject_04",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [1.0, 2.0],
            },
        ],
    )
    _write_usr_dataset(
        usr_root,
        "promoter/demo_context_set",
        [
            {
                "id": "ctx_01",
                "subject_id": "subject_01",
                "context_id": "c1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context": [1.0, -1.0],
            },
            {
                "id": "ctx_02",
                "subject_id": "subject_02",
                "context_id": "c1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context": [3.0, -1.0],
            },
            {
                "id": "ctx_03",
                "subject_id": "subject_03",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [1.0, 4.0],
            },
            {
                "id": "ctx_04",
                "subject_id": "subject_04",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [3.0, 5.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)

    for view_id in ["z20_60", "z20_1k_anchor"]:
        result = _RUNNER.invoke(
            app,
            ["view", "materialize", view_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    for command in [
        ["alignment", "build", "anchor_ctx"],
        ["view", "derive", "delta20"],
        ["scalar", "derive", "delta20_norm"],
    ]:
        result = _RUNNER.invoke(app, [*command, "--workspace", workspace_dir.as_posix(), "--json"])
        assert result.exit_code == 0, result.stdout

    sample_result = _RUNNER.invoke(
        app,
        [
            "sample",
            "build",
            "delta_sample",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "delta20",
            "--strategy",
            "random",
            "--n",
            "3",
            "--seed",
            "17",
            "--json",
        ],
    )
    assert sample_result.exit_code == 0, sample_result.stdout

    reduce_result = _RUNNER.invoke(
        app,
        [
            "view",
            "reduce",
            "delta20",
            "--workspace",
            workspace_dir.as_posix(),
            "--run-id",
            "delta20_pca",
            "--dims",
            "2",
            "--sample",
            "delta_sample",
            "--reduced-view-id",
            "delta20_pc2",
            "--json",
        ],
    )
    assert reduce_result.exit_code == 0, reduce_result.stdout
    reduce_payload = json.loads(reduce_result.stdout)
    assert reduce_payload["artifact_kind"] == "reducer"
    assert reduce_payload["metrics"]["fit_rows"] == 3
    assert reduce_payload["metrics"]["reduced_view_rows"] == 4

    reduced_matrix = np.load(workspace_dir / "outputs" / "reduced_views" / "delta20_pc2" / "matrix.npy")
    assert reduced_matrix.shape == (4, 2)

    export_result = _RUNNER.invoke(
        app,
        ["export", "matrix", "x_demo", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert export_result.exit_code == 0, export_result.stdout
    export_payload = json.loads(export_result.stdout)
    assert export_payload["artifact_kind"] == "export_bundle"

    export_dir = workspace_dir / "outputs" / "exports" / "x_demo"
    matrix = np.load(export_dir / "matrix.npy")
    features = pq.read_table(export_dir / "features.parquet").to_pylist()
    rows = pq.read_table(export_dir / "rows.parquet")
    assert matrix.shape == (4, 3)
    assert matrix.dtype == np.float32
    assert rows.num_rows == 4
    assert [row["feature_name"] for row in features] == ["delta20_pc_001", "delta20_pc_002", "delta20_norm"]
    assert (workspace_dir / "outputs" / "reducers" / "delta20_pca" / "manifest.json").is_file()
    assert (workspace_dir / "outputs" / "reduced_views" / "delta20_pc2" / "manifest.json").is_file()
    assert (export_dir / "manifest.json").is_file()
