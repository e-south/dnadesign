"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase2_alignment_distance_workflow.py

Phase 2/3 workflow tests for alignment, derived views, scalars, and distances.

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
                    "ethanol_vs_cipro": {
                        "derive": {
                            "kind": "column_expression",
                            "source": "primary_landmark_distances",
                            "expression": "d_sul_ap - d_spy_p",
                            "output_column": "ethanol_vs_cipro",
                        }
                    },
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_workspace_config_explicit_alignment_keys(workspace_dir: Path, usr_root: Path) -> None:
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
                "metadata": {"include": ["usr_label__primary"]},
                "alignments": {
                    "anchor_ctx": {
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
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_phase2_alignment_distance_and_scalar_flow(tmp_path: Path) -> None:
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
                "embedding_context": [2.0, -1.0],
            },
            {
                "id": "ctx_03",
                "subject_id": "subject_03",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [1.0, 1.0],
            },
            {
                "id": "ctx_04",
                "subject_id": "subject_04",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [2.0, 1.0],
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

    alignment_result = _RUNNER.invoke(
        app,
        ["alignment", "build", "anchor_ctx", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert alignment_result.exit_code == 0, alignment_result.stdout
    alignment_payload = json.loads(alignment_result.stdout)
    assert alignment_payload["artifact_kind"] == "alignment_set"

    derive_result = _RUNNER.invoke(
        app,
        ["view", "derive", "delta20", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert derive_result.exit_code == 0, derive_result.stdout
    delta_dir = workspace_dir / "outputs" / "views" / "delta20"
    delta_matrix = np.load(delta_dir / "matrix.npy")
    np.testing.assert_allclose(delta_matrix, np.asarray([[1.0, -1.0]] * 4, dtype=np.float32))

    scalar_result = _RUNNER.invoke(
        app,
        ["scalar", "derive", "delta20_norm", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert scalar_result.exit_code == 0, scalar_result.stdout
    scalar_table = pq.read_table(workspace_dir / "outputs" / "scalars" / "delta20_norm" / "table.parquet")
    np.testing.assert_allclose(scalar_table.column("delta20_norm").to_pylist(), [np.sqrt(2.0)] * 4)

    distance_result = _RUNNER.invoke(
        app,
        [
            "distance",
            "score",
            "primary_landmark_distances",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z20_60",
            "--landmark",
            "spy_p",
            "--landmark",
            "sul_ap",
            "--json",
        ],
    )
    assert distance_result.exit_code == 0, distance_result.stdout
    distance_table = pq.read_table(
        workspace_dir / "outputs" / "distances" / "primary_landmark_distances" / "table.parquet"
    )
    assert "d_spy_p" in distance_table.column_names
    assert "d_sul_ap" in distance_table.column_names

    expression_result = _RUNNER.invoke(
        app,
        ["scalar", "derive", "ethanol_vs_cipro", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert expression_result.exit_code == 0, expression_result.stdout
    expression_table = pq.read_table(workspace_dir / "outputs" / "scalars" / "ethanol_vs_cipro" / "table.parquet")
    assert "ethanol_vs_cipro" in expression_table.column_names
    assert (workspace_dir / "outputs" / "alignments" / "anchor_ctx" / "manifest.json").is_file()
    assert (workspace_dir / "outputs" / "distances" / "primary_landmark_distances" / "manifest.json").is_file()


def test_alignment_build_supports_explicit_left_and_right_key_columns(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(
        usr_root,
        "promoter/demo_anchor_set",
        [
            {"id": "anchor_01", "usr_label__primary": "spyP", "embedding_anchor": [0.0, 0.0]},
            {"id": "anchor_02", "usr_label__primary": "sulAp", "embedding_anchor": [1.0, 0.0]},
        ],
    )
    _write_usr_dataset(
        usr_root,
        "promoter/demo_context_set",
        [
            {
                "id": "ctx_01",
                "construct__anchor_id": "anchor_01",
                "context_id": "c1",
                "usr_label__primary": "spyP",
                "embedding_context": [0.0, 1.0],
            },
            {
                "id": "ctx_02",
                "construct__anchor_id": "anchor_02",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "embedding_context": [1.0, 1.0],
            },
        ],
    )
    _write_workspace_config_explicit_alignment_keys(workspace_dir, usr_root)

    for view_id in ["z20_60", "z20_1k_anchor"]:
        result = _RUNNER.invoke(
            app,
            ["view", "materialize", view_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    alignment_result = _RUNNER.invoke(
        app,
        ["alignment", "build", "anchor_ctx", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert alignment_result.exit_code == 0, alignment_result.stdout
    payload = json.loads(alignment_result.stdout)
    assert payload["artifact_kind"] == "alignment_set"
    assert payload["metrics"]["rows"] == 2
