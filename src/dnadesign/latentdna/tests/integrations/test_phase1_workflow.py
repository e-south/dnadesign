"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase1_workflow.py

Phase 1 tracer-bullet workflow tests for latentdna.

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

from dnadesign.latentdna.cli import app

_RUNNER = CliRunner()


def _write_usr_dataset(root: Path, dataset: str) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"row_{index:02d}",
            "subject_id": f"subject_{index:02d}",
            "usr_label__primary": "spyP" if index % 2 == 0 else "sulAp",
            "densegen__plan": "plan_a" if index % 3 == 0 else "plan_b",
            "embedding": [float(index), float(index % 5), float(index % 7), float(index % 11)],
        }
        for index in range(18)
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, dataset_dir / "records.parquet")


def _write_workspace_config(workspace_dir: Path, usr_root: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "stress_ethanol_cipro_latent_atlas", "output_root": "./outputs/latentdna"},
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
                    }
                },
                "metadata": {"include": ["usr_label__primary", "densegen__plan"]},
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo", "context": "anchor_only"},
                        "role": "primary",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_phase1_usr_to_view_to_projection_to_plot_flow(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set")
    _write_workspace_config(workspace_dir, usr_root)

    inspect_result = _RUNNER.invoke(
        app,
        ["inspect", "source", "anchor60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert inspect_result.exit_code == 0, inspect_result.stdout
    inspect_payload = json.loads(inspect_result.stdout)
    assert inspect_payload["status"] == "ok"
    assert inspect_payload["data"]["row_count"] == 18
    assert "embedding" in inspect_payload["data"]["vector_columns"]

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout
    materialize_payload = json.loads(materialize_result.stdout)
    assert materialize_payload["artifact_kind"] == "view"

    view_dir = workspace_dir / "outputs" / "latentdna" / "views" / "z20_60"
    assert (view_dir / "matrix.npy").is_file()
    assert (view_dir / "rows.parquet").is_file()

    matrix = np.load(view_dir / "matrix.npy")
    assert matrix.dtype == np.float32
    assert matrix.shape == (18, 4)

    sample_result = _RUNNER.invoke(
        app,
        [
            "sample",
            "build",
            "atlas_sample",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z20_60",
            "--strategy",
            "stratified",
            "--group-column",
            "usr_label__primary",
            "--n",
            "10",
            "--seed",
            "17",
            "--json",
        ],
    )
    assert sample_result.exit_code == 0, sample_result.stdout
    sample_payload = json.loads(sample_result.stdout)
    assert sample_payload["artifact_kind"] == "sample_set"
    assert sample_payload["metrics"]["rows"] == 10

    projection_result = _RUNNER.invoke(
        app,
        [
            "projection",
            "fit",
            "z20_60",
            "--workspace",
            workspace_dir.as_posix(),
            "--sample",
            "atlas_sample",
            "--run-id",
            "umap_z20_60",
            "--seed",
            "17",
            "--json",
        ],
    )
    assert projection_result.exit_code == 0, projection_result.stdout
    projection_payload = json.loads(projection_result.stdout)
    assert projection_payload["artifact_kind"] == "projection"

    projection_dir = workspace_dir / "outputs" / "latentdna" / "projections" / "umap_z20_60"
    coords = pq.read_table(projection_dir / "coords.parquet").to_pydict()
    assert len(coords["x"]) == 10
    assert len(coords["y"]) == 10

    plot_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "atlas_scatter",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "projection_scatter",
            "--projection",
            "umap_z20_60",
            "--color-column",
            "usr_label__primary",
            "--json",
        ],
    )
    assert plot_result.exit_code == 0, plot_result.stdout
    plot_payload = json.loads(plot_result.stdout)
    assert plot_payload["artifact_kind"] == "plot"

    plot_dir = workspace_dir / "outputs" / "latentdna" / "plots" / "atlas_scatter"
    assert (plot_dir / "plot.svg").is_file()
    assert (plot_dir / "plot.png").is_file()
    assert (plot_dir / "manifest.json").is_file()


def test_phase1_plot_render_force_preserves_existing_artifact_on_failure(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set")
    _write_workspace_config(workspace_dir, usr_root)

    assert (
        _RUNNER.invoke(
            app,
            ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
        ).exit_code
        == 0
    )
    assert (
        _RUNNER.invoke(
            app,
            [
                "sample",
                "build",
                "atlas_sample",
                "--workspace",
                workspace_dir.as_posix(),
                "--view",
                "z20_60",
                "--strategy",
                "all",
                "--json",
            ],
        ).exit_code
        == 0
    )
    assert (
        _RUNNER.invoke(
            app,
            [
                "projection",
                "fit",
                "z20_60",
                "--workspace",
                workspace_dir.as_posix(),
                "--sample",
                "atlas_sample",
                "--run-id",
                "umap_z20_60",
                "--json",
            ],
        ).exit_code
        == 0
    )

    first_render = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "atlas_scatter",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "projection_scatter",
            "--projection",
            "umap_z20_60",
            "--json",
        ],
    )
    assert first_render.exit_code == 0, first_render.stdout

    plot_dir = workspace_dir / "outputs" / "latentdna" / "plots" / "atlas_scatter"
    assert (plot_dir / "plot.svg").is_file()
    assert (plot_dir / "plot.png").is_file()
    assert (plot_dir / "manifest.json").is_file()

    failed_force = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "atlas_scatter",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "projection_scatter",
            "--projection",
            "missing_projection",
            "--force",
            "--json",
        ],
    )
    assert failed_force.exit_code != 0
    assert "projection artifact is missing" in failed_force.stdout
    assert (plot_dir / "plot.svg").is_file()
    assert (plot_dir / "plot.png").is_file()
    assert (plot_dir / "manifest.json").is_file()


def test_phase1_plot_render_rejects_non_identifier_plot_id(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set")
    _write_workspace_config(workspace_dir, usr_root)

    invalid_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "../atlas_scatter",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "projection_scatter",
            "--projection",
            "umap_z20_60",
            "--json",
        ],
    )
    assert invalid_result.exit_code != 0
    assert "plot id must use lowercase snake_case" in invalid_result.stdout
