"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_source_projection_plot_workflow.py

Tracer-bullet workflow tests for latentdna source, projection, and plot flows.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app
from dnadesign.latentdna.src.services.operation_lock_service import operation_lock_path

_RUNNER = CliRunner()
_LOCK_HOLDER_SCRIPT = """
import fcntl
import pathlib
import sys
import time

lock_path = pathlib.Path(sys.argv[1])
ready_path = pathlib.Path(sys.argv[2])
stop_path = pathlib.Path(sys.argv[3])
lock_path.parent.mkdir(parents=True, exist_ok=True)
with lock_path.open("a+", encoding="utf-8") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    ready_path.write_text("ready", encoding="utf-8")
    while not stop_path.exists():
        time.sleep(0.05)
"""


def _write_plot_semantics(workspace_dir: Path, plot_id: str) -> str:
    semantics_ref = f"plot_semantics/{plot_id}.yaml"
    semantics_path = workspace_dir / semantics_ref
    semantics_path.parent.mkdir(parents=True, exist_ok=True)
    semantics_path.write_text(
        yaml.safe_dump(
            {
                "plot_id": plot_id,
                "research_question": f"What does {plot_id} show?",
                "evidence_tier": "qc",
                "encoding_summary": f"QC fixture semantics for {plot_id}.",
                "sampling_scope": "Fixture-sized workflow sample.",
                "interpretation_guardrails": ["Fixture semantics are descriptive only."],
                "caption_md": f"QC fixture plot for {plot_id}.",
                "alt_text": f"QC fixture plot for {plot_id}.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return semantics_ref


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


def _write_workspace_config(
    workspace_dir: Path,
    usr_root: Path,
    *,
    metadata_include: list[str] | None = None,
    reference_sets: dict[str, object] | None = None,
    plots: dict[str, object] | None = None,
) -> None:
    metadata_columns = ["usr_label__primary", "densegen__plan"] if metadata_include is None else metadata_include
    resolved_plots: dict[str, object] = {}
    for plot_id, config in (plots or {}).items():
        plot_config = dict(config)
        plot_config["semantics_ref"] = plot_config.get("semantics_ref") or _write_plot_semantics(workspace_dir, plot_id)
        resolved_plots[plot_id] = plot_config
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
                    }
                },
                "metadata": {"include": metadata_columns},
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo", "context": "anchor_only"},
                        "role": "primary",
                    }
                },
                "reference_sets": reference_sets or {},
                "plots": resolved_plots,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _wait_for_file(path: Path, *, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _append_projection_recipe(workspace_dir: Path, *, recipe_id: str) -> None:
    config_path = workspace_dir / "config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload.setdefault("recipes", {})[recipe_id] = {
        "steps": [
            {
                "id": f"{recipe_id}_fit_projection",
                "op": "projection.fit",
                "params": {
                    "view": "z20_60",
                    "sample": "atlas_sample",
                    "run_id": "umap_z20_60",
                    "seed": 17,
                },
            }
        ]
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_usr_to_view_to_projection_to_plot_flow(tmp_path: Path) -> None:
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

    view_dir = workspace_dir / "outputs" / "views" / "z20_60"
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

    projection_dir = workspace_dir / "outputs" / "projections" / "umap_z20_60"
    coords = pq.read_table(projection_dir / "coords.parquet").to_pydict()
    assert len(coords["x"]) == 10
    assert len(coords["y"]) == 10
    projection_staging_root = workspace_dir / "outputs" / "runs" / "_staging" / "projections"
    assert not projection_staging_root.exists() or not any(projection_staging_root.iterdir())

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

    plot_dir = workspace_dir / "outputs" / "plots" / "atlas_scatter"
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


def test_inline_plot_render_force_preserves_existing_artifact_on_failure(tmp_path: Path) -> None:
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

    plot_dir = workspace_dir / "outputs" / "plots" / "atlas_scatter"
    assert (plot_dir / "plot.svg").is_file()
    assert (plot_dir / "plot.png").is_file()
    assert (plot_dir / "manifest.json").is_file()


def test_projection_annotations_work_when_reference_set_columns_are_not_public_metadata(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set")
    _write_workspace_config(
        workspace_dir,
        usr_root,
        metadata_include=["densegen__plan"],
        reference_sets={
            "promoter_wt_core": {
                "ids": ["spyP", "sulAp"],
                "match_column": "usr_label__primary",
                "label_column": "usr_label__primary",
                "label_mode": "label_and_highlight",
            }
        },
        plots={
            "atlas_scatter": {
                "kind": "projection_scatter",
                "projection": "umap_z20_60",
                "annotation": {"reference_set": "promoter_wt_core"},
            }
        },
    )

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout

    view_dir = workspace_dir / "outputs" / "views" / "z20_60"
    view_rows = pq.read_table(view_dir / "rows.parquet").to_pydict()
    assert "usr_label__primary" in view_rows

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
            "densegen__plan",
            "--reference-set",
            "promoter_wt_core",
            "--n",
            "6",
            "--seed",
            "17",
            "--json",
        ],
    )
    assert sample_result.exit_code == 0, sample_result.stdout

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

    projection_dir = workspace_dir / "outputs" / "projections" / "umap_z20_60"
    coords = pq.read_table(projection_dir / "coords.parquet").to_pydict()
    assert "usr_label__primary" in coords
    assert {"spyP", "sulAp"}.issubset(set(coords["usr_label__primary"]))

    plot_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "atlas_scatter",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )
    assert plot_result.exit_code == 0, plot_result.stdout

    plot_manifest = json.loads(
        (workspace_dir / "outputs" / "plots" / "atlas_scatter" / "manifest.json").read_text(encoding="utf-8")
    )
    assert plot_manifest["stats"]["reference_set_complete"] is True


def test_view_materialize_force_preserves_existing_artifact_on_failure(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    dataset = "promoter/demo_anchor_set"
    _write_usr_dataset(usr_root, dataset)
    _write_workspace_config(workspace_dir, usr_root)

    first_materialize = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert first_materialize.exit_code == 0, first_materialize.stdout

    view_dir = workspace_dir / "outputs" / "views" / "z20_60"
    assert (view_dir / "matrix.npy").is_file()
    assert (view_dir / "rows.parquet").is_file()
    assert (view_dir / "manifest.json").is_file()

    broken_rows = pa.Table.from_pylist(
        [
            {
                "id": f"row_{index:02d}",
                "subject_id": f"subject_{index:02d}",
                "usr_label__primary": "spyP" if index % 2 == 0 else "sulAp",
                "densegen__plan": "plan_a" if index % 3 == 0 else "plan_b",
            }
            for index in range(18)
        ]
    )
    pq.write_table(broken_rows, usr_root / dataset / "records.parquet")

    failed_force = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--force", "--json"],
    )
    assert failed_force.exit_code != 0
    assert "vector column is missing" in failed_force.stdout
    assert (view_dir / "matrix.npy").is_file()
    assert (view_dir / "rows.parquet").is_file()
    assert (view_dir / "manifest.json").is_file()


def test_projection_fit_force_preserves_existing_artifact_on_failure(tmp_path: Path) -> None:
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

    first_projection = _RUNNER.invoke(
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
    )
    assert first_projection.exit_code == 0, first_projection.stdout

    projection_dir = workspace_dir / "outputs" / "projections" / "umap_z20_60"
    assert (projection_dir / "coords.parquet").is_file()
    assert (projection_dir / "manifest.json").is_file()

    sample_rows_path = workspace_dir / "outputs" / "samples" / "atlas_sample" / "rows.parquet"
    pq.write_table(pq.read_table(sample_rows_path).slice(0, 2), sample_rows_path)

    failed_force = _RUNNER.invoke(
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
            "--force",
            "--json",
        ],
    )
    assert failed_force.exit_code != 0
    assert "at least 3 sampled rows" in failed_force.stdout
    assert (projection_dir / "coords.parquet").is_file()
    assert (projection_dir / "manifest.json").is_file()


def test_plot_render_rejects_non_identifier_plot_id(tmp_path: Path) -> None:
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


def test_projection_fit_rejects_parallel_lock_contention(tmp_path: Path) -> None:
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

    lock_path = operation_lock_path(workspace_dir / "outputs", operation="projection_fit")
    ready_path = tmp_path / "projection-fit-lock.ready"
    stop_path = tmp_path / "projection-fit-lock.stop"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _LOCK_HOLDER_SCRIPT,
            lock_path.as_posix(),
            ready_path.as_posix(),
            stop_path.as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_file(ready_path)
        result = _RUNNER.invoke(
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
        )
        assert result.exit_code == 21
        assert "another projection fit is already in progress for this workspace" in result.stdout
        assert "serializes heavy projection fits to avoid aggregate memory pressure" in result.stdout
    finally:
        stop_path.write_text("stop", encoding="utf-8")
        stdout, stderr = holder.communicate(timeout=5)
        assert holder.returncode == 0, stdout + stderr


def test_recipe_run_rejects_parallel_projection_lock_contention(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set")
    _write_workspace_config(workspace_dir, usr_root)
    _append_projection_recipe(workspace_dir, recipe_id="projection_recipe")

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

    lock_path = operation_lock_path(workspace_dir / "outputs", operation="projection_fit")
    ready_path = tmp_path / "projection-recipe-lock.ready"
    stop_path = tmp_path / "projection-recipe-lock.stop"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _LOCK_HOLDER_SCRIPT,
            lock_path.as_posix(),
            ready_path.as_posix(),
            stop_path.as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_file(ready_path)
        result = _RUNNER.invoke(
            app,
            [
                "recipe",
                "run",
                "projection_recipe",
                "--workspace",
                workspace_dir.as_posix(),
                "--json",
            ],
        )
        assert result.exit_code == 21
        assert "another projection fit is already in progress for this workspace" in result.stdout
    finally:
        stop_path.write_text("stop", encoding="utf-8")
        stdout, stderr = holder.communicate(timeout=5)
        assert holder.returncode == 0, stdout + stderr
