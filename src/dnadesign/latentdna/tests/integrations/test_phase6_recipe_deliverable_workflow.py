"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase6_recipe_deliverable_workflow.py

Phase 6 workflow tests for recipe orchestration and deliverable status/run.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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
                    "metric": "euclidean",
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
                        "vector": {"kind": "column", "name": "embedding_anchor"},
                        "coordinate_space_id": "demo_space_anchor",
                        "tags": {"model": "demo", "context": "anchor_only"},
                        "role": "primary",
                    }
                },
                "plots": {
                    "atlas_demo_plot": {
                        "kind": "projection_scatter",
                        "projection": "umap_z20_60",
                    }
                },
                "recipes": {
                    "atlas_recipe": {
                        "steps": [
                            {
                                "id": "materialize_anchor",
                                "op": "view.materialize",
                                "params": {"view": "z20_60"},
                            },
                            {
                                "id": "sample_all",
                                "op": "sample.build",
                                "depends_on": ["materialize_anchor"],
                                "params": {
                                    "sample_id": "atlas_sample",
                                    "view": "z20_60",
                                    "strategy": "all",
                                    "seed": 17,
                                },
                            },
                            {
                                "id": "fit_projection",
                                "op": "projection.fit",
                                "depends_on": ["sample_all"],
                                "params": {
                                    "view": "z20_60",
                                    "sample": "atlas_sample",
                                    "run_id": "umap_z20_60",
                                    "metric": "euclidean",
                                    "seed": 17,
                                },
                            },
                            {
                                "id": "render_plot",
                                "op": "plot.render",
                                "depends_on": ["fit_projection"],
                                "params": {"plot_id": "atlas_demo_plot"},
                            },
                        ]
                    }
                },
                "deliverables": {
                    "atlas_demo": {
                        "title": "Demo projection deliverable",
                        "summary": "Demo projection deliverable.",
                        "question": "How does the demo projection materialize a view and notebook?",
                        "section": "Demo",
                        "recipe": "atlas_recipe",
                        "requires": {"views": ["z20_60"]},
                        "outputs": {
                            "samples": ["atlas_sample"],
                            "projections": ["umap_z20_60"],
                            "plots": ["atlas_demo_plot"],
                        },
                        "docs_refs": [],
                        "acceptance_checks": [],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_phase6_recipe_and_deliverable_flow(tmp_path: Path) -> None:
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
                "embedding_anchor": [0.0, 1.0],
            },
            {
                "id": "anchor_03",
                "subject_id": "subject_03",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [10.0, 0.0],
            },
            {
                "id": "anchor_04",
                "subject_id": "subject_04",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [10.0, 1.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)

    missing_plot_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "atlas_demo_plot",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )
    assert missing_plot_result.exit_code != 0
    assert "projection artifact is missing" in missing_plot_result.stdout

    validate_result = _RUNNER.invoke(
        app,
        ["recipe", "validate", "atlas_recipe", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert validate_result.exit_code == 0, validate_result.stdout
    validate_payload = json.loads(validate_result.stdout)
    assert validate_payload["status"] == "ok"
    assert validate_payload["step_order"] == [
        "materialize_anchor",
        "sample_all",
        "fit_projection",
        "render_plot",
    ]

    list_result = _RUNNER.invoke(app, ["deliverable", "list", "--workspace", workspace_dir.as_posix(), "--json"])
    assert list_result.exit_code == 0, list_result.stdout
    list_payload = json.loads(list_result.stdout)
    assert list_payload["deliverables"][0]["deliverable_id"] == "atlas_demo"

    status_before = _RUNNER.invoke(
        app,
        ["deliverable", "status", "atlas_demo", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert status_before.exit_code == 0, status_before.stdout
    status_before_payload = json.loads(status_before.stdout)
    assert status_before_payload["status"] == "missing"
    checks_before = {entry["name"]: entry for entry in status_before_payload["checks"]}
    assert checks_before["view:z20_60"]["status"] == "missing"

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "atlas_demo", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout
    run_payload = json.loads(run_result.stdout)
    assert run_payload["artifact_kind"] == "deliverable"
    assert run_payload["artifact_id"] == "atlas_demo"
    assert run_payload["metrics"]["executed_steps"] == 4
    assert run_payload["metrics"]["skipped_steps"] == 0

    plot_dir = workspace_dir / "outputs" / "plots" / "atlas_demo_plot"
    assert (plot_dir / "plot.svg").is_file()
    assert (plot_dir / "plot.png").is_file()

    rerun_result = _RUNNER.invoke(
        app,
        ["recipe", "run", "atlas_recipe", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert rerun_result.exit_code == 0, rerun_result.stdout
    rerun_payload = json.loads(rerun_result.stdout)
    assert rerun_payload["artifact_kind"] == "recipe"
    assert rerun_payload["artifact_id"] == "atlas_recipe"
    assert rerun_payload["metrics"]["executed_steps"] == 0
    assert rerun_payload["metrics"]["skipped_steps"] == 4

    status_after = _RUNNER.invoke(
        app,
        ["deliverable", "status", "atlas_demo", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert status_after.exit_code == 0, status_after.stdout
    status_after_payload = json.loads(status_after.stdout)
    assert status_after_payload["status"] == "ok"

    shutil.rmtree(plot_dir)
    partial_status = _RUNNER.invoke(
        app,
        ["deliverable", "status", "atlas_demo", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert partial_status.exit_code == 0, partial_status.stdout
    partial_payload = json.loads(partial_status.stdout)
    assert partial_payload["status"] == "attention"
    outputs = {entry["name"]: entry for entry in partial_payload["outputs"]}
    assert outputs["plot:atlas_demo_plot"]["status"] == "missing"
    assert outputs["projection:umap_z20_60"]["status"] == "ok"
