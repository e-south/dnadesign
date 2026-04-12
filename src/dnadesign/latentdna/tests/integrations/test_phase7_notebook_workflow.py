"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase7_notebook_workflow.py

Phase 7 workflow tests for notebook scaffolds over persisted artifacts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.cli import app

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
                "workspace": {"id": "stress_ethanol_cipro_latent_atlas", "output_root": "./outputs/latentdna"},
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
                "notebooks": {
                    "atlas_review": {
                        "kind": "artifact_review",
                        "title": "Atlas artifact review",
                        "description": "Load persisted atlas artifacts without recomputing them.",
                        "artifacts": [
                            {"kind": "view", "id": "z20_60"},
                            {"kind": "sample_set", "id": "atlas_sample"},
                            {"kind": "projection", "id": "umap_z20_60"},
                        ],
                    }
                },
                "recipes": {
                    "atlas_notebook_recipe": {
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
                                "params": {
                                    "plot_id": "atlas_demo_plot",
                                    "kind": "projection_scatter",
                                    "projection": ["umap_z20_60"],
                                },
                            },
                            {
                                "id": "generate_notebook",
                                "op": "notebook.generate",
                                "depends_on": ["render_plot"],
                                "params": {
                                    "notebook": "atlas_review",
                                },
                            },
                        ]
                    }
                },
                "deliverables": {
                    "atlas_review_bundle": {
                        "kind": "notebook_bundle",
                        "description": "Projection plot plus persisted artifact notebook scaffold.",
                        "recipe": "atlas_notebook_recipe",
                        "requires": {"views": ["z20_60"]},
                        "outputs": {
                            "samples": ["atlas_sample"],
                            "projections": ["umap_z20_60"],
                            "plots": ["atlas_demo_plot"],
                            "notebooks": ["atlas_review"],
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_phase7_notebook_generation_flow(tmp_path: Path) -> None:
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

    missing_notebook_result = _RUNNER.invoke(
        app,
        ["notebook", "generate", "atlas_review", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert missing_notebook_result.exit_code != 0
    assert "artifact is missing for notebook generation" in missing_notebook_result.stdout

    status_before = _RUNNER.invoke(
        app,
        ["deliverable", "status", "atlas_review_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert status_before.exit_code == 0, status_before.stdout
    status_before_payload = json.loads(status_before.stdout)
    assert status_before_payload["status"] == "missing"

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "atlas_review_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout
    run_payload = json.loads(run_result.stdout)
    assert run_payload["artifact_kind"] == "deliverable"
    assert run_payload["artifact_id"] == "atlas_review_bundle"
    assert run_payload["metrics"]["executed_steps"] == 5
    assert run_payload["metrics"]["skipped_steps"] == 0

    notebook_dir = workspace_dir / "outputs" / "latentdna" / "notebooks" / "atlas_review"
    notebook_path = notebook_dir / "notebook.py"
    assert notebook_path.is_file()
    notebook_text = notebook_path.read_text(encoding="utf-8")
    assert "import marimo" in notebook_text
    assert "__generated_with" in notebook_text
    assert 'app = marimo.App(width="full")' in notebook_text
    assert "Atlas artifact review" in notebook_text
    assert '"id": "z20_60"' in notebook_text
    assert "def load_artifact" in notebook_text
    assert "def discover_plot_artifacts" in notebook_text
    assert "def render_file" in notebook_text
    assert 'label="Artifact"' in notebook_text
    assert 'label="Workspace plot"' in notebook_text
    assert '"Workspace plots": workspace_plot_browser_panel' in notebook_text
    assert 'PLOT_ARTIFACT_ROOT = WORKSPACE_DIR / "outputs" / "latentdna" / "plots"' in notebook_text
    assert (notebook_dir / "manifest.json").is_file()

    rerun_result = _RUNNER.invoke(
        app,
        ["recipe", "run", "atlas_notebook_recipe", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert rerun_result.exit_code == 0, rerun_result.stdout
    rerun_payload = json.loads(rerun_result.stdout)
    assert rerun_payload["metrics"]["executed_steps"] == 0
    assert rerun_payload["metrics"]["skipped_steps"] == 5

    status_after = _RUNNER.invoke(
        app,
        ["deliverable", "status", "atlas_review_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert status_after.exit_code == 0, status_after.stdout
    status_after_payload = json.loads(status_after.stdout)
    assert status_after_payload["status"] == "ok"
    outputs = {entry["name"]: entry for entry in status_after_payload["outputs"]}
    assert outputs["notebook:atlas_review"]["status"] == "ok"
