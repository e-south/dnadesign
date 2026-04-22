"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_notebook_generation_workflow.py

Workflow tests for notebook scaffolds over persisted artifacts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app
from dnadesign.latentdna.src.contracts.deliverable import DeliverableStatusResult

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
                        "tags": {"model": "20b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                        "role": "primary",
                    }
                },
                "notebooks": {
                    "atlas_review": {
                        "kind": "workspace",
                        "title": "Atlas workspace notebook",
                        "description": "Review persisted atlas artifacts without recomputing them.",
                        "default_deliverable": "atlas_review_bundle",
                        "default_surface": "plots",
                        "ordered_plots": ["atlas_demo_plot"],
                    }
                },
                "plots": {
                    "atlas_demo_plot": {
                        "kind": "projection_scatter",
                        "projection": "umap_z20_60",
                        "semantics_ref": "plot_semantics/atlas_demo_plot.yaml",
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
                                "params": {"plot_id": "atlas_demo_plot"},
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
                        "recipe": "atlas_notebook_recipe",
                        "title": "Atlas review notebook bundle",
                        "section": "atlas",
                        "question": "Can the persisted atlas artifacts be reviewed without recomputation?",
                        "summary": (
                            "Projection plot plus persisted artifact notebook scaffold for the atlas review workflow."
                        ),
                        "requires": {
                            "sources": ["anchor60"],
                            "views": ["z20_60"],
                            "recipes": ["atlas_notebook_recipe"],
                        },
                        "outputs": {
                            "views": ["z20_60"],
                            "samples": ["atlas_sample"],
                            "projections": ["umap_z20_60"],
                            "plots": ["atlas_demo_plot"],
                            "notebooks": ["atlas_review"],
                        },
                        "docs_refs": [],
                        "acceptance_checks": [
                            {"kind": "required_plot_kind", "value": "projection_scatter"},
                        ],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace_dir / "plot_semantics").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "plot_semantics" / "atlas_demo_plot.yaml").write_text(
        yaml.safe_dump(
            {
                "plot_id": "atlas_demo_plot",
                "question": "Can the atlas review notebook render the persisted projection plot?",
                "decision_role": "debug",
                "encoding": "Single persisted projection scatter for the atlas review fixture.",
                "scope": "All rows in the fixture atlas sample.",
                "guardrails": ["Fixture-only QC plot for notebook workflow coverage."],
                "caption": "QC projection used to validate notebook generation and smoke behavior.",
                "alt_text": "Projection scatter for the notebook workflow fixture.",
                "preprocessing_md": "Fixture semantics do not declare additional preprocessing.",
                "math_md": "Fixture semantics do not declare a mathematical definition.",
                "rationale_md": "Fixture semantics exist only to validate notebook generation and smoke behavior.",
                "limitations_md": "Fixture semantics are not a study-facing scientific contract.",
                "failure_modes_md": "Replace fixture semantics before using the plot outside tests.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_notebook_generation_flow(tmp_path: Path) -> None:
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
    assert missing_notebook_result.exit_code == 0, missing_notebook_result.stdout
    missing_notebook_payload = json.loads(missing_notebook_result.stdout)
    assert missing_notebook_payload["status"] == "attention"
    assert "atlas_demo_plot" in "".join(missing_notebook_payload["warnings"])

    smoke_result = _RUNNER.invoke(
        app,
        ["notebook", "smoke", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert smoke_result.exit_code == 18, smoke_result.stdout
    smoke_payload = json.loads(smoke_result.stdout)
    assert smoke_payload["status"] == "error"
    assert smoke_payload["checks"]["notebook_exists"] is True
    assert smoke_payload["checks"]["control_plane_loads"] is True
    assert smoke_payload["checks"]["default_deliverable_ready"] is False

    inspect_health_result = _RUNNER.invoke(
        app,
        ["inspect", "notebook-health", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert inspect_health_result.exit_code == 18, inspect_health_result.stdout
    inspect_health_payload = json.loads(inspect_health_result.stdout)
    assert inspect_health_payload["data"]["health"]["status"] == "error"
    assert inspect_health_payload["data"]["health"]["checks"]["notebook_exists"] is True

    status_before = _RUNNER.invoke(
        app,
        ["deliverable", "status", "atlas_review_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert status_before.exit_code == 0, status_before.stdout
    status_before_payload = json.loads(status_before.stdout)
    assert status_before_payload["status"] == "attention"

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "atlas_review_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout
    run_payload = json.loads(run_result.stdout)
    assert run_payload["artifact_kind"] == "deliverable"
    assert run_payload["artifact_id"] == "atlas_review_bundle"
    assert run_payload["run_id"].startswith("deliverable__atlas_review_bundle__")
    assert run_payload["metrics"]["executed_steps"] == 5
    assert run_payload["metrics"]["skipped_steps"] == 0

    recipe_run_path = next(
        path
        for path in (workspace_dir / "outputs" / "runs").iterdir()
        if path.name.startswith("recipe__atlas_notebook_recipe__")
    )

    deliverable_run_json = workspace_dir / "outputs" / "runs" / run_payload["run_id"] / "run.json"
    recipe_run_json = recipe_run_path / "run.json"
    catalog_json = workspace_dir / "outputs" / "catalog.json"
    assert deliverable_run_json.is_file()
    assert recipe_run_json.is_file()
    assert catalog_json.is_file()
    assert json.loads(deliverable_run_json.read_text(encoding="utf-8"))["state"] == "succeeded"
    assert json.loads(recipe_run_json.read_text(encoding="utf-8"))["state"] == "succeeded"
    catalog_payload = json.loads(catalog_json.read_text(encoding="utf-8"))
    assert catalog_payload["workspace_id"] == "stress_ethanol_cipro_latent_atlas"
    assert any(row["deliverable_id"] == "atlas_review_bundle" for row in catalog_payload["deliverables"])
    sample_manifest = json.loads(
        (workspace_dir / "outputs" / "samples" / "atlas_sample" / "manifest.json").read_text(encoding="utf-8")
    )
    projection_manifest = json.loads(
        (workspace_dir / "outputs" / "projections" / "umap_z20_60" / "manifest.json").read_text(encoding="utf-8")
    )
    plot_manifest = json.loads(
        (workspace_dir / "outputs" / "plots" / "atlas_demo_plot" / "manifest.json").read_text(encoding="utf-8")
    )
    assert sample_manifest["inputs"][0]["path"].endswith("/views/z20_60/manifest.json")
    assert {entry["path"] for entry in projection_manifest["inputs"]} == {
        (workspace_dir / "outputs" / "views" / "z20_60" / "manifest.json").as_posix(),
        (workspace_dir / "outputs" / "samples" / "atlas_sample" / "manifest.json").as_posix(),
    }
    assert plot_manifest["inputs"][0]["path"].endswith("/projections/umap_z20_60/manifest.json")

    notebook_dir = workspace_dir / "outputs" / "notebooks" / "atlas_review"
    notebook_path = notebook_dir / "notebook.py"
    controls_path = notebook_dir / "controls.json"
    assert notebook_path.is_file()
    assert controls_path.is_file()
    notebook_manifest = json.loads((notebook_dir / "manifest.json").read_text(encoding="utf-8"))
    assert notebook_manifest["inputs"][0]["path"].endswith("/plots/atlas_demo_plot/manifest.json")
    assert any(
        entry["role"] == "workspace_config" and entry["path"].endswith("/config.yaml")
        for entry in notebook_manifest["source_provenance"]
    )
    notebook_text = notebook_path.read_text(encoding="utf-8")
    assert "import marimo" in notebook_text
    assert "__generated_with" in notebook_text
    assert 'app = marimo.App(width="full")' in notebook_text
    assert "Atlas workspace notebook" in notebook_text
    assert "load_workspace_notebook_controls(CONTROL_PATH)" in notebook_text
    assert "build_workspace_browser_runtime(" in notebook_text
    assert "runtime.support.notebook_theme()" in notebook_text
    assert "_plot_review = runtime.plot_review" in notebook_text
    assert "render_projection_grid(" in notebook_text
    assert "comparison_scope_note" in notebook_text
    assert 'runtime["' not in notebook_text
    assert "_controls = load_workspace_notebook_controls(CONTROL_PATH)" in notebook_text
    assert '_runtime_paths = _controls["runtime_paths"]' in notebook_text
    assert '_runtime_paths["workspace_relative_path"]' in notebook_text
    assert '_runtime_paths["output_relative_path"]' in notebook_text
    assert '_runtime_paths["catalog_relative_path"]' in notebook_text
    assert '_runtime_paths["health_relative_path"]' in notebook_text
    assert "parents[3]" not in notebook_text
    assert 'label="Section"' not in notebook_text
    assert 'label="Deliverable"' not in notebook_text
    assert 'label="Plot"' in notebook_text
    assert 'label="Model"' in notebook_text
    assert 'label="Family"' in notebook_text
    assert 'label="Context"' in notebook_text
    assert 'label="Layout"' in notebook_text
    assert 'label="Geometry"' in notebook_text
    assert 'label="Hue"' in notebook_text
    assert notebook_text.count("searchable=True") == 3
    assert "on_change=set_requested_hue" in notebook_text
    assert 'label="Left geometry"' in notebook_text
    assert 'label="Right geometry"' in notebook_text
    assert "Plots" in notebook_text
    assert "Geometry audit" in notebook_text
    assert "Comparison audit" in notebook_text
    assert "mo.state(default_tab)" in notebook_text
    assert "value=active_top_tab" in notebook_text
    assert "on_change=set_active_top_tab" in notebook_text
    assert "lazy=True" in notebook_text
    assert "Review the current artifact set for representation health" in notebook_text
    assert "Point positions are fixed by the saved coordinates" in notebook_text
    assert "Jump list:" not in notebook_text
    assert "mo.accordion(" in notebook_text
    assert '"Caption", "caption_md"' in notebook_text
    assert '_section_blocks.append(_support.mo.md(str(_active_card["caption_md"])))' not in notebook_text
    assert "(compatible_geometries or _geometry.geometry_rows)" not in notebook_text
    assert "No compatible geometry for this selection" in notebook_text
    assert "Deliverable: **" not in notebook_text
    assert "_badge =" not in notebook_text
    assert "Overview" not in notebook_text
    assert "Catalog" not in notebook_text
    assert (notebook_dir / "manifest.json").is_file()
    controls_payload = json.loads(controls_path.read_text(encoding="utf-8"))
    assert controls_payload["schema_version"] == "latentdna.workspace_notebook_controls.v4"
    assert controls_payload["workspace_id"] == "stress_ethanol_cipro_latent_atlas"
    assert controls_payload["notebook_id"] == "atlas_review"
    assert controls_payload["runtime_paths"]["workspace_relative_path"] == "../../.."
    assert controls_payload["runtime_paths"]["output_relative_path"] == "../.."
    assert controls_payload["runtime_paths"]["catalog_relative_path"] == "../../catalog.json"
    assert controls_payload["runtime_paths"]["health_relative_path"] == "health.json"
    assert controls_payload["plot_controls"]["default_surface"] == "plots"
    assert controls_payload["plot_controls"]["ordered_plot_ids"] == ["atlas_demo_plot"]

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
    health_after = _RUNNER.invoke(
        app,
        ["inspect", "notebook-health", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert health_after.exit_code == 0, health_after.stdout
    health_after_payload = json.loads(health_after.stdout)
    assert health_after_payload["data"]["health"]["status"] == "ok"

    export_path = workspace_dir / "atlas_review.html"
    export_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            notebook_path.as_posix(),
            "-o",
            export_path.as_posix(),
            "-f",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert export_result.returncode == 0, export_result.stderr or export_result.stdout
    assert export_path.is_file()

    controls_path.write_text('{"schema_version":"latentdna.workspace_notebook_controls.v1"}', encoding="utf-8")
    invalid_export_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            notebook_path.as_posix(),
            "-o",
            export_path.as_posix(),
            "-f",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid_export_result.returncode != 0
    assert "WorkspaceNotebookControls" in (invalid_export_result.stderr + invalid_export_result.stdout)

    invalid_validate_result = _RUNNER.invoke(
        app,
        ["validate", "workspace", "--workspace", workspace_dir.as_posix(), "--deep", "--json"],
    )
    assert invalid_validate_result.exit_code != 0
    assert "workspace notebook controls are invalid for atlas_review" in invalid_validate_result.stdout


def test_notebook_smoke_uses_live_default_deliverable_status(tmp_path: Path, monkeypatch) -> None:
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
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [1.0, 1.0],
            },
            {
                "id": "anchor_03",
                "subject_id": "subject_03",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor": [0.0, 1.0],
            },
            {
                "id": "anchor_04",
                "subject_id": "subject_04",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [1.0, 0.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "atlas_review_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout

    def _attention_default_deliverable(*_args, **_kwargs):
        return DeliverableStatusResult(
            deliverable_id="atlas_review_bundle",
            title="Atlas review notebook bundle",
            section="atlas",
            question="Can the persisted atlas artifacts be reviewed without recomputation?",
            summary="Projection plot plus persisted artifact notebook scaffold for the atlas review workflow.",
            status="attention",
            checks=[],
            outputs=[],
            docs_refs=[],
            acceptance_checks=[],
            warnings=["plot freshness requires attention"],
        )

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.notebook_service._default_deliverable_status",
        _attention_default_deliverable,
    )

    smoke_result = _RUNNER.invoke(
        app,
        ["notebook", "smoke", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert smoke_result.exit_code != 0, smoke_result.stdout
    smoke_payload = json.loads(smoke_result.stdout)
    assert smoke_payload["status"] == "error"
    assert smoke_payload["checks"]["default_deliverable_ready"] is False
    assert "plot freshness requires attention" in "".join(smoke_payload["warnings"])


def test_notebook_smoke_errors_when_ordered_plot_live_inputs_fail(tmp_path: Path, monkeypatch) -> None:
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

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "atlas_review_bundle", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout

    def _broken_plot_review_frames(plot_spec, *, joinable_tables, output_root):
        del plot_spec, joinable_tables, output_root
        frame = pd.DataFrame()
        frame.attrs["load_error"] = "projection artifact is not fresh for `umap_z20_60`: status=attention"
        return [frame]

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.notebook_service.load_plot_review_frames",
        _broken_plot_review_frames,
    )

    smoke_result = _RUNNER.invoke(
        app,
        ["notebook", "smoke", "--workspace", workspace_dir.as_posix(), "--json"],
    )

    assert smoke_result.exit_code == 18, smoke_result.stdout
    smoke_payload = json.loads(smoke_result.stdout)
    assert smoke_payload["status"] == "error"
    assert smoke_payload["checks"]["ordered_plot_live_inputs_ready"] is False
    assert "ordered plot `atlas_demo_plot` live inputs failed" in "".join(smoke_payload["warnings"])


def test_notebook_generate_records_smoke_failures_in_manifest_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
            }
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)

    def _broken_smoke(_workspace, *, notebook_id=None):
        assert notebook_id == "atlas_review"
        return {
            "workspace_id": "stress_ethanol_cipro_latent_atlas",
            "notebook_id": "atlas_review",
            "status": "error",
            "checks": {
                "notebook_exists": True,
                "control_plane_loads": True,
                "imports_resolve": False,
                "plot_catalog_loads": True,
                "default_deliverable_ready": True,
                "static_links_resolve": True,
                "ordered_plot_live_inputs_ready": True,
            },
            "warnings": ["imports_resolve failed: broken import"],
        }

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.notebook_service.smoke_workspace_notebook",
        _broken_smoke,
    )

    generate_result = _RUNNER.invoke(
        app,
        ["notebook", "generate", "atlas_review", "--workspace", workspace_dir.as_posix(), "--json"],
    )

    assert generate_result.exit_code == 0, generate_result.stdout
    payload = json.loads(generate_result.stdout)
    assert payload["status"] == "error"
    assert "imports_resolve failed: broken import" in "".join(payload["warnings"])

    manifest = json.loads(
        (workspace_dir / "outputs" / "notebooks" / "atlas_review" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "error"
    assert "imports_resolve failed: broken import" in "".join(manifest["warnings"])

    health = json.loads(
        (workspace_dir / "outputs" / "notebooks" / "atlas_review" / "health.json").read_text(encoding="utf-8")
    )
    assert health["status"] == "error"
    assert health["checks"]["imports_resolve"] is False
    assert "imports_resolve failed: broken import" in "".join(health["warnings"])


def test_notebook_generate_overwrites_stale_health_when_smoke_refresh_raises(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
            }
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)
    health_path = workspace_dir / "outputs" / "notebooks" / "atlas_review" / "health.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(
        json.dumps(
            {
                "workspace_id": "stress_ethanol_cipro_latent_atlas",
                "notebook_id": "atlas_review",
                "status": "ok",
                "checks": {
                    "notebook_exists": True,
                    "control_plane_loads": True,
                    "imports_resolve": True,
                    "plot_catalog_loads": True,
                    "default_deliverable_ready": True,
                    "static_links_resolve": True,
                    "ordered_plot_live_inputs_ready": True,
                },
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    def _raising_smoke(_workspace, *, notebook_id=None):
        assert notebook_id == "atlas_review"
        raise RuntimeError("simulated smoke refresh failure")

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.notebook_service.smoke_workspace_notebook",
        _raising_smoke,
    )

    generate_result = _RUNNER.invoke(
        app,
        ["notebook", "generate", "atlas_review", "--workspace", workspace_dir.as_posix(), "--json"],
    )

    assert generate_result.exit_code == 0, generate_result.stdout
    payload = json.loads(generate_result.stdout)
    assert payload["status"] == "error"
    assert "notebook health refresh failed: simulated smoke refresh failure" in "".join(payload["warnings"])

    health = json.loads(health_path.read_text(encoding="utf-8"))
    assert health["status"] == "error"
    assert health["checks"]["imports_resolve"] is False
    assert health["checks"]["static_links_resolve"] is False
    assert "notebook health refresh failed: simulated smoke refresh failure" in "".join(health["warnings"])


def test_load_catalog_payload_reuses_loaded_context_when_catalog_missing(monkeypatch, tmp_path: Path) -> None:
    from dnadesign.latentdna.src.services.notebook_service import _load_catalog_payload

    workspace_dir = tmp_path / "workspace"
    outputs_dir = workspace_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    fake_context = type(
        "FakeContext",
        (),
        {
            "workspace_dir": workspace_dir,
            "output_root": outputs_dir,
        },
    )()

    def _unexpected_reload(_workspace):
        raise AssertionError("workspace reload not expected")

    def _catalog_from_context(context):
        assert context is fake_context
        return {"workspace_id": "catalog_demo", "plots": []}

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.load_workspace_config",
        _unexpected_reload,
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.workspace_catalog_from_context",
        _catalog_from_context,
    )

    payload = _load_catalog_payload(fake_context)

    assert payload["workspace_id"] == "catalog_demo"
