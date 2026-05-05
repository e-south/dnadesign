"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_plot_diagnostics_workflow.py

Workflow tests for the remaining read-only plot kinds.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app

_RUNNER = CliRunner()


def _write_plot_semantics(workspace_dir: Path, plot_id: str) -> str:
    semantics_ref = f"plot_semantics/{plot_id}.yaml"
    semantics_path = workspace_dir / semantics_ref
    semantics_path.parent.mkdir(parents=True, exist_ok=True)
    semantics_path.write_text(
        yaml.safe_dump(
            {
                "plot_id": plot_id,
                "question": f"What does {plot_id} show?",
                "decision_role": "debug",
                "encoding": f"QC fixture semantics for {plot_id}.",
                "scope": "Fixture-sized workflow sample.",
                "guardrails": ["Fixture semantics are descriptive only."],
                "caption": f"QC fixture plot for {plot_id}.",
                "alt_text": f"QC fixture plot for {plot_id}.",
                "preprocessing_md": "Fixture semantics do not declare additional preprocessing.",
                "math_md": "Fixture semantics do not declare a mathematical definition.",
                "rationale_md": "Fixture semantics exist only to exercise diagnostic rendering.",
                "limitations_md": "Fixture semantics are not a study-facing scientific contract.",
                "failure_modes_md": "Replace fixture semantics before using the plot outside tests.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return semantics_ref


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
                "plots": {
                    "primary_landmark_scatter": {
                        "kind": "distance_scatter",
                        "distance": "primary_landmark_distances",
                        "x_column": "d_spy_p",
                        "y_column": "d_sul_ap",
                        "color_column": "densegen__plan",
                        "semantics_ref": _write_plot_semantics(workspace_dir, "primary_landmark_scatter"),
                    },
                    "spy_distance_distribution": {
                        "kind": "distribution",
                        "distance": "primary_landmark_distances",
                        "value_column": "d_spy_p",
                        "color_column": "densegen__plan",
                        "semantics_ref": _write_plot_semantics(workspace_dir, "spy_distance_distribution"),
                    },
                    "primary_agreement_summary": {
                        "kind": "agreement_summary",
                        "agreement": "primary_vs_primary",
                        "semantics_ref": _write_plot_semantics(workspace_dir, "primary_agreement_summary"),
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_distance_distribution_and_agreement_summary_plots(tmp_path: Path) -> None:
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
                "embedding_anchor": [0.1, 0.2],
            },
            {
                "id": "anchor_03",
                "subject_id": "subject_03",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [4.0, 4.0],
            },
            {
                "id": "anchor_04",
                "subject_id": "subject_04",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [4.2, 4.1],
            },
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)

    view_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert view_result.exit_code == 0, view_result.stdout

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

    for neighbor_id in ["left_knn", "right_knn"]:
        result = _RUNNER.invoke(
            app,
            [
                "neighbors",
                "fit",
                neighbor_id,
                "--workspace",
                workspace_dir.as_posix(),
                "--view",
                "z20_60",
                "--k",
                "1",
                "--backend",
                "exact",
                "--metric",
                "euclidean",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.stdout

    agreement_result = _RUNNER.invoke(
        app,
        [
            "agreement",
            "compare",
            "primary_vs_primary",
            "--workspace",
            workspace_dir.as_posix(),
            "--left-neighbors",
            "left_knn",
            "--right-neighbors",
            "right_knn",
            "--json",
        ],
    )
    assert agreement_result.exit_code == 0, agreement_result.stdout

    distance_scatter_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "primary_landmark_scatter",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )
    assert distance_scatter_result.exit_code == 0, distance_scatter_result.stdout
    distance_scatter_payload = json.loads(distance_scatter_result.stdout)
    assert distance_scatter_payload["artifact_kind"] == "plot"
    distance_scatter_preview_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "primary_landmark_scatter",
            "--workspace",
            workspace_dir.as_posix(),
            "--dry-run",
            "--force",
            "--json",
        ],
    )
    assert distance_scatter_preview_result.exit_code == 0, distance_scatter_preview_result.stdout
    distance_scatter_preview_payload = json.loads(distance_scatter_preview_result.stdout)
    assert distance_scatter_preview_payload["dry_run"] is True
    assert distance_scatter_preview_payload["inputs"] == distance_scatter_payload["inputs"]

    distribution_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "spy_distance_distribution",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )
    assert distribution_result.exit_code == 0, distribution_result.stdout
    distribution_payload = json.loads(distribution_result.stdout)
    assert distribution_payload["artifact_kind"] == "plot"
    distribution_preview_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "spy_distance_distribution",
            "--workspace",
            workspace_dir.as_posix(),
            "--dry-run",
            "--force",
            "--json",
        ],
    )
    assert distribution_preview_result.exit_code == 0, distribution_preview_result.stdout
    distribution_preview_payload = json.loads(distribution_preview_result.stdout)
    assert distribution_preview_payload["dry_run"] is True
    assert distribution_preview_payload["inputs"] == distribution_payload["inputs"]

    agreement_plot_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "primary_agreement_summary",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )
    assert agreement_plot_result.exit_code == 0, agreement_plot_result.stdout
    agreement_plot_payload = json.loads(agreement_plot_result.stdout)
    assert agreement_plot_payload["artifact_kind"] == "plot"

    output_root = workspace_dir / "outputs" / "plots"
    for plot_id in ["primary_landmark_scatter", "spy_distance_distribution", "primary_agreement_summary"]:
        plot_dir = output_root / plot_id
        assert (plot_dir / "plot.svg").is_file()
        assert (plot_dir / "plot.png").is_file()
        manifest = json.loads((plot_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["artifact_kind"] == "plot"

    scatter_manifest = json.loads(
        (output_root / "primary_landmark_scatter" / "manifest.json").read_text(encoding="utf-8")
    )
    assert scatter_manifest["params"]["plot_kind"] == "distance_scatter"
    assert scatter_manifest["params"]["distance_id"] == "primary_landmark_distances"

    distribution_manifest = json.loads(
        (output_root / "spy_distance_distribution" / "manifest.json").read_text(encoding="utf-8")
    )
    assert distribution_manifest["params"]["plot_kind"] == "distribution"
    assert distribution_manifest["params"]["input_kind"] == "distance_set"

    agreement_manifest = json.loads(
        (output_root / "primary_agreement_summary" / "manifest.json").read_text(encoding="utf-8")
    )
    assert agreement_manifest["params"]["plot_kind"] == "agreement_summary"
    assert agreement_manifest["params"]["agreement_id"] == "primary_vs_primary"


def test_distance_scoring_fails_fast_when_landmark_selector_matches_no_rows(tmp_path: Path) -> None:
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
                "usr_label__primary": "control",
                "densegen__plan": "plan_a",
                "embedding_anchor": [0.0, 0.0],
            },
            {
                "id": "anchor_02",
                "subject_id": "subject_02",
                "usr_label__primary": "sample",
                "densegen__plan": "plan_b",
                "embedding_anchor": [1.0, 1.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)

    view_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert view_result.exit_code == 0, view_result.stdout

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
            "--json",
        ],
    )

    assert distance_result.exit_code != 0
    assert "landmark spy_p matched no rows" in distance_result.stdout


def test_plot_render_rejects_mixed_named_and_inline_specs(tmp_path: Path) -> None:
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

    result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "primary_landmark_scatter",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "distance_scatter",
            "--distance",
            "primary_landmark_distances",
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert "either a named workspace plot recipe or inline plot flags" in result.stdout
