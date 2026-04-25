"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_downstream_freshness_workflow.py

Workflow tests for downstream freshness/readiness reporting over
table-derived scalars and agreement-summary plots.

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
                "rationale_md": "Fixture semantics exist only to exercise freshness workflows.",
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
                "workspace": {"id": "latentdna_downstream_freshness_demo", "output_root": "./outputs"},
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
                "scalars": {
                    "ethanol_vs_cipro": {
                        "derive": {
                            "kind": "column_expression",
                            "source": "primary_landmark_distances",
                            "expression": "d_sul_ap - d_spy_p",
                            "output_column": "ethanol_vs_cipro",
                        }
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
                    "control_distance_margins_distribution": {
                        "kind": "distribution",
                        "scalar": "ethanol_vs_cipro",
                        "value_column": "ethanol_vs_cipro",
                        "color_column": "densegen__plan",
                        "semantics_ref": _write_plot_semantics(
                            workspace_dir,
                            "control_distance_margins_distribution",
                        ),
                    },
                    "primary_agreement_summary": {
                        "kind": "agreement_summary",
                        "agreement": "primary_vs_primary",
                        "semantics_ref": _write_plot_semantics(workspace_dir, "primary_agreement_summary"),
                    },
                },
                "notebooks": {
                    "agreement_review": {
                        "kind": "workspace",
                        "title": "Agreement review",
                        "description": "Read-only workspace notebook for downstream agreement diagnostics.",
                        "default_deliverable": "downstream_freshness_bundle",
                    }
                },
                "recipes": {
                    "downstream_freshness_recipe": {
                        "steps": [
                            {
                                "id": "materialize_anchor_view",
                                "op": "view.materialize",
                                "params": {"view": "z20_60"},
                            },
                            {
                                "id": "score_landmark_distances",
                                "op": "distance.score",
                                "depends_on": ["materialize_anchor_view"],
                                "params": {
                                    "distance": "primary_landmark_distances",
                                    "view": "z20_60",
                                    "landmark": ["spy_p", "sul_ap"],
                                    "metric": "euclidean",
                                },
                            },
                            {
                                "id": "derive_margin_scalar",
                                "op": "scalar.derive",
                                "depends_on": ["score_landmark_distances"],
                                "params": {"scalar": "ethanol_vs_cipro"},
                            },
                            {
                                "id": "build_left_neighbors",
                                "op": "neighbors.fit",
                                "depends_on": ["materialize_anchor_view"],
                                "params": {
                                    "neighbors": "left_knn",
                                    "view": "z20_60",
                                    "k": 1,
                                    "backend": "exact",
                                    "metric": "euclidean",
                                },
                            },
                            {
                                "id": "build_right_neighbors",
                                "op": "neighbors.fit",
                                "depends_on": ["materialize_anchor_view"],
                                "params": {
                                    "neighbors": "right_knn",
                                    "view": "z20_60",
                                    "k": 1,
                                    "backend": "exact",
                                    "metric": "euclidean",
                                },
                            },
                            {
                                "id": "compare_agreement",
                                "op": "agreement.compare",
                                "depends_on": ["build_left_neighbors", "build_right_neighbors"],
                                "params": {
                                    "agreement": "primary_vs_primary",
                                    "left_neighbors": "left_knn",
                                    "right_neighbors": "right_knn",
                                },
                            },
                            {
                                "id": "render_scalar_distribution",
                                "op": "plot.render",
                                "depends_on": ["derive_margin_scalar"],
                                "params": {"plot_id": "control_distance_margins_distribution"},
                            },
                            {
                                "id": "render_agreement_summary",
                                "op": "plot.render",
                                "depends_on": ["compare_agreement"],
                                "params": {"plot_id": "primary_agreement_summary"},
                            },
                            {
                                "id": "generate_agreement_review",
                                "op": "notebook.generate",
                                "depends_on": ["render_agreement_summary"],
                                "params": {
                                    "notebook": "agreement_review",
                                },
                            },
                        ]
                    }
                },
                "deliverables": {
                    "downstream_freshness_bundle": {
                        "recipe": "downstream_freshness_recipe",
                        "title": "Downstream freshness bundle",
                        "section": "freshness",
                        "question": (
                            "Do downstream scalar and agreement artifacts stay fresh when inputs are unchanged?"
                        ),
                        "summary": "Fresh downstream scalar and agreement diagnostics.",
                        "requires": {
                            "sources": ["anchor60"],
                            "views": ["z20_60"],
                            "recipes": ["downstream_freshness_recipe"],
                        },
                        "outputs": {
                            "views": ["z20_60"],
                            "distances": ["primary_landmark_distances"],
                            "scalars": ["ethanol_vs_cipro"],
                            "agreements": ["primary_vs_primary"],
                            "plots": [
                                "control_distance_margins_distribution",
                                "primary_agreement_summary",
                            ],
                            "notebooks": ["agreement_review"],
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


def test_downstream_freshness_surfaces_stay_ok_when_inputs_are_fresh(tmp_path: Path) -> None:
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

    run_result = _RUNNER.invoke(
        app,
        [
            "deliverable",
            "run",
            "downstream_freshness_bundle",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )
    assert run_result.exit_code == 0, run_result.stdout
    run_payload = json.loads(run_result.stdout)
    assert run_payload["status"] == "ok"
    assert not any(
        "default deliverable requires attention before the notebook is end-to-end ready" in warning
        for warning in run_payload["warnings"]
    )

    status_result = _RUNNER.invoke(
        app,
        [
            "deliverable",
            "status",
            "downstream_freshness_bundle",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )
    assert status_result.exit_code == 0, status_result.stdout
    status_payload = json.loads(status_result.stdout)
    assert status_payload["status"] == "ok"
    output_statuses = {entry["name"]: entry["status"] for entry in status_payload["outputs"]}
    assert output_statuses["scalar_table:ethanol_vs_cipro"] == "ok"
    assert output_statuses["plot:control_distance_margins_distribution"] == "ok"
    assert output_statuses["plot:primary_agreement_summary"] == "ok"
    assert output_statuses["notebook:agreement_review"] == "ok"

    runs_result = _RUNNER.invoke(
        app,
        ["runs", "list", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert runs_result.exit_code == 0, runs_result.stdout
    runs_payload = json.loads(runs_result.stdout)
    statuses = {
        (entry["artifact_kind"], entry["artifact_id"]): entry["status"]
        for entry in runs_payload["runs"]
        if (entry["artifact_kind"], entry["artifact_id"])
        in {
            ("notebook", "agreement_review"),
            ("scalar_table", "ethanol_vs_cipro"),
            ("plot", "control_distance_margins_distribution"),
            ("plot", "primary_agreement_summary"),
        }
    }
    assert statuses[("notebook", "agreement_review")] == "ok"
    assert statuses[("scalar_table", "ethanol_vs_cipro")] == "ok"
    assert statuses[("plot", "control_distance_margins_distribution")] == "ok"
    assert statuses[("plot", "primary_agreement_summary")] == "ok"
