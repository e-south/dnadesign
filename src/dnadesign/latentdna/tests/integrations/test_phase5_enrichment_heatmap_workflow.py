"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase5_enrichment_heatmap_workflow.py

Phase 5 workflow tests for neighborhood enrichment and heatmap rendering.

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
                "landmarks": {
                    "spy_p": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "spyP"},
                        "representation": {"mode": "rows"},
                    },
                    "sul_ap": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "sulAp"},
                        "representation": {"mode": "rows"},
                    },
                },
                "cohorts": {
                    "plan": {
                        "kind": "column",
                        "source": "anchor60",
                        "column": "densegen__plan",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_phase5_landmark_enrichment_and_heatmap_flow(tmp_path: Path) -> None:
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
                "embedding_anchor": [0.0, 0.2],
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
                "embedding_anchor": [10.0, 0.2],
            },
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)

    view_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert view_result.exit_code == 0, view_result.stdout

    neighbors_result = _RUNNER.invoke(
        app,
        [
            "neighbors",
            "fit",
            "anchor_knn",
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
    assert neighbors_result.exit_code == 0, neighbors_result.stdout

    enrichment_result = _RUNNER.invoke(
        app,
        [
            "enrich",
            "score",
            "landmark_plan_enrichment",
            "--workspace",
            workspace_dir.as_posix(),
            "--neighbors",
            "anchor_knn",
            "--cohort",
            "plan",
            "--landmark",
            "spy_p",
            "--landmark",
            "sul_ap",
            "--json",
        ],
    )
    assert enrichment_result.exit_code == 0, enrichment_result.stdout
    enrichment_payload = json.loads(enrichment_result.stdout)
    assert enrichment_payload["artifact_kind"] == "enrichment_set"

    enrichment_dir = workspace_dir / "outputs" / "latentdna" / "enrichments" / "landmark_plan_enrichment"
    enrichment_rows = {
        (row["landmark_id"], row["cohort_value"]): row
        for row in pq.read_table(enrichment_dir / "table.parquet").to_pylist()
    }
    assert enrichment_rows[("spy_p", "plan_a")]["neighbor_hits"] == 2
    assert enrichment_rows[("spy_p", "plan_a")]["enrichment_delta"] == 0.5
    assert enrichment_rows[("spy_p", "plan_b")]["neighbor_hits"] == 0
    assert enrichment_rows[("spy_p", "plan_b")]["enrichment_delta"] == -0.5
    assert enrichment_rows[("sul_ap", "plan_a")]["neighbor_hits"] == 0
    assert enrichment_rows[("sul_ap", "plan_a")]["enrichment_delta"] == -0.5
    assert enrichment_rows[("sul_ap", "plan_b")]["neighbor_hits"] == 2
    assert enrichment_rows[("sul_ap", "plan_b")]["enrichment_delta"] == 0.5

    summary = json.loads((enrichment_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 4
    assert summary["cohort_id"] == "plan"
    assert summary["landmarks"] == ["spy_p", "sul_ap"]

    plot_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "landmark_plan_heatmap",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "heatmap",
            "--enrichment",
            "landmark_plan_enrichment",
            "--value-column",
            "enrichment_delta",
            "--json",
        ],
    )
    assert plot_result.exit_code == 0, plot_result.stdout
    plot_payload = json.loads(plot_result.stdout)
    assert plot_payload["artifact_kind"] == "plot"

    plot_dir = workspace_dir / "outputs" / "latentdna" / "plots" / "landmark_plan_heatmap"
    assert (plot_dir / "plot.svg").is_file()
    assert (plot_dir / "plot.png").is_file()
