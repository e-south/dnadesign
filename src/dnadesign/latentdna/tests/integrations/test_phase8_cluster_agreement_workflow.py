"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase8_cluster_agreement_workflow.py

Phase 8 workflow tests for minimal clustering plus richer agreement summaries.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
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
                        "coordinate_space_id": "demo_space_anchor",
                        "tags": {"model": "demo", "context": "anchor_only"},
                        "role": "primary",
                    },
                    "z20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "embedding_context"},
                        "coordinate_space_id": "demo_space_context",
                        "tags": {"model": "demo", "context": "template_1kb"},
                        "role": "challenger",
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


def test_phase8_cluster_and_rich_agreement_flow(tmp_path: Path) -> None:
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
                "embedding_anchor": [100.0, 0.0],
            },
            {
                "id": "anchor_04",
                "subject_id": "subject_04",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [101.0, 0.0],
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
                "embedding_context": [0.0, 0.0],
            },
            {
                "id": "ctx_02",
                "subject_id": "subject_02",
                "context_id": "c1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context": [100.0, 0.0],
            },
            {
                "id": "ctx_03",
                "subject_id": "subject_03",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [0.1, 0.0],
            },
            {
                "id": "ctx_04",
                "subject_id": "subject_04",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [100.1, 0.0],
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

    for neighbor_id, view_id in [("anchor_knn", "z20_60"), ("context_knn", "z20_1k_anchor")]:
        result = _RUNNER.invoke(
            app,
            [
                "neighbors",
                "fit",
                neighbor_id,
                "--workspace",
                workspace_dir.as_posix(),
                "--view",
                view_id,
                "--alignment",
                "anchor_ctx",
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

    for cluster_id, view_id in [("anchor_clusters", "z20_60"), ("context_clusters", "z20_1k_anchor")]:
        result = _RUNNER.invoke(
            app,
            [
                "cluster",
                "fit",
                cluster_id,
                "--workspace",
                workspace_dir.as_posix(),
                "--view",
                view_id,
                "--alignment",
                "anchor_ctx",
                "--n-clusters",
                "2",
                "--seed",
                "17",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["artifact_kind"] == "cluster_set"

    agreement_result = _RUNNER.invoke(
        app,
        [
            "agreement",
            "compare",
            "anchor_vs_context_rich",
            "--workspace",
            workspace_dir.as_posix(),
            "--left-neighbors",
            "anchor_knn",
            "--right-neighbors",
            "context_knn",
            "--left-clusters",
            "anchor_clusters",
            "--right-clusters",
            "context_clusters",
            "--landmark",
            "spy_p",
            "--landmark",
            "sul_ap",
            "--json",
        ],
    )
    assert agreement_result.exit_code == 0, agreement_result.stdout
    agreement_payload = json.loads(agreement_result.stdout)
    assert agreement_payload["artifact_kind"] == "agreement_set"

    agreement_dir = workspace_dir / "outputs" / "latentdna" / "agreements" / "anchor_vs_context_rich"
    agreement_rows = pq.read_table(agreement_dir / "table.parquet").to_pylist()

    knn_rows = [row for row in agreement_rows if row["method"] == "knn_overlap"]
    assert len(knn_rows) == 4
    assert [row["shared_neighbor_count"] for row in knn_rows] == [0, 0, 0, 0]
    assert [row["overlap_fraction"] for row in knn_rows] == [0.0, 0.0, 0.0, 0.0]

    cluster_rows = [row for row in agreement_rows if row["method"] == "cluster_agreement"]
    assert {row["metric"] for row in cluster_rows} == {"adjusted_rand_index", "normalized_mutual_information"}
    cluster_metrics = {row["metric"]: row["value"] for row in cluster_rows}
    assert cluster_metrics["adjusted_rand_index"] == pytest.approx(-0.5)
    assert cluster_metrics["normalized_mutual_information"] == pytest.approx(0.0)

    landmark_rows = [row for row in agreement_rows if row["method"] == "landmark_neighbor_overlap"]
    assert {row["landmark_id"] for row in landmark_rows} == {"spy_p", "sul_ap"}
    assert [row["jaccard_overlap"] for row in landmark_rows] == [0.0, 0.0]

    summary = json.loads((agreement_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["methods"] == ["cluster_agreement", "knn_overlap", "landmark_neighbor_overlap"]
    assert summary["knn_overlap"]["mean_overlap_fraction"] == 0.0
    assert summary["cluster_agreement"]["adjusted_rand_index"] == pytest.approx(-0.5)
    assert summary["landmark_neighbor_overlap"]["mean_jaccard_overlap"] == 0.0
