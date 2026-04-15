"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase4_neighbors_agreement_workflow.py

Phase 4 workflow tests for neighbor artifacts and cross-view agreement.

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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_phase4_neighbors_and_agreement_flow(tmp_path: Path) -> None:
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
                "embedding_context": [10.0, 0.0],
            },
            {
                "id": "ctx_03",
                "subject_id": "subject_03",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [0.0, 1.0],
            },
            {
                "id": "ctx_04",
                "subject_id": "subject_04",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [10.0, 1.0],
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

    anchor_neighbors_result = _RUNNER.invoke(
        app,
        [
            "neighbors",
            "fit",
            "anchor_knn",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z20_60",
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
    assert anchor_neighbors_result.exit_code == 0, anchor_neighbors_result.stdout
    anchor_neighbors_payload = json.loads(anchor_neighbors_result.stdout)
    assert anchor_neighbors_payload["artifact_kind"] == "neighbor_set"

    context_neighbors_result = _RUNNER.invoke(
        app,
        [
            "neighbors",
            "fit",
            "context_knn",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z20_1k_anchor",
            "--alignment",
            "anchor_ctx",
            "--k",
            "1",
            "--backend",
            "approximate",
            "--metric",
            "euclidean",
            "--json",
        ],
    )
    assert context_neighbors_result.exit_code == 0, context_neighbors_result.stdout

    anchor_indices = np.load(workspace_dir / "outputs" / "neighbors" / "anchor_knn" / "indices.npy")
    context_indices = np.load(workspace_dir / "outputs" / "neighbors" / "context_knn" / "indices.npy")
    assert anchor_indices.shape == (4, 1)
    assert context_indices.shape == (4, 1)

    agreement_result = _RUNNER.invoke(
        app,
        [
            "agreement",
            "compare",
            "anchor_vs_context_knn",
            "--workspace",
            workspace_dir.as_posix(),
            "--left-neighbors",
            "anchor_knn",
            "--right-neighbors",
            "context_knn",
            "--json",
        ],
    )
    assert agreement_result.exit_code == 0, agreement_result.stdout
    agreement_payload = json.loads(agreement_result.stdout)
    assert agreement_payload["artifact_kind"] == "agreement_set"

    agreement_dir = workspace_dir / "outputs" / "agreements" / "anchor_vs_context_knn"
    agreement_table = pq.read_table(agreement_dir / "table.parquet")
    assert agreement_table.column("shared_neighbor_count").to_pylist() == [0, 0, 0, 0]
    assert agreement_table.column("overlap_fraction").to_pylist() == [0.0, 0.0, 0.0, 0.0]

    summary = json.loads((agreement_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mean_overlap_fraction"] == 0.0
    assert summary["rows"] == 4
