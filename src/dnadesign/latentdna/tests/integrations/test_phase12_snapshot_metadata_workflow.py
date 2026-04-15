"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase12_snapshot_metadata_workflow.py

Phase 12 workflow tests for snapshot metadata companions.

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


def _write_usr_dataset(root: Path, dataset: str, rows: list[dict[str, object]]) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dataset_dir / "records.parquet")


def test_phase12_snapshot_build_persists_metadata_companion_without_vectors(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
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
                "embedding_context_20b": [1.0, 0.0, 0.0],
            },
            {
                "id": "ctx_02",
                "subject_id": "subject_02",
                "context_id": "c2",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context_20b": [2.0, 0.0, 1.0],
            },
        ],
    )
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
                    "ctx1k": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_context_set",
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "context_key": "context_id",
                    },
                },
                "metadata": {"include": ["usr_label__primary", "densegen__plan"]},
                "views": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "snapshot",
            "build",
            "ctx1k_snapshot",
            "--source",
            "ctx1k",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["artifact_kind"] == "snapshot"
    snapshot_dir = workspace_dir / "outputs" / "snapshots" / "ctx1k_snapshot"
    metadata_table = pq.read_table(snapshot_dir / "metadata.parquet")
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))

    assert metadata_table.column_names == [
        "id",
        "subject_id",
        "context_id",
        "usr_label__primary",
        "densegen__plan",
    ]
    assert metadata_table.num_rows == 2
    assert "embedding_context_20b" not in metadata_table.column_names
    assert {item["path"] for item in manifest["outputs"]} == {"rows.parquet", "metadata.parquet"}
    assert manifest["params"]["row_columns"] == ["id", "subject_id", "context_id"]
    assert manifest["params"]["metadata_columns"] == [
        "id",
        "subject_id",
        "context_id",
        "usr_label__primary",
        "densegen__plan",
    ]
