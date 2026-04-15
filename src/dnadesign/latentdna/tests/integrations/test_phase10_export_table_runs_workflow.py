"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase10_export_table_runs_workflow.py

Phase 10 workflow tests for tabular exports plus artifact inventory operations.

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


def _write_workspace_config(workspace_dir: Path, usr_root: Path) -> None:
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
                    "anchor_ctx_20b": {
                        "left": "z20_1k_anchor",
                        "right": "z20_60",
                        "on": "subject_key",
                        "support": "intersection",
                    },
                },
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding_anchor_20b"},
                        "coordinate_space_id": "demo_space_20b",
                        "tags": {"model": "20b", "context": "anchor_only"},
                        "role": "primary",
                    },
                    "z20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "embedding_context_20b"},
                        "coordinate_space_id": "demo_space_20b",
                        "tags": {"model": "20b", "context": "template_1kb"},
                        "role": "primary",
                    },
                    "delta20": {
                        "derive": {
                            "kind": "vector_difference",
                            "left": "z20_1k_anchor",
                            "right": "z20_60",
                            "alignment": "anchor_ctx_20b",
                        },
                        "coordinate_space_id": "demo_space_20b",
                        "tags": {"operation": "difference"},
                        "role": "primary",
                    },
                },
                "scalars": {
                    "delta20_norm": {"derive": {"kind": "vector_norm", "view": "delta20", "norm": "l2"}},
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
                "exports": {
                    "x2_primary_20b_table": {
                        "row_basis": "anchor_ctx_20b",
                        "blocks": [
                            {
                                "kind": "reduced_view",
                                "block_id": "z20_60_pc",
                                "source": "z20_60_pc2",
                                "feature_prefix": "z20_60",
                                "alignment": "anchor_ctx_20b",
                            },
                            {
                                "kind": "reduced_view",
                                "block_id": "delta20_pc",
                                "source": "delta20_pc2",
                                "feature_prefix": "delta20",
                            },
                            {
                                "kind": "table_columns",
                                "block_id": "landmark_distances",
                                "source": "primary_landmark_distances",
                                "columns": ["d_spy_p", "d_sul_ap"],
                                "alignment": "anchor_ctx_20b",
                            },
                            {
                                "kind": "table_columns",
                                "block_id": "delta20_scalars",
                                "source": "delta20_norm",
                                "columns": ["delta20_norm"],
                            },
                        ],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _subject_ids(table: pa.Table) -> list[str]:
    return [str(value) for value in table["subject_id"].to_pylist()]


def _build_export_prerequisites(workspace_dir: Path) -> None:
    for view_id in ["z20_60", "z20_1k_anchor"]:
        result = _RUNNER.invoke(
            app,
            ["view", "materialize", view_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    for command in [
        ["alignment", "build", "anchor_ctx_20b"],
        ["view", "derive", "delta20"],
        ["scalar", "derive", "delta20_norm"],
        [
            "distance",
            "score",
            "primary_landmark_distances",
            "--view",
            "z20_60",
            "--landmark",
            "spy_p",
            "--landmark",
            "sul_ap",
        ],
        ["view", "reduce", "z20_60", "--run-id", "z20_60_pca", "--dims", "2", "--reduced-view-id", "z20_60_pc2"],
        ["view", "reduce", "delta20", "--run-id", "delta20_pca", "--dims", "2", "--reduced-view-id", "delta20_pc2"],
    ]:
        result = _RUNNER.invoke(app, [*command, "--workspace", workspace_dir.as_posix(), "--json"])
        assert result.exit_code == 0, result.stdout


def test_phase10_tabular_export_preserves_feature_order_and_alignment(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(
        usr_root,
        "promoter/demo_anchor_set",
        [
            {
                "id": "anchor_04",
                "subject_id": "subject_04",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor_20b": [4.0, 0.0, 1.0],
            },
            {
                "id": "anchor_01",
                "subject_id": "subject_01",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor_20b": [1.0, 0.0, 0.0],
            },
            {
                "id": "anchor_03",
                "subject_id": "subject_03",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor_20b": [3.0, 1.0, 1.0],
            },
            {
                "id": "anchor_02",
                "subject_id": "subject_02",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor_20b": [2.0, 0.0, 1.0],
            },
        ],
    )
    _write_usr_dataset(
        usr_root,
        "promoter/demo_context_set",
        [
            {
                "id": "ctx_02",
                "subject_id": "subject_02",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context_20b": [3.0, 1.0, 1.0],
            },
            {
                "id": "ctx_04",
                "subject_id": "subject_04",
                "context_id": "c1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context_20b": [5.0, 2.0, 1.0],
            },
            {
                "id": "ctx_01",
                "subject_id": "subject_01",
                "context_id": "c1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context_20b": [2.0, 1.0, 0.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)
    _build_export_prerequisites(workspace_dir)

    result = _RUNNER.invoke(
        app,
        ["export", "table", "x2_primary_20b_table", "--workspace", workspace_dir.as_posix(), "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["artifact_kind"] == "export_bundle"
    export_dir = workspace_dir / "outputs" / "exports" / "x2_primary_20b_table"
    export_table = pq.read_table(export_dir / "table.parquet")
    feature_table = pq.read_table(export_dir / "features.parquet")
    rows_table = pq.read_table(export_dir / "rows.parquet")

    assert export_table.num_rows == 3
    assert rows_table.num_rows == 3
    assert _subject_ids(rows_table) == ["subject_01", "subject_02", "subject_04"]
    assert export_table.column_names == [
        "subject_id",
        "left_count",
        "right_count",
        "z20_60_pc_001",
        "z20_60_pc_002",
        "delta20_pc_001",
        "delta20_pc_002",
        "d_spy_p",
        "d_sul_ap",
        "delta20_norm",
    ]
    assert feature_table["feature_name"].to_pylist() == export_table.column_names[3:]


def test_phase10_runs_inventory_and_artifact_inspection_cover_new_export(tmp_path: Path) -> None:
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
                "embedding_anchor_20b": [1.0, 0.0, 0.0],
            },
            {
                "id": "anchor_02",
                "subject_id": "subject_02",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor_20b": [2.0, 0.0, 1.0],
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
                "embedding_context_20b": [2.0, 1.0, 0.0],
            },
            {
                "id": "ctx_02",
                "subject_id": "subject_02",
                "context_id": "c1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context_20b": [3.0, 1.0, 1.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, usr_root)
    _build_export_prerequisites(workspace_dir)
    export_result = _RUNNER.invoke(
        app,
        ["export", "table", "x2_primary_20b_table", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert export_result.exit_code == 0, export_result.stdout

    runs_result = _RUNNER.invoke(app, ["runs", "list", "--workspace", workspace_dir.as_posix(), "--json"])
    assert runs_result.exit_code == 0, runs_result.stdout
    runs_payload = json.loads(runs_result.stdout)
    assert any(
        run["artifact_kind"] == "export_bundle" and run["artifact_id"] == "x2_primary_20b_table"
        for run in runs_payload["runs"]
    )

    show_result = _RUNNER.invoke(
        app,
        ["runs", "show", "export_bundle", "x2_primary_20b_table", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert show_result.exit_code == 0, show_result.stdout
    show_payload = json.loads(show_result.stdout)
    assert show_payload["artifact"]["artifact_kind"] == "export_bundle"
    assert show_payload["artifact"]["artifact_id"] == "x2_primary_20b_table"

    inspect_result = _RUNNER.invoke(app, ["inspect", "artifacts", "--workspace", workspace_dir.as_posix(), "--json"])
    assert inspect_result.exit_code == 0, inspect_result.stdout
    inspect_payload = json.loads(inspect_result.stdout)
    assert any(
        item["artifact_kind"] == "alignment_set" and item["artifact_id"] == "anchor_ctx_20b"
        for item in inspect_payload["data"]["artifacts"]
    )
    assert any(
        item["artifact_kind"] == "export_bundle" and item["artifact_id"] == "x2_primary_20b_table"
        for item in inspect_payload["data"]["artifacts"]
    )

    views_result = _RUNNER.invoke(app, ["inspect", "views", "--workspace", workspace_dir.as_posix(), "--json"])
    assert views_result.exit_code == 0, views_result.stdout
    views_payload = json.loads(views_result.stdout)
    assert any(item["view_id"] == "delta20" and item["materialized"] is True for item in views_payload["data"]["views"])

    alignment_result = _RUNNER.invoke(
        app,
        ["inspect", "alignment", "anchor_ctx_20b", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert alignment_result.exit_code == 0, alignment_result.stdout
    alignment_payload = json.loads(alignment_result.stdout)
    assert alignment_payload["data"]["alignment"]["alignment_id"] == "anchor_ctx_20b"
    assert alignment_payload["data"]["artifact"]["stats"]["matched_rows"] == 2

    prune_result = _RUNNER.invoke(
        app,
        ["runs", "prune", "export_bundle", "x2_primary_20b_table", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert prune_result.exit_code == 0, prune_result.stdout
    assert not (workspace_dir / "outputs" / "exports" / "x2_primary_20b_table").exists()
