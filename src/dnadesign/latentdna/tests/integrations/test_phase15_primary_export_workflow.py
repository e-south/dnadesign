"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase15_primary_export_workflow.py

Phase 15 workflow tests for the primary aligned export lane.

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


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_workspace_config(workspace_dir: Path, *, anchor_path: Path, context_path: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "latentdna_primary_export_demo", "output_root": "./outputs/latentdna"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "parquet",
                        "path": anchor_path.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                    "ctx1k": {
                        "kind": "parquet",
                        "path": context_path.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "context_key": "context_id",
                    },
                },
                "metadata": {"include": ["usr_label__primary", "densegen__plan"]},
                "alignments": {
                    "anchor_ctx_20b": {
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
                        "coordinate_space_id": "demo_20b_anchor_space",
                        "tags": {"model": "20b", "context": "anchor_only"},
                        "role": "primary",
                    },
                    "z20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "embedding_context"},
                        "coordinate_space_id": "demo_20b_anchor_space",
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
                        "coordinate_space_id": "demo_20b_anchor_space",
                        "tags": {"operation": "difference"},
                        "role": "primary",
                    },
                },
                "scalars": {"delta20_norm": {"derive": {"kind": "vector_norm", "view": "delta20", "norm": "l2"}}},
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
                    "x2_primary_20b": {
                        "row_basis": "anchor_ctx_20b",
                        "blocks": [
                            {
                                "kind": "reduced_view",
                                "block_id": "z20_60_pc",
                                "source": "z20_60_pc02",
                                "feature_prefix": "z20_60",
                                "alignment": "anchor_ctx_20b",
                            },
                            {
                                "kind": "reduced_view",
                                "block_id": "delta20_pc",
                                "source": "delta20_pc02",
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
                    }
                },
                "recipes": {
                    "x2_primary_20b_recipe": {
                        "steps": [
                            {"id": "materialize_anchor", "op": "view.materialize", "params": {"view": "z20_60"}},
                            {
                                "id": "materialize_context",
                                "op": "view.materialize",
                                "params": {"view": "z20_1k_anchor"},
                            },
                            {
                                "id": "build_alignment",
                                "op": "alignment.build",
                                "depends_on": ["materialize_anchor", "materialize_context"],
                                "params": {"alignment": "anchor_ctx_20b"},
                            },
                            {
                                "id": "derive_delta",
                                "op": "view.derive",
                                "depends_on": ["build_alignment"],
                                "params": {"view": "delta20"},
                            },
                            {
                                "id": "derive_delta_norm",
                                "op": "scalar.derive",
                                "depends_on": ["derive_delta"],
                                "params": {"scalar": "delta20_norm"},
                            },
                            {
                                "id": "score_distances",
                                "op": "distance.score",
                                "depends_on": ["materialize_anchor"],
                                "params": {
                                    "distance": "primary_landmark_distances",
                                    "view": "z20_60",
                                    "landmark": ["spy_p", "sul_ap"],
                                },
                            },
                            {
                                "id": "reduce_anchor",
                                "op": "view.reduce",
                                "depends_on": ["score_distances"],
                                "params": {
                                    "view": "z20_60",
                                    "run_id": "z20_60_pca",
                                    "dims": 2,
                                    "reduced_view_id": "z20_60_pc02",
                                },
                            },
                            {
                                "id": "reduce_delta",
                                "op": "view.reduce",
                                "depends_on": ["derive_delta_norm"],
                                "params": {
                                    "view": "delta20",
                                    "run_id": "delta20_pca",
                                    "dims": 2,
                                    "reduced_view_id": "delta20_pc02",
                                },
                            },
                            {
                                "id": "export_matrix",
                                "op": "export.matrix",
                                "depends_on": ["reduce_anchor", "reduce_delta"],
                                "params": {"export": "x2_primary_20b"},
                            },
                        ]
                    }
                },
                "deliverables": {
                    "x2_primary_20b": {
                        "kind": "export_bundle",
                        "description": "Aligned primary 20B export bundle.",
                        "recipe": "x2_primary_20b_recipe",
                        "requires": {
                            "views": ["z20_60", "z20_1k_anchor"],
                            "alignments": ["anchor_ctx_20b"],
                            "landmarks": ["spy_p", "sul_ap"],
                            "exports": ["x2_primary_20b"],
                        },
                        "outputs": {
                            "views": ["delta20"],
                            "scalars": ["delta20_norm"],
                            "distances": ["primary_landmark_distances"],
                            "reducers": ["z20_60_pca", "delta20_pca"],
                            "reduced_views": ["z20_60_pc02", "delta20_pc02"],
                            "exports": ["x2_primary_20b"],
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_phase15_primary_export_deliverable_flow(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    anchor_path = tmp_path / "inputs" / "anchor60.parquet"
    context_path = tmp_path / "inputs" / "ctx1k.parquet"
    _write_parquet(
        anchor_path,
        [
            {
                "id": "anchor_01",
                "subject_id": "subject_01",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor": [0.0, 0.0, 0.0],
            },
            {
                "id": "anchor_02",
                "subject_id": "subject_02",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_anchor": [0.0, 1.0, 0.0],
            },
            {
                "id": "anchor_03",
                "subject_id": "subject_03",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [4.0, 0.0, 1.0],
            },
            {
                "id": "anchor_04",
                "subject_id": "subject_04",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_anchor": [4.0, 1.0, 1.0],
            },
        ],
    )
    _write_parquet(
        context_path,
        [
            {
                "id": "ctx_01",
                "subject_id": "subject_01",
                "context_id": "template_1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context": [1.0, 0.0, 1.0],
            },
            {
                "id": "ctx_02",
                "subject_id": "subject_02",
                "context_id": "template_1",
                "usr_label__primary": "spyP",
                "densegen__plan": "plan_a",
                "embedding_context": [1.0, 2.0, 0.0],
            },
            {
                "id": "ctx_03",
                "subject_id": "subject_03",
                "context_id": "template_1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [5.0, 1.0, 1.0],
            },
            {
                "id": "ctx_04",
                "subject_id": "subject_04",
                "context_id": "template_1",
                "usr_label__primary": "sulAp",
                "densegen__plan": "plan_b",
                "embedding_context": [6.0, 1.0, 2.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, anchor_path=anchor_path, context_path=context_path)

    status_before = _RUNNER.invoke(
        app,
        ["deliverable", "status", "x2_primary_20b", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert status_before.exit_code == 0, status_before.stdout
    assert json.loads(status_before.stdout)["status"] == "missing"

    run_result = _RUNNER.invoke(
        app,
        ["deliverable", "run", "x2_primary_20b", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert run_result.exit_code == 0, run_result.stdout
    run_payload = json.loads(run_result.stdout)
    assert run_payload["artifact_kind"] == "deliverable"
    assert run_payload["artifact_id"] == "x2_primary_20b"
    assert run_payload["metrics"]["executed_steps"] == 9
    assert run_payload["metrics"]["skipped_steps"] == 0

    export_dir = workspace_dir / "outputs" / "latentdna" / "exports" / "x2_primary_20b"
    assert (export_dir / "matrix.npy").is_file()
    assert (export_dir / "rows.parquet").is_file()
    assert (export_dir / "features.parquet").is_file()
    feature_names = pq.read_table(export_dir / "features.parquet").column("feature_name").to_pylist()
    assert feature_names == [
        "z20_60_pc_001",
        "z20_60_pc_002",
        "delta20_pc_001",
        "delta20_pc_002",
        "d_spy_p",
        "d_sul_ap",
        "delta20_norm",
    ]

    status_after = _RUNNER.invoke(
        app,
        ["deliverable", "status", "x2_primary_20b", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert status_after.exit_code == 0, status_after.stdout
    assert json.loads(status_after.stdout)["status"] == "ok"

    rerun_result = _RUNNER.invoke(
        app,
        ["recipe", "run", "x2_primary_20b_recipe", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert rerun_result.exit_code == 0, rerun_result.stdout
    rerun_payload = json.loads(rerun_result.stdout)
    assert rerun_payload["metrics"]["executed_steps"] == 0
    assert rerun_payload["metrics"]["skipped_steps"] == 9
