"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_phase16_stats_cli_surface_workflow.py

Phase 16 workflow tests for view stats, scalar-table joins, and common CLI
preview/quiet behavior.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app

_RUNNER = CliRunner()


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_matrix_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    rows = pa.Table.from_pylist(
        [
            {"id": "bundle_01", "subject_id": "subject_01", "cohort": "a"},
            {"id": "bundle_02", "subject_id": "subject_02", "cohort": "a"},
            {"id": "bundle_03", "subject_id": "subject_03", "cohort": "b"},
        ]
    )
    pq.write_table(rows, bundle_dir / "rows.parquet")
    np.save(
        bundle_dir / "matrix.npy",
        np.asarray(
            [
                [3.0, 4.0, 0.0, 0.0],
                [0.0, 5.0, 12.0, 0.0],
                [8.0, 15.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "export_bundle",
                "artifact_id": "bundle_source",
                "workspace_id": "fixture_workspace",
                "command": "fixture",
                "status": "ok",
                "outputs": [
                    {"path": "matrix.npy", "media_type": "application/x-npy"},
                    {"path": "rows.parquet", "media_type": "application/x-parquet"},
                ],
                "stats": {"rows": 3, "dims": 4},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_workspace_config(workspace_dir: Path, bundle_dir: Path, context_path: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "latentdna_phase16_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "bundle_source": {
                        "kind": "matrix_bundle",
                        "path": bundle_dir.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                    "context_source": {
                        "kind": "parquet",
                        "path": context_path.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "context_key": "context_id",
                    },
                },
                "metadata": {"include": ["cohort", "label"]},
                "views": {
                    "bundle_view": {
                        "source": "bundle_source",
                        "vector": {"kind": "bundle_matrix"},
                        "coordinate_space_id": "bundle_space",
                        "tags": {"model": "bundle"},
                        "role": "primary",
                    },
                    "context_view": {
                        "source": "context_source",
                        "vector": {"kind": "column", "name": "embedding_context"},
                        "coordinate_space_id": "context_space",
                        "tags": {"model": "context"},
                        "role": "primary",
                    },
                    "context_by_subject": {
                        "derive": {
                            "kind": "aggregate_by_key",
                            "view": "context_view",
                            "key": "subject_key",
                            "aggregation": "mean",
                        },
                        "coordinate_space_id": "context_space",
                        "tags": {"operation": "aggregate"},
                        "role": "primary",
                    },
                },
                "scalars": {
                    "bundle_norm": {
                        "derive": {
                            "kind": "vector_norm",
                            "view": "bundle_view",
                            "norm": "l2",
                            "output_column": "bundle_norm",
                        }
                    },
                    "context_norm": {
                        "derive": {
                            "kind": "vector_norm",
                            "view": "context_by_subject",
                            "norm": "l2",
                            "output_column": "context_norm",
                        }
                    },
                    "joined_norms": {
                        "derive": {
                            "kind": "join_tables",
                            "sources": ["bundle_norm", "context_norm"],
                            "on": ["subject_id"],
                        }
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_phase16_view_stats_scalar_join_and_common_cli_flags(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    bundle_dir = tmp_path / "bundle_source"
    _write_matrix_bundle(bundle_dir)
    context_path = tmp_path / "inputs" / "context.parquet"
    _write_parquet(
        context_path,
        [
            {
                "id": "ctx_01",
                "subject_id": "subject_01",
                "context_id": "a",
                "label": "spyP",
                "embedding_context": [1.0, 0.0],
            },
            {
                "id": "ctx_02",
                "subject_id": "subject_01",
                "context_id": "b",
                "label": "spyP",
                "embedding_context": [3.0, 2.0],
            },
            {
                "id": "ctx_03",
                "subject_id": "subject_02",
                "context_id": "a",
                "label": "sulAp",
                "embedding_context": [10.0, 0.0],
            },
            {
                "id": "ctx_04",
                "subject_id": "subject_02",
                "context_id": "b",
                "label": "sulAp",
                "embedding_context": [14.0, 4.0],
            },
            {
                "id": "ctx_05",
                "subject_id": "subject_03",
                "context_id": "a",
                "label": "J23105",
                "embedding_context": [2.0, 2.0],
            },
            {
                "id": "ctx_06",
                "subject_id": "subject_03",
                "context_id": "b",
                "label": "J23105",
                "embedding_context": [6.0, 6.0],
            },
        ],
    )
    _write_workspace_config(workspace_dir, bundle_dir, context_path)

    preview_result = _RUNNER.invoke(
        app,
        [
            "view",
            "materialize",
            "bundle_view",
            "--workspace",
            workspace_dir.as_posix(),
            "--dry-run",
            "--json",
        ],
    )
    assert preview_result.exit_code == 0, preview_result.stdout
    preview_payload = json.loads(preview_result.stdout)
    assert preview_payload["dry_run"] is True
    assert preview_payload["artifact_kind"] == "view"
    assert preview_payload["artifact_id"] == "bundle_view"
    assert preview_payload["outputs"] == [(workspace_dir / "outputs" / "views" / "bundle_view").as_posix()]
    assert not (workspace_dir / "outputs" / "views" / "bundle_view").exists()

    for view_id in ["bundle_view", "context_view"]:
        result = _RUNNER.invoke(
            app,
            ["view", "materialize", view_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    reduce_result = _RUNNER.invoke(
        app,
        [
            "view",
            "reduce",
            "bundle_view",
            "--workspace",
            workspace_dir.as_posix(),
            "--run-id",
            "bundle_pca",
            "--dims",
            "2",
            "--reduced-view-id",
            "bundle_pca_view",
            "--json",
        ],
    )
    assert reduce_result.exit_code == 0, reduce_result.stdout

    derive_result = _RUNNER.invoke(
        app,
        ["view", "derive", "context_by_subject", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert derive_result.exit_code == 0, derive_result.stdout

    for scalar_id in ["bundle_norm", "context_norm", "joined_norms"]:
        result = _RUNNER.invoke(
            app,
            ["scalar", "derive", scalar_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    stats_result = _RUNNER.invoke(
        app,
        ["view", "stats", "bundle_view", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert stats_result.exit_code == 0, stats_result.stdout
    stats_payload = json.loads(stats_result.stdout)
    assert stats_payload["schema_version"] == "latentdna.view_stats.v1"
    assert stats_payload["artifact_kind"] == "view"
    assert stats_payload["artifact_id"] == "bundle_view"
    assert stats_payload["rows"] == 3
    assert stats_payload["dims"] == 4
    assert stats_payload["missing_values"] == 0
    assert stats_payload["mean_norm"] == pytest.approx((5.0 + 13.0 + 17.0) / 3.0)

    reduced_stats_result = _RUNNER.invoke(
        app,
        ["view", "stats", "bundle_pca_view", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert reduced_stats_result.exit_code == 0, reduced_stats_result.stdout
    reduced_stats_payload = json.loads(reduced_stats_result.stdout)
    assert reduced_stats_payload["artifact_kind"] == "reduced_view"
    assert reduced_stats_payload["artifact_id"] == "bundle_pca_view"
    assert reduced_stats_payload["dims"] == 2
    assert len(reduced_stats_payload["explained_variance_ratio"]) == 2
    assert reduced_stats_payload["source_view_id"] == "bundle_view"

    joined_table = pq.read_table(workspace_dir / "outputs" / "scalars" / "joined_norms" / "table.parquet")
    assert joined_table.column_names == ["id", "subject_id", "cohort", "bundle_norm", "label", "context_norm"]
    assert joined_table.to_pylist() == [
        {
            "id": "bundle_01",
            "subject_id": "subject_01",
            "cohort": "a",
            "bundle_norm": pytest.approx(5.0),
            "label": "spyP",
            "context_norm": pytest.approx(np.sqrt(5.0)),
        },
        {
            "id": "bundle_02",
            "subject_id": "subject_02",
            "cohort": "a",
            "bundle_norm": pytest.approx(13.0),
            "label": "sulAp",
            "context_norm": pytest.approx(np.sqrt(148.0)),
        },
        {
            "id": "bundle_03",
            "subject_id": "subject_03",
            "cohort": "b",
            "bundle_norm": pytest.approx(17.0),
            "label": "J23105",
            "context_norm": pytest.approx(np.sqrt(32.0)),
        },
    ]

    quiet_result = _RUNNER.invoke(
        app,
        [
            "sample",
            "build",
            "bundle_all",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "bundle_view",
            "--strategy",
            "all",
            "--quiet",
        ],
    )
    assert quiet_result.exit_code == 0, quiet_result.stdout
    quiet_output = quiet_result.stdout.strip()
    assert quiet_output.startswith("ok: sample_set:bundle_all")
    assert "artifact_kind" not in quiet_output
    assert "workspace_id" not in quiet_output
    assert len(quiet_output.splitlines()) == 1
