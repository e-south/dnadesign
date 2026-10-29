"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_multiview_extensions_workflow.py

Workflow tests for matrix-bundle views and extended derived/scalar.

Module Author(s): Eric J. South
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
from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.views.derive import _project_matrix_to_reference_rows, _write_block_normalized_matrix

_RUNNER = CliRunner()


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_matrix_bundle(bundle_dir: Path, *, include_manifest: bool = True) -> None:
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
    if include_manifest:
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


def _manual_block_normalized(matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.float64)
    mean = source.mean(axis=0, keepdims=True)
    std = source.std(axis=0, keepdims=True)
    zero_mask = np.asarray(std[0] <= 1e-8, dtype=bool)
    std[:, zero_mask] = 1.0
    working = np.asarray((source - mean) / std, dtype=np.float32)
    working[:, zero_mask] = 0.0
    norms = np.linalg.norm(working, axis=1, keepdims=True)
    return np.asarray(working / np.maximum(norms, 1e-8), dtype=np.float32)


def test_block_normalized_concatenate_uses_stable_stats_for_large_offsets() -> None:
    row_count = 20_000
    positions = np.arange(row_count, dtype=np.float32)
    matrix = np.column_stack(
        [
            np.float32(1_000_000.0) + np.mod(positions, 7.0),
            np.float32(1_000_000.0) + np.mod(positions * 3.0, 11.0),
        ]
    ).astype(np.float32)
    output = np.zeros_like(matrix)

    _write_block_normalized_matrix(
        matrix,
        output,
        column_start=0,
        column_stop=matrix.shape[1],
        projection_indices=None,
        center=True,
        scale=True,
        nonfinite_policy="error",
        zero_variance_policy="drop_or_zero",
        zero_row_policy="zero",
    )

    np.testing.assert_allclose(output, _manual_block_normalized(matrix), rtol=2e-5, atol=2e-5)


def test_block_normalized_concatenate_projection_indices_preserve_reference_row_axis() -> None:
    matrix = np.asarray(
        [
            [10.0, 1.0, 0.0],
            [20.0, 2.0, 1.0],
            [30.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    output = np.zeros_like(matrix)
    projection_indices = np.asarray([2, 0, 1], dtype=np.int64)

    _write_block_normalized_matrix(
        matrix,
        output,
        column_start=0,
        column_stop=matrix.shape[1],
        projection_indices=projection_indices,
        center=True,
        scale=True,
        nonfinite_policy="error",
        zero_variance_policy="drop_or_zero",
        zero_row_policy="zero",
    )

    np.testing.assert_allclose(output, _manual_block_normalized(matrix)[projection_indices])


def test_concatenate_uses_manifest_declared_record_key() -> None:
    reference_rows = pa.table(
        {
            "construct_subject__id": ["subject_a", "subject_b"],
            "parent_sequence_id": ["forward_a", "forward_b"],
        }
    )
    candidate_rows = pa.table(
        {
            "construct_subject__id": ["subject_b", "subject_a"],
            "parent_sequence_id": ["reverse_b", "reverse_a"],
        }
    )
    candidate_matrix = np.asarray([[2.0, 20.0], [1.0, 10.0]], dtype=np.float32)

    projected = _project_matrix_to_reference_rows(
        reference_rows,
        candidate_rows,
        candidate_matrix,
        input_view="reverse_construct_view",
        reference_manifest={"params": {"record_key": "construct_subject__id"}},
        candidate_manifest={"params": {"record_key": "construct_subject__id"}},
    )

    np.testing.assert_array_equal(projected, np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32))


def _write_workspace_config(workspace_dir: Path, bundle_dir: Path, context_path: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "latentdna_ext_demo", "output_root": "./outputs"},
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
                        "metadata_include": ["cohort"],
                    },
                    "context_source": {
                        "kind": "parquet",
                        "path": context_path.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "context_key": "context_id",
                        "metadata_include": ["label"],
                    },
                },
                "metadata": {"include": []},
                "views": {
                    "bundle_view": {
                        "source": "bundle_source",
                        "vector": {"kind": "bundle_matrix"},
                        "coordinate_space_id": "bundle_space",
                        "tags": {"model": "bundle"},
                        "role": "primary",
                    },
                    "bundle_norm": {
                        "derive": {"kind": "normalize", "view": "bundle_view", "method": "l2"},
                        "coordinate_space_id": "bundle_space",
                        "tags": {"operation": "normalize"},
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
                    "bundle_reduced": {
                        "derive": {"kind": "apply_reducer", "view": "bundle_view", "reducer": "bundle_pca"},
                        "coordinate_space_id": "bundle_space_pca",
                        "tags": {"operation": "apply_reducer"},
                        "role": "primary",
                    },
                    "bundle_reduced_norm": {
                        "derive": {"kind": "normalize", "view": "bundle_reduced", "method": "l2"},
                        "coordinate_space_id": "bundle_space_pca",
                        "tags": {"operation": "normalize"},
                        "role": "primary",
                    },
                    "bundle_concat": {
                        "derive": {
                            "kind": "concatenate",
                            "inputs": ["bundle_reduced", "bundle_reduced_norm"],
                        },
                        "coordinate_space_id": "bundle_concat_space",
                        "tags": {"operation": "concatenate"},
                        "role": "primary",
                    },
                    "bundle_block_concat": {
                        "derive": {
                            "kind": "block_normalized_concatenate",
                            "inputs": ["bundle_view", "bundle_norm"],
                        },
                        "coordinate_space_id": "bundle_block_concat_space",
                        "tags": {"operation": "block_normalized_concatenate"},
                        "role": "primary",
                    },
                },
                "scalars": {
                    "bundle_norm_scalar": {"derive": {"kind": "vector_norm", "view": "bundle_view", "norm": "l2"}},
                    "bundle_norm_selected": {
                        "derive": {
                            "kind": "select_columns",
                            "source": "bundle_norm_scalar",
                            "columns": ["bundle_norm_scalar"],
                        }
                    },
                    "bundle_norm_renamed": {
                        "derive": {
                            "kind": "rename_columns",
                            "source": "bundle_norm_selected",
                            "renames": {"bundle_norm_scalar": "bundle_norm_value"},
                        }
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_matrix_bundle_and_extended_derive_flow(tmp_path: Path) -> None:
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
        ],
    )
    _write_workspace_config(workspace_dir, bundle_dir, context_path)

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
            "--json",
        ],
    )
    assert reduce_result.exit_code == 0, reduce_result.stdout

    validate_result = _RUNNER.invoke(
        app,
        ["validate", "workspace", "--workspace", workspace_dir.as_posix(), "--deep", "--json"],
    )
    assert validate_result.exit_code == 0, validate_result.stdout
    validate_payload = json.loads(validate_result.stdout)
    assert validate_payload["status"] == "ok"
    assert any(
        detail["view_id"] == "bundle_view" and detail["declaration_kind"] == "source_backed"
        for detail in validate_payload["view_details"]
    )

    for view_id in [
        "bundle_norm",
        "context_by_subject",
        "bundle_reduced",
        "bundle_reduced_norm",
        "bundle_concat",
        "bundle_block_concat",
    ]:
        result = _RUNNER.invoke(
            app,
            ["view", "derive", view_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    for scalar_id in ["bundle_norm_scalar", "bundle_norm_selected", "bundle_norm_renamed"]:
        result = _RUNNER.invoke(
            app,
            ["scalar", "derive", scalar_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    inspect_result = _RUNNER.invoke(
        app,
        ["inspect", "source", "bundle_source", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert inspect_result.exit_code == 0, inspect_result.stdout
    inspect_payload = json.loads(inspect_result.stdout)
    assert "bundle_matrix" in inspect_payload["data"]["vector_columns"]

    outputs = workspace_dir / "outputs"
    bundle_matrix = np.load(outputs / "views" / "bundle_view" / "matrix.npy")
    assert bundle_matrix.shape == (3, 4)

    normalized_matrix = np.load(outputs / "views" / "bundle_norm" / "matrix.npy")
    assert normalized_matrix.shape == (3, 4)
    assert np.allclose(np.linalg.norm(normalized_matrix, axis=1), 1.0)

    aggregated_rows = pq.read_table(outputs / "views" / "context_by_subject" / "rows.parquet").to_pylist()
    aggregated_matrix = np.load(outputs / "views" / "context_by_subject" / "matrix.npy")
    assert [row["subject_id"] for row in aggregated_rows] == ["subject_01", "subject_02"]
    assert np.allclose(aggregated_matrix, np.asarray([[2.0, 1.0], [12.0, 2.0]], dtype=np.float32))

    reduced_matrix = np.load(outputs / "views" / "bundle_reduced" / "matrix.npy")
    assert reduced_matrix.shape == (3, 2)

    concatenated_matrix = np.load(outputs / "views" / "bundle_concat" / "matrix.npy")
    assert concatenated_matrix.shape == (3, 4)
    block_concatenated_matrix = np.load(outputs / "views" / "bundle_block_concat" / "matrix.npy")
    assert block_concatenated_matrix.shape == (3, 8)
    assert np.allclose(np.linalg.norm(block_concatenated_matrix[:, :4], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(block_concatenated_matrix[:, 4:], axis=1), 1.0)
    assert np.allclose(block_concatenated_matrix[:, :4], _manual_block_normalized(bundle_matrix))
    assert np.allclose(block_concatenated_matrix[:, 4:], _manual_block_normalized(normalized_matrix))
    renamed_table = pq.read_table(outputs / "scalars" / "bundle_norm_renamed" / "table.parquet")
    assert "bundle_norm_value" in renamed_table.column_names


def test_matrix_bundle_source_rejects_ambiguous_matrix_payloads(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    bundle_dir = tmp_path / "bundle_source"
    _write_matrix_bundle(bundle_dir)
    np.savez(bundle_dir / "matrix.npz", matrix=np.ones((3, 4), dtype=np.float32))
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
            }
        ],
    )
    _write_workspace_config(workspace_dir, bundle_dir, context_path)

    result = _RUNNER.invoke(
        app,
        ["inspect", "source", "bundle_source", "--workspace", workspace_dir.as_posix(), "--json"],
    )

    assert result.exit_code != 0
    assert "ambiguous matrix payloads" in result.stdout


def test_concatenate_rejects_non_unique_reference_join_keys() -> None:
    reference_rows = pa.Table.from_pylist(
        [
            {"id": "row_a"},
            {"id": "row_a"},
        ]
    )
    candidate_rows = pa.Table.from_pylist(
        [
            {"id": "row_a"},
            {"id": "row_b"},
        ]
    )
    candidate_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ContractViolationError, match="concatenate reference rows are non-unique"):
        _project_matrix_to_reference_rows(
            reference_rows,
            candidate_rows,
            candidate_matrix,
            input_view="candidate_view",
        )


def test_concatenate_tries_alternate_join_keys_when_first_candidate_is_invalid() -> None:
    reference_rows = pa.Table.from_pylist(
        [
            {"construct__anchor_id": "anchor_a", "id": "context_a"},
            {"construct__anchor_id": "anchor_b", "id": "context_b"},
        ]
    )
    candidate_rows = pa.Table.from_pylist(
        [
            {"construct__anchor_id": "duplicate_anchor", "id": "anchor_a"},
            {"construct__anchor_id": "duplicate_anchor", "id": "anchor_b"},
        ]
    )
    candidate_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    projected = _project_matrix_to_reference_rows(
        reference_rows,
        candidate_rows,
        candidate_matrix,
        input_view="candidate_view",
    )

    assert np.allclose(projected, candidate_matrix)


def test_matrix_bundle_view_requires_manifest(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    bundle_dir = tmp_path / "bundle_source"
    _write_matrix_bundle(bundle_dir, include_manifest=False)
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
            }
        ],
    )
    _write_workspace_config(workspace_dir, bundle_dir, context_path)

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "bundle_view", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code != 0
    assert "manifest.json" in materialize_result.stdout
