"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_trait_axis_projection.py

Trait-axis projection scalar contract tests.

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

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.io.json_io import read_json
from dnadesign.latentdna.src.scalars.build import build_scalar_artifact
from dnadesign.latentdna.src.services.scalar_service import build_scalar
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _write_workspace_config(workspace_dir: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "trait_axis_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor": {
                        "kind": "parquet",
                        "path": "inputs/anchor.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "views": {
                    "trait_view": {
                        "source": "anchor",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "trait_axis_demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_view_artifact(
    workspace_dir: Path,
    *,
    rows: list[dict[str, object]],
    matrix: np.ndarray,
) -> None:
    view_dir = workspace_dir / "outputs" / "views" / "trait_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray(matrix, dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(json.dumps({"params": {"record_key": "id"}}, indent=2), encoding="utf-8")


def _trait_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "low_1",
            "subject_id": "low_1",
            "cohort": "fit",
            "source": "fit_source",
            "trait_value": 0.0,
            "parent_id": None,
            "parent_key": None,
        },
        {
            "id": "low_2",
            "subject_id": "low_2",
            "cohort": "fit",
            "source": "fit_source",
            "trait_value": 1.0,
            "parent_id": None,
            "parent_key": None,
        },
        {
            "id": "high_1",
            "subject_id": "high_1",
            "cohort": "fit",
            "source": "fit_source",
            "trait_value": 10.0,
            "parent_id": None,
            "parent_key": None,
        },
        {
            "id": "high_2",
            "subject_id": "high_2",
            "cohort": "fit",
            "source": "fit_source",
            "trait_value": 11.0,
            "parent_id": None,
            "parent_key": None,
        },
        {
            "id": "reference_1",
            "subject_id": "reference_1",
            "cohort": "reference",
            "source": "reference_source",
            "trait_value": None,
            "parent_id": None,
            "parent_key": None,
        },
        {
            "id": "mutant_1",
            "subject_id": "mutant_1",
            "cohort": "sensitivity",
            "source": "sensitivity_source",
            "trait_value": None,
            "parent_id": "high_1",
            "parent_key": "rt_parent",
        },
    ]


def _trait_matrix() -> np.ndarray:
    return np.asarray(
        [
            [-1.0, 0.0],
            [-0.9, 0.1],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )


def _row_params() -> dict[str, object]:
    return {
        "candidate_id_column": "id",
        "metadata_columns": ["subject_id", "cohort", "source"],
        "candidate_views": ["trait_view"],
        "axes": [
            {
                "trait_id": "demo_trait",
                "axis_id": "demo_trait_axis",
                "label": "Demo trait",
                "source_value_column": "trait_value",
                "primary_endpoint_definition_id": "quantile_0_50",
                "fit_population": {
                    "population_id": "axis_fit_rows",
                    "role": "fit",
                    "where": [{"column": "cohort", "equals": "fit"}, {"column": "trait_value", "finite": True}],
                },
                "endpoint_groups": {
                    "method": "quantile",
                    "value_column": "trait_value",
                    "low_quantile": 0.50,
                    "high_quantile": 0.50,
                    "min_low_rows": 2,
                    "min_high_rows": 2,
                },
                "endpoint_sensitivity": {
                    "enabled": True,
                    "endpoint_definitions": ["min_max", "quantile_0_50"],
                },
                "score_populations": [
                    {
                        "population_id": "fit_rows",
                        "role": "fit",
                        "where": [{"column": "cohort", "equals": "fit"}],
                    },
                    {
                        "population_id": "reference_rows",
                        "role": "reference",
                        "where": [{"column": "cohort", "equals": "reference"}],
                    },
                    {
                        "population_id": "sensitivity_rows",
                        "role": "sensitivity",
                        "where": [{"column": "cohort", "equals": "sensitivity"}],
                    },
                ],
                "parent_key": "parent_key",
                "parent_candidate_id_column": "parent_id",
            }
        ],
    }


def test_trait_axis_projection_rows_scores_populations_and_declares_sidecars(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    _write_view_artifact(workspace_dir, rows=_trait_rows(), matrix=_trait_matrix())

    build_scalar(
        workspace_dir,
        "demo_trait_axis_rows",
        builder_kind="trait_axis_projection_rows",
        params=_row_params(),
    )

    scalar_dir = workspace_dir / "outputs" / "scalars" / "demo_trait_axis_rows"
    table = pq.read_table(scalar_dir / "table.parquet").to_pylist()
    manifest = read_json(scalar_dir / "manifest.json")

    assert manifest["outputs"] == [
        {"path": "table.parquet", "media_type": "application/x-parquet"},
        {"path": "provenance.json", "media_type": "application/json"},
        {"path": "fitted_axes.parquet", "media_type": "application/x-parquet"},
    ]
    assert manifest["stats"]["normalization_policy"] == "row_l2"
    assert manifest["stats"]["configured_trait_count"] == 1
    assert manifest["stats"]["configured_view_count"] == 1
    assert {row["endpoint_definition_id"] for row in table} == {"min_max", "quantile_0_50"}
    assert {row["population_role"] for row in table} == {"fit", "reference", "sensitivity"}
    assert len(table) == 12

    fit_quantile = [
        row for row in table if row["population_role"] == "fit" and row["endpoint_definition_id"] == "quantile_0_50"
    ]
    assert {row["endpoint_group"] for row in fit_quantile} == {"low", "high"}
    assert min(row["axis_projection"] for row in fit_quantile if row["endpoint_group"] == "high") > 0.9
    assert max(row["axis_projection"] for row in fit_quantile if row["endpoint_group"] == "low") < -0.9

    mutant = next(
        row for row in table if row["candidate_id"] == "mutant_1" and row["endpoint_definition_id"] == "quantile_0_50"
    )
    assert mutant["parent_candidate_id"] == "high_1"
    assert mutant["axis_delta"] < 0
    assert mutant["orthogonal_delta"] > 0

    axes = pq.read_table(scalar_dir / "fitted_axes.parquet").to_pylist()
    assert {(row["view_id"], row["endpoint_definition_id"]) for row in axes} == {
        ("trait_view", "min_max"),
        ("trait_view", "quantile_0_50"),
    }
    assert read_json(scalar_dir / "provenance.json")["builder_kind"] == "trait_axis_projection_rows"


def test_trait_axis_projection_rows_rejects_sensitivity_fit_overlap(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    rows = _trait_rows()
    rows[0]["cohort"] = "sensitivity"
    _write_view_artifact(workspace_dir, rows=rows, matrix=_trait_matrix())
    params = _row_params()
    axis = dict(params["axes"][0])  # type: ignore[index]
    axis["score_populations"] = [
        {"population_id": "sensitivity_rows", "role": "sensitivity", "where": [{"column": "cohort", "equals": "fit"}]}
    ]
    params["axes"] = [axis]

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    with pytest.raises(ContractViolationError, match="sensitivity.*overlaps fit"):
        build_scalar_artifact(
            context,
            scalar_id="bad_trait_axis_rows",
            builder_kind="trait_axis_projection_rows",
            params=params,
        )


@pytest.mark.parametrize(
    ("selector", "match"),
    [
        ({"column": "trait_value", "finite": "true"}, "requires a boolean value"),
        ({"column": "source", "in_values": "fit_source"}, "requires a non-empty sequence"),
        ({"column": "source", "regex": "["}, "invalid regex"),
    ],
)
def test_trait_axis_projection_rows_rejects_malformed_selectors(
    tmp_path: Path,
    selector: dict[str, object],
    match: str,
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    _write_view_artifact(workspace_dir, rows=_trait_rows(), matrix=_trait_matrix())
    params = _row_params()
    axis = dict(params["axes"][0])  # type: ignore[index]
    axis["fit_population"] = {
        "population_id": "axis_fit_rows",
        "role": "fit",
        "where": [selector],
    }
    params["axes"] = [axis]

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    with pytest.raises(ContractViolationError, match=match):
        build_scalar_artifact(
            context,
            scalar_id="bad_trait_axis_rows",
            builder_kind="trait_axis_projection_rows",
            params=params,
        )


def test_trait_axis_projection_rows_rejects_missing_metadata_columns(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    _write_view_artifact(workspace_dir, rows=_trait_rows(), matrix=_trait_matrix())
    params = _row_params()
    params["metadata_columns"] = ["cohort", "missing_metadata"]

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    with pytest.raises(ContractViolationError, match="metadata_columns are missing"):
        build_scalar_artifact(
            context,
            scalar_id="bad_trait_axis_rows",
            builder_kind="trait_axis_projection_rows",
            params=params,
        )


def test_trait_axis_projection_marks_missing_parent_mapping_invalid(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    rows = _trait_rows()
    rows[-1]["parent_id"] = None
    rows[-1]["parent_key"] = "rt_parent"
    _write_view_artifact(workspace_dir, rows=rows, matrix=_trait_matrix())

    build_scalar(
        workspace_dir,
        "demo_trait_axis_rows",
        builder_kind="trait_axis_projection_rows",
        params=_row_params(),
    )

    scalar_dir = workspace_dir / "outputs" / "scalars" / "demo_trait_axis_rows"
    table = pq.read_table(scalar_dir / "table.parquet").to_pylist()
    manifest = read_json(scalar_dir / "manifest.json")
    invalid_mutants = [row for row in table if row["candidate_id"] == "mutant_1"]

    assert len(invalid_mutants) == 2
    assert {row["row_status"] for row in invalid_mutants} == {"invalid"}
    assert {row["row_status_reason"] for row in invalid_mutants} == {"missing_parent_candidate_id_column=parent_id"}
    assert {row["axis_delta"] for row in invalid_mutants} == {None}
    assert manifest["stats"]["invalid_skipped_row_count"] == 2

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    summary_artifact = build_scalar_artifact(
        context,
        scalar_id="demo_trait_axis_summary",
        builder_kind="trait_axis_projection_summary",
        params={
            "source_scalar": "demo_trait_axis_rows",
            "score_columns": ["axis_projection"],
        },
    )
    summary = pq.read_table(summary_artifact.artifact_dir / "table.parquet").to_pylist()

    assert {row["invalid_row_count"] for row in summary} == {1}
    assert {row["row_status_reasons"] for row in summary} == {"missing_parent_candidate_id_column=parent_id:1"}


def test_trait_axis_projection_summary_preserves_endpoint_stability_and_axis_concordance(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    scalar_dir = workspace_dir / "outputs" / "scalars" / "source_trait_rows"
    scalar_dir.mkdir(parents=True)
    rows = [
        {
            "candidate_id": "low_1",
            "view_id": "trait_view",
            "trait_id": "trait_a",
            "axis_id": "trait_a_axis",
            "endpoint_definition_id": "quantile_0_50",
            "primary_endpoint_definition_id": "quantile_0_50",
            "population_id": "fit_rows",
            "population_role": "fit",
            "endpoint_group": "low",
            "source_value": 0.0,
            "source_value_available": True,
            "axis_projection": -1.0,
            "endpoint_margin": -1.8,
            "row_status": "ok",
            "row_status_reason": "",
        },
        {
            "candidate_id": "low_2",
            "view_id": "trait_view",
            "trait_id": "trait_a",
            "axis_id": "trait_a_axis",
            "endpoint_definition_id": "quantile_0_50",
            "primary_endpoint_definition_id": "quantile_0_50",
            "population_id": "fit_rows",
            "population_role": "fit",
            "endpoint_group": "low",
            "source_value": 1.0,
            "source_value_available": True,
            "axis_projection": -0.8,
            "endpoint_margin": -1.5,
            "row_status": "ok",
            "row_status_reason": "",
        },
        {
            "candidate_id": "high_1",
            "view_id": "trait_view",
            "trait_id": "trait_a",
            "axis_id": "trait_a_axis",
            "endpoint_definition_id": "quantile_0_50",
            "primary_endpoint_definition_id": "quantile_0_50",
            "population_id": "fit_rows",
            "population_role": "fit",
            "endpoint_group": "high",
            "source_value": 10.0,
            "source_value_available": True,
            "axis_projection": 0.8,
            "endpoint_margin": 1.5,
            "row_status": "ok",
            "row_status_reason": "",
        },
        {
            "candidate_id": "high_2",
            "view_id": "trait_view",
            "trait_id": "trait_a",
            "axis_id": "trait_a_axis",
            "endpoint_definition_id": "quantile_0_50",
            "primary_endpoint_definition_id": "quantile_0_50",
            "population_id": "fit_rows",
            "population_role": "fit",
            "endpoint_group": "high",
            "source_value": 11.0,
            "source_value_available": True,
            "axis_projection": 1.0,
            "endpoint_margin": 1.8,
            "row_status": "ok",
            "row_status_reason": "",
        },
    ]
    sensitivity_rows = [
        {**row, "endpoint_definition_id": "min_max", "axis_projection": row["axis_projection"] * 0.9} for row in rows
    ]
    trait_b_rows = [
        {
            **row,
            "trait_id": "trait_b",
            "axis_id": "trait_b_axis",
            "endpoint_definition_id": row["endpoint_definition_id"],
        }
        for row in [*rows, *sensitivity_rows]
    ]
    pq.write_table(pa.Table.from_pylist([*rows, *sensitivity_rows, *trait_b_rows]), scalar_dir / "table.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "trait_id": "trait_a",
                    "axis_id": "trait_a_axis",
                    "view_id": "trait_view",
                    "endpoint_definition_id": "quantile_0_50",
                    "normalization_policy": "row_l2",
                    "axis_vector": [1.0, 0.0],
                },
                {
                    "trait_id": "trait_a",
                    "axis_id": "trait_a_axis",
                    "view_id": "trait_view",
                    "endpoint_definition_id": "min_max",
                    "normalization_policy": "row_l2",
                    "axis_vector": [1.0, 0.0],
                },
                {
                    "trait_id": "trait_b",
                    "axis_id": "trait_b_axis",
                    "view_id": "trait_view",
                    "endpoint_definition_id": "quantile_0_50",
                    "normalization_policy": "row_l2",
                    "axis_vector": [0.0, 1.0],
                },
            ]
        ),
        scalar_dir / "fitted_axes.parquet",
    )
    (scalar_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "scalar_table",
                "artifact_id": "source_trait_rows",
                "workspace_id": "trait_axis_demo",
                "created_at": "2026-05-27T00:00:00+00:00",
                "tool_version": "test",
                "command": "scalar build",
                "status": "ok",
                "params": {"builder_kind": "trait_axis_projection_rows"},
                "outputs": [
                    {"path": "table.parquet", "media_type": "application/x-parquet"},
                    {"path": "fitted_axes.parquet", "media_type": "application/x-parquet"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="trait_axis_summary",
        builder_kind="trait_axis_projection_summary",
        params={
            "source_scalar": "source_trait_rows",
            "score_columns": ["axis_projection", "endpoint_margin"],
            "concordance": {
                "enabled": True,
                "compare_trait_ids": [["trait_a", "trait_b"]],
            },
        },
    )

    summary = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    primary = next(
        row for row in summary if row["trait_id"] == "trait_a" and row["endpoint_definition_id"] == "quantile_0_50"
    )
    assert primary["total_scored_rows"] == 4
    assert primary["fit_population_row_count"] == 4
    assert primary["low_endpoint_row_count"] == 2
    assert primary["high_endpoint_row_count"] == 2
    assert primary["axis_projection_spearman"] == pytest.approx(1.0)
    assert primary["axis_projection_endpoint_effect"] > 0
    assert primary["axis_vector_primary_concordance"] == pytest.approx(1.0)
    assert primary["axis_vector_primary_angle"] == pytest.approx(0.0)
    assert primary["invalid_row_count"] == 0
    assert primary["row_status_reasons"] == ""
    sensitivity = next(
        row for row in summary if row["trait_id"] == "trait_a" and row["endpoint_definition_id"] == "min_max"
    )
    assert sensitivity["axis_projection_primary_spearman"] == pytest.approx(1.0)
    assert sensitivity["axis_projection_effect_sign_matches_primary"] is True
    assert sensitivity["axis_vector_primary_concordance"] == pytest.approx(1.0)

    concordance = pq.read_table(artifact.artifact_dir / "axis_concordance.parquet").to_pylist()
    assert concordance == [
        {
            "view_id": "trait_view",
            "endpoint_definition_id": "quantile_0_50",
            "left_trait_id": "trait_a",
            "right_trait_id": "trait_b",
            "left_axis_id": "trait_a_axis",
            "right_axis_id": "trait_b_axis",
            "axis_concordance": 0.0,
            "axis_angle": pytest.approx(np.pi / 2.0),
            "normalization_policy": "row_l2",
        }
    ]
    assert any(
        input_ref.kind == "scalar_sidecar" and input_ref.path.name == "fitted_axes.parquet"
        for input_ref in artifact.inputs
    )
    assert artifact.outputs == [("axis_concordance.parquet", "application/x-parquet")]

    (artifact.artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "latentdna.manifest.v1",
                "artifact_kind": "scalar_table",
                "artifact_id": "trait_axis_summary",
                "workspace_id": "trait_axis_demo",
                "created_at": "2026-05-27T00:00:00+00:00",
                "tool_version": "test",
                "command": "scalar build",
                "status": "ok",
                "params": {"builder_kind": "trait_axis_projection_summary"},
                "outputs": [
                    {"path": "table.parquet", "media_type": "application/x-parquet"},
                    {"path": "axis_concordance.parquet", "media_type": "application/x-parquet"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    concordance_artifact = build_scalar_artifact(
        context,
        scalar_id="trait_axis_concordance",
        builder_kind="scalar_sidecar_table",
        params={
            "source_scalar": "trait_axis_summary",
            "sidecar": "axis_concordance.parquet",
            "required_columns": ["view_id", "endpoint_definition_id", "axis_concordance"],
        },
    )
    concordance_table = pq.read_table(concordance_artifact.artifact_dir / "table.parquet").to_pylist()
    assert concordance_table == concordance
    assert concordance_artifact.inputs[0].kind == "scalar_manifest"
    assert concordance_artifact.inputs[1].kind == "scalar_sidecar"
