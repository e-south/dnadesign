"""Scalar builder coverage for notebook-facing cohort inventory tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.scalars.build import (
    _cosine_distance_correlation,
    _reference_neighbor_metrics,
    build_scalar_artifact,
)
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_workspace_config(
    workspace_dir: Path,
    *,
    anchor_record_key: str = "id",
    anchor_subject_key: str = "subject_id",
    context_record_key: str = "id",
    context_subject_key: str = "subject_id",
) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "dataset_overview_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_60bp": {
                        "kind": "parquet",
                        "path": "inputs/anchor.parquet",
                        "record_key": anchor_record_key,
                        "subject_key": anchor_subject_key,
                    },
                    "full_context_1kb": {
                        "kind": "parquet",
                        "path": "inputs/context.parquet",
                        "record_key": context_record_key,
                        "subject_key": context_subject_key,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_margin_workspace_config(workspace_dir: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "margin_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_60bp": {
                        "kind": "parquet",
                        "path": "inputs/anchor.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "views": {
                    "degenerate_view": {
                        "source": "anchor_60bp",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "7b", "family": "intermediate_embedding", "scope": "anchor_60bp"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _base_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "row_1",
            "subject_id": "row_1",
            "usr_label__primary": "dense_row_1",
            "densegen__plan": "background_only__sig35=f",
            "densegen__required_regulators": [],
        },
        {
            "id": "row_2",
            "subject_id": "row_2",
            "usr_label__primary": "dense_row_2",
            "densegen__plan": "ethanol__sig35=d",
            "densegen__required_regulators": ["baeR"],
        },
        {
            "id": "row_3",
            "subject_id": "row_3",
            "usr_label__primary": "dense_row_3",
            "densegen__plan": "ciprofloxacin__sig35=e",
            "densegen__required_regulators": ["lexA"],
        },
        {
            "id": "row_4",
            "subject_id": "row_4",
            "usr_label__primary": "dense_row_4",
            "densegen__plan": "ethanol_ciprofloxacin__sig35=b",
            "densegen__required_regulators": ["baeR", "lexA"],
        },
        {
            "id": "row_5",
            "subject_id": "row_5",
            "usr_label__primary": "dense_row_5",
            "densegen__plan": "ethanol__sig35=c",
            "densegen__required_regulators": ["cpxR"],
        },
        {
            "id": "control",
            "subject_id": "control",
            "usr_label__primary": "J23105",
            "densegen__plan": None,
            "densegen__required_regulators": [],
        },
    ]


def test_dataset_overview_builds_dimension_panels_with_shared_denominator(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    rows = _base_rows()
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    _write_source(workspace_dir / "inputs" / "context.parquet", rows)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="dataset_overview_counts",
        builder_kind="dataset_overview",
        params={"source_ids": ["anchor_60bp", "full_context_1kb"]},
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    dimensions = {row["dimension"] for row in table}

    assert artifact.stats["denominator"] == 6
    assert dimensions == {"provenance", "generation_plan", "sig35_variant"}
    assert "anchor_60bp" not in dimensions
    assert "full_context_1kb" not in dimensions
    assert {row["dimension_label"] for row in table} == {"Provenance", "Generation plan", "Sigma-35 variant"}

    for dimension in dimensions:
        dimension_rows = [row for row in table if row["dimension"] == dimension]
        assert sum(int(row["count"]) for row in dimension_rows) == 6
        assert pytest.approx(sum(float(row["fraction"]) for row in dimension_rows), rel=0, abs=1e-9) == 1.0


def test_dataset_overview_rejects_mismatched_partitions_across_sources(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    anchor_rows = _base_rows()
    context_rows = _base_rows()
    context_rows[1] = {
        **context_rows[1],
        "densegen__plan": "background_only__sig35=d",
    }
    _write_source(workspace_dir / "inputs" / "anchor.parquet", anchor_rows)
    _write_source(workspace_dir / "inputs" / "context.parquet", context_rows)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)

    with pytest.raises(ContractViolationError, match="matching cohort partitions across sources"):
        build_scalar_artifact(
            context,
            scalar_id="dataset_overview_counts",
            builder_kind="dataset_overview",
            params={"source_ids": ["anchor_60bp", "full_context_1kb"]},
        )


def test_dataset_overview_accepts_aligned_subject_populations_with_distinct_record_keys(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(
        workspace_dir,
        anchor_record_key="id",
        anchor_subject_key="subject_id",
        context_record_key="context_id",
        context_subject_key="anchor_id",
    )
    anchor_rows = _base_rows()
    context_rows = [
        {
            "context_id": f"context_{index}",
            "anchor_id": row["id"],
            "usr_label__primary": row["usr_label__primary"],
            "densegen__plan": row["densegen__plan"],
            "densegen__required_regulators": row["densegen__required_regulators"],
        }
        for index, row in enumerate(anchor_rows, start=1)
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", anchor_rows)
    _write_source(workspace_dir / "inputs" / "context.parquet", context_rows)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="dataset_overview_counts",
        builder_kind="dataset_overview",
        params={"source_ids": ["anchor_60bp", "full_context_1kb"]},
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    assert sum(int(row["count"]) for row in table if row["dimension"] == "provenance") == len(anchor_rows)


def test_cohort_similarity_margin_returns_nan_for_degenerate_standardized_view(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "bg_1", "subject_id": "bg_1", "design_family": "background_only", "embedding": [1.0, 1.0]},
        {"id": "bg_2", "subject_id": "bg_2", "design_family": "background_only", "embedding": [1.0, 1.0]},
        {"id": "eth_1", "subject_id": "eth_1", "design_family": "ethanol", "embedding": [1.0, 1.0]},
        {"id": "eth_2", "subject_id": "eth_2", "design_family": "ethanol", "embedding": [1.0, 1.0]},
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.ones((4, 2), dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="degenerate_design_margins",
        builder_kind="cohort_similarity_margin",
        params={
            "view_id": "degenerate_view",
            "cohort_column": "design_family",
            "leave_one_out": True,
            "margin_pairs": [
                {
                    "target_values": ["ethanol"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_ethanol_vs_background",
                }
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    values = np.asarray(table["synthetic_margin_ethanol_vs_background"].to_pylist(), dtype=np.float32)
    assert np.isnan(values).all()
    assert artifact.stats["synthetic_margin_ethanol_vs_background_degenerate_reference"] is True


def test_cohort_similarity_margin_honors_sample_scope(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {"id": "bg_1", "subject_id": "bg_1", "design_family": "background_only", "embedding": [0.0, 1.0]},
        {"id": "bg_2", "subject_id": "bg_2", "design_family": "background_only", "embedding": [0.1, 0.9]},
        {"id": "eth_1", "subject_id": "eth_1", "design_family": "ethanol", "embedding": [1.0, 0.0]},
        {"id": "eth_2", "subject_id": "eth_2", "design_family": "ethanol", "embedding": [0.9, 0.1]},
        {"id": "cip_1", "subject_id": "cip_1", "design_family": "ciprofloxacin", "embedding": [0.6, 0.4]},
        {"id": "cip_2", "subject_id": "cip_2", "design_family": "ciprofloxacin", "embedding": [0.4, 0.6]},
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )

    sample_rows = rows[:4]
    sample_dir = workspace_dir / "outputs" / "samples" / "ethanol_background_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sample_rows), sample_dir / "rows.parquet")

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="sampled_design_margins",
        builder_kind="cohort_similarity_margin",
        params={
            "view_id": "degenerate_view",
            "sample_id": "ethanol_background_sample",
            "cohort_column": "design_family",
            "leave_one_out": True,
            "margin_pairs": [
                {
                    "target_values": ["ethanol"],
                    "control_values": ["background_only"],
                    "output_column": "synthetic_margin_ethanol_vs_background",
                }
            ],
        },
    )

    table = pq.read_table(artifact.artifact_dir / "table.parquet")
    assert table.num_rows == 4
    assert set(table["id"].to_pylist()) == {"bg_1", "bg_2", "eth_1", "eth_2"}
    assert artifact.stats["sample_id"] == "ethanol_background_sample"
    assert any(input_ref.kind == "sample_set" for input_ref in artifact.inputs)


def test_reference_alignment_summary_fails_when_required_references_are_missing(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "bg_1",
            "subject_id": "bg_1",
            "design_family": "background_only",
            "usr_label__primary": "dense_bg",
            "embedding": [0.0, 1.0],
        },
        {
            "id": "eth_1",
            "subject_id": "eth_1",
            "design_family": "ethanol",
            "usr_label__primary": "dense_eth",
            "embedding": [1.0, 0.0],
        },
        {
            "id": "cip_1",
            "subject_id": "cip_1",
            "design_family": "ciprofloxacin",
            "usr_label__primary": "dense_cip",
            "embedding": [0.5, 0.5],
        },
        {
            "id": "spyp",
            "subject_id": "spyp",
            "design_family": "control",
            "usr_label__primary": "spyp",
            "embedding": [0.8, 0.2],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)

    with pytest.raises(ContractViolationError, match="requires carried SpyP and SulA rows"):
        build_scalar_artifact(
            context,
            scalar_id="reference_alignment_summary_metrics",
            builder_kind="reference_alignment_summary",
            params={"candidates": [{"view_id": "degenerate_view"}]},
        )


def test_sigma35_ordinal_audit_emits_confidence_intervals_for_spearman_summary_rows(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_margin_workspace_config(workspace_dir)
    rows = [
        {
            "id": "a_f",
            "subject_id": "a_f",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "f",
            "spacer_length": 16,
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "id": "a_d",
            "subject_id": "a_d",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "d",
            "spacer_length": 16,
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "id": "a_e",
            "subject_id": "a_e",
            "design_family": "background_only",
            "design_regulator_composition": "background_only",
            "sig35_variant": "e",
            "spacer_length": 16,
            "embedding": [-1.0, 0.0, 0.0],
        },
        {
            "id": "b_f",
            "subject_id": "b_f",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "f",
            "spacer_length": 17,
            "embedding": [0.9, 0.1, 0.0],
        },
        {
            "id": "b_d",
            "subject_id": "b_d",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "d",
            "spacer_length": 17,
            "embedding": [0.1, 0.9, 0.0],
        },
        {
            "id": "b_e",
            "subject_id": "b_e",
            "design_family": "ethanol",
            "design_regulator_composition": "cpxR_only",
            "sig35_variant": "e",
            "spacer_length": 17,
            "embedding": [-0.9, 0.1, 0.0],
        },
    ]
    _write_source(workspace_dir / "inputs" / "anchor.parquet", rows)
    view_dir = workspace_dir / "outputs" / "views" / "degenerate_view"
    view_dir.mkdir(parents=True, exist_ok=True)
    np.save(view_dir / "matrix.npy", np.asarray([row["embedding"] for row in rows], dtype=np.float32))
    pq.write_table(pa.Table.from_pylist(rows), view_dir / "rows.parquet")
    (view_dir / "manifest.json").write_text(
        json.dumps({"params": {"record_key": "id"}}, indent=2),
        encoding="utf-8",
    )

    study_inputs_dir = workspace_dir / "study_inputs"
    study_inputs_dir.mkdir(parents=True, exist_ok=True)
    (study_inputs_dir / "sig35_order.yaml").write_text(
        yaml.safe_dump(
            {
                "source": "test fixture",
                "exploratory": False,
                "order": [
                    {"variant_id": "f", "sequence": "TTGACA", "rank": 1},
                    {"variant_id": "d", "sequence": "TTTACA", "rank": 2},
                    {"variant_id": "e", "sequence": "TAGACA", "rank": 3},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    artifact = build_scalar_artifact(
        context,
        scalar_id="sigma35_ordinal_audit_metrics",
        builder_kind="sigma35_ordinal_audit",
        params={
            "candidates": [{"view_id": "degenerate_view"}],
            "sig35_order_path": "study_inputs/sig35_order.yaml",
            "bootstrap_iterations": 20,
            "permutations": 20,
            "balance_columns": ["design_family", "spacer_length"],
        },
    )

    rows_table = pq.read_table(artifact.artifact_dir / "table.parquet").to_pylist()
    ci_metric_ids = {
        "sig35_ordinal_spearman",
        "sig35_balanced_ordinal_spearman",
        "sig35_within_family_mean_spearman",
        "sig35_within_regulator_mean_spearman",
    }
    for row in rows_table:
        if row["metric_id"] not in ci_metric_ids:
            continue
        assert row["ci_lower"] is not None
        assert row["ci_upper"] is not None


def test_reference_neighbor_metrics_pool_row_level_censored_ranks_across_tasks() -> None:
    rows = [
        {"usr_label__primary": "dense_eth_1", "design_family": "ethanol"},
        {"usr_label__primary": "dense_eth_2", "design_family": "ethanol"},
        {"usr_label__primary": "dense_cipro_1", "design_family": "ciprofloxacin"},
        {"usr_label__primary": "spyp", "design_family": "control"},
        {"usr_label__primary": "sulAp", "design_family": "control"},
    ]
    indices = np.asarray(
        [
            [3, 1, 2, 4],
            [2, 4, 3, 0],
            [0, 1, 3, 4],
            [0, 1, 2, 4],
            [0, 1, 2, 3],
        ],
        dtype=np.int64,
    )

    metrics = _reference_neighbor_metrics(
        rows,
        indices,
        label_column="design_family",
        ethanol_values={"ethanol"},
        cipro_values={"ciprofloxacin"},
    )

    assert metrics["reference_in_knn_rate"] == pytest.approx(1.0)
    assert metrics["reference_neighbor_topk_censored_rank_median"] == pytest.approx(3.0)


def test_cosine_distance_correlation_returns_nan_for_single_row_inputs() -> None:
    left = np.asarray([[1.0, 0.0]], dtype=np.float32)
    right = np.asarray([[0.0, 1.0]], dtype=np.float32)

    result = _cosine_distance_correlation(left, right)

    assert np.isnan(result)
